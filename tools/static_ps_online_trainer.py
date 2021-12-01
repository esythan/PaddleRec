# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
from utils.static_ps.reader_helper import get_reader, get_example_num, get_file_list, get_word_num
from utils.static_ps.program_helper import get_model, get_strategy, set_dump_config
from utils.static_ps.flow_helper_accessor import *
from utils.static_ps.metric_helper import *
from utils.static_ps.time_helper import *
from utils.static_ps.common import YamlHelper, is_distributed_env, get_utils_file_path
import argparse
import time
import sys
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle
import os
import warnings
import logging
import paddle.fluid as fluid
from paddle.distributed.fleet.utils.fs import LocalFS, HDFSClient

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("PaddleRec train script")
    parser.add_argument(
        '-m',
        '--config_yaml',
        type=str,
        required=True,
        help='config file path')
    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    yaml_helper = YamlHelper()
    config = yaml_helper.load_yaml(args.config_yaml, ["table_parameters"])
    config["yaml_path"] = args.config_yaml
    config["config_abs_dir"] = args.abs_dir
    yaml_helper.print_yaml(config)
    return config


class Main(object):
    def __init__(self, config):
        self.metrics = {}
        self.config = config
        self.exe = None
        self.use_gloo = config.get("runner.use_gloo", False)
        self.reader_type = config.get("runner.reader_type", "InMemoryDataset")
        self.split_interval = config.get("runner.split_interval", 5)
        self.split_per_pass = config.get("runner.split_per_pass", 1)
        self.save_delta_frequency = config.get("runner.save_delta_frequency",
                                               6)
        self.checkpoint_per_pass = config.get("runner.checkpoint_per_pass", 6)
        self.save_first_base = config.get("runner.save_first_base", False)
        self.start_day = config.get("runner.start_day")
        self.end_day = config.get("runner.end_day")
        self.save_model_path = self.config.get("runner.model_save_path")
        self.need_train_dump = self.config.get("runner.need_train_dump", False)
        self.need_infer_dump = self.config.get("runner.need_infer_dump", False)
        if config.get("runner.fs_client.uri") is not None:
            self.hadoop_config = {}
            for key in ["uri", "user", "passwd", "hadoop_bin"]:
                self.hadoop_config[key] = config.get("runner.fs_client." + key,
                                                     "")
            self.hadoop_fs_name = self.hadoop_config.get("uri")
            self.hadoop_fs_ugi = self.hadoop_config.get(
                "user") + "," + self.hadoop_config.get("passwd")
            # prefix = "hdfs:/" if self.hadoop_fs_name.startswith("hdfs:") else "afs:/"
            # self.save_model_path = prefix + self.save_model_path.strip("/")
        else:
            self.hadoop_fs_name, self.hadoop_fs_ugi = None, None
        self.train_local = self.hadoop_fs_name is None or self.hadoop_fs_ugi is None
        if not self.train_local:
            configs = {
                "fs.default.name": self.hadoop_fs_name,
                "hadoop.job.ugi": self.hadoop_fs_ugi
            }
            self.hadoop_client = HDFSClient("$HADOOP_HOME", configs)
        else:
            self.hadoop_client = None

    def run(self):
        if self.use_gloo:
            os.environ["PADDLE_WITH_GLOO"] = "1"
            role = role_maker.PaddleCloudRoleMaker(init_gloo=True)
            fleet.init(role)
        else:
            fleet.init()
        self.init_network()
        if fleet.is_server():
            self.run_server()
        elif fleet.is_worker():
            self.run_online_worker()
            fleet.stop_worker()
            # self.record_result()
        logger.info("Run Success, Exit.")

    def init_network(self):
        self.model = get_model(self.config)
        self.input_data = self.model.create_feeds()
        self.metrics = self.model.net(self.input_data)
        self.inference_feed_vars = self.model.inference_feed_vars
        self.inference_target_var = self.model.inference_target_var
        if config.get("runner.need_prune", False):
            # DSSM prune net
            self.inference_feed_vars = self.model.prune_feed_vars
            self.inference_target_var = self.model.prune_target_var
        if config.get("runner.need_train_dump", False):
            self.train_dump_fields = self.model.train_dump_fields if hasattr(
                self.model, "train_dump_fields") else []
            self.train_dump_params = self.model.train_dump_params if hasattr(
                self.model, "train_dump_params") else []
        if config.get("runner.need_infer_dump", False):
            self.infer_dump_fields = self.model.infer_dump_fields if hasattr(
                self.model, "infer_dump_fields") else []

        thread_stat_var_names = [
            self.model.auc_stat_list[2].name, self.model.auc_stat_list[3].name
        ]
        thread_stat_var_names += [i.name for i in self.model.metric_list]
        thread_stat_var_names = list(set(thread_stat_var_names))
        self.config['stat_var_names'] = thread_stat_var_names

        self.metric_list = list(self.model.auc_stat_list) + list(
            self.model.metric_list)
        self.metric_types = ["int64"] * len(self.model.auc_stat_list) + [
            "float32"
        ] * len(self.model.metric_list)

        logger.info("cpu_num: {}".format(os.getenv("CPU_NUM")))
        self.model.create_optimizer(get_strategy(self.config))

    def run_server(self):
        logger.info("Run Server Begin")
        # fleet.init_server(config.get("runner.warmup_model_path", "./warmup"))
        fleet.init_server()
        fleet.run_server()

    def wait_and_prepare_dataset(self, day, pass_index):
        train_data_path = self.config.get("runner.train_data_dir", [])
        dataset = fluid.DatasetFactory().create_dataset(self.reader_type)
        dataset.set_use_var(self.input_data)
        dataset.set_batch_size(self.config.get('runner.train_batch_size', 1))
        dataset.set_thread(self.config.get('runner.train_thread_num', 1))
        if not self.train_local:
            dataset.set_hdfs_config(self.hadoop_fs_name, self.hadoop_fs_ugi)
            logger.info("set hadoop_fs_name = {}, fs_ugi={}".format(
                self.hadoop_fs_name, self.hadoop_fs_ugi))
        dataset.set_parse_ins_id(self.config.get("runner.parse_ins_id", False))
        dataset.set_parse_content(
            self.config.get("runner.parse_content", False))

        cur_path = []
        for i in self.online_intervals[pass_index - 1]:
            cur_path.append(train_data_path.rstrip("/") + "/" + day + "/" + i)
        global_file_list = file_ls(cur_path, self.train_local,
                                   self.hadoop_client)
        my_file_list = fleet.util.get_file_shard(global_file_list)
        logger.info("my_file_list = {}".format(my_file_list))
        dataset.set_filelist(my_file_list)

        self.pipe_command = "{} {} {}".format(
            self.config.get("runner.pipe_command"),
            config.get("yaml_path"), get_utils_file_path())
        dataset.set_pipe_command(self.pipe_command)
        dataset.load_into_memory()

        # t2 = time.time()
        # dataset.global_shuffle(fleet, 12) ##TODO: thread configure
        # t3 = time.time()
        # print('global_shuffle time cost: ', (t3-t2)/60.0)
        # shuffle_data_size = dataset.get_shuffle_data_size(fleet)
        # local_data_size = dataset.get_shuffle_data_size()
        # data_size_list = fleet.util.all_gather(local_data_size)
        # print('after global_shuffle data_size_list: ', data_size_list)
        # print('after global_shuffle data_size: ', shuffle_data_size)

        return dataset

    def wait_and_prepare_infer_dataset(self, day, pass_index):
        test_data_path = self.config.get("runner.infer_data_dir", [])
        dataset = fluid.DatasetFactory().create_dataset(self.reader_type)
        dataset.set_use_var(self.input_data)
        dataset.set_batch_size(self.config.get('runner.infer_batch_size', 1))
        dataset.set_thread(self.config.get('runner.infer_thread_num', 1))
        if not self.train_local:
            dataset.set_hdfs_config(self.hadoop_fs_name, self.hadoop_fs_ugi)
            logger.info("set hadoop_fs_name = {}, fs_ugi={}".format(
                self.hadoop_fs_name, self.hadoop_fs_ugi))
        dataset.set_parse_ins_id(self.config.get("runner.parse_ins_id", False))
        dataset.set_parse_content(
            self.config.get("runner.parse_content", False))

        cur_path = []
        for i in self.online_intervals[pass_index - 1]:
            cur_path.append(test_data_path.rstrip("/") + "/" + day + "/" + i)
        global_file_list = file_ls(cur_path, self.train_local,
                                   self.hadoop_client)
        my_file_list = fleet.util.get_file_shard(global_file_list)
        logger.info("my_file_list = {}".format(my_file_list))
        dataset.set_filelist(my_file_list)

        self.pipe_command = "{} {} {}".format(
            self.config.get("runner.pipe_command"),
            config.get("yaml_path"), get_utils_file_path())
        dataset.set_pipe_command(self.pipe_command)
        dataset.load_into_memory()
        return dataset

    def run_online_worker(self):
        logger.info("Run Online Worker Begin")
        use_cuda = int(config.get("runner.use_gpu"))
        place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
        self.exe = paddle.static.Executor(place)

        with open("./{}_worker_main_program.prototxt".format(
                fleet.worker_index()), 'w+') as f:
            f.write(str(paddle.static.default_main_program()))
        with open("./{}_worker_startup_program.prototxt".format(
                fleet.worker_index()), 'w+') as f:
            f.write(str(paddle.static.default_startup_program()))

        self.exe.run(paddle.static.default_startup_program())
        fleet.init_worker()

        self.online_intervals = get_online_pass_interval(
            self.start_day, self.end_day, self.split_interval,
            self.split_per_pass, False)
        if self.train_local and self.save_model_path and (
                not os.path.exists(self.save_model_path)):
            os.makedirs(self.save_model_path)

        last_day, last_pass, last_path, model_base_key = get_last_save_model(
            self.save_model_path, self.train_local, self.hadoop_client)
        logger.info(
            "get_last_save_model last_day = {}, last_pass = {}, last_path = {}, model_base_key = {}".
            format(last_day, last_pass, last_path, model_base_key))
        if last_day != -1 and fleet.is_first_worker():
            print("last_path:", last_path)
            load_model(last_path, 0, self.train_local, self.hadoop_client)
        fleet.barrier_worker()

        save_first_base = self.save_first_base
        day = self.start_day

        infer_first = True
        while int(day) <= int(self.end_day):
            logger.info("training a new day {}, end_day = {}".format(
                day, self.end_day))
            if last_day != -1 and int(day) < last_day:
                day = int(get_next_day(day))
                continue
            # base_model_saved = False
            save_model_path = self.save_model_path
            for pass_id in range(1, 1 + len(self.online_intervals)):
                print(last_day, day, last_pass, pass_id)
                if (last_day != -1 and int(day) == last_day) and (
                        last_pass != -1 and int(pass_id) <= last_pass):
                    # base_model_saved = True
                    continue
                if fleet.is_first_worker() and save_first_base:
                    save_first_base = False
                    last_base_day, last_base_path, tmp_xbox_base_key = \
                        get_last_save_xbox_base(save_model_path, self.train_local, self.hadoop_client)
                    logger.info(
                        "get_last_save_xbox_base, last_base_day = {}, last_base_path = {}, tmp_xbox_base_key = {}".
                        format(last_base_day, last_base_path,
                               tmp_xbox_base_key))
                    if int(day) > last_base_day:
                        model_base_key = int(time.time())
                        save_xbox_model(save_model_path, day, -1, self.exe,
                                        self.inference_feed_vars,
                                        self.inference_target_var,
                                        self.train_local, self.hadoop_client)
                        write_xbox_donefile(
                            output_path=save_model_path,
                            day=day,
                            pass_id=-1,
                            model_base_key=model_base_key,
                            train_local=self.train_local,
                            client=self.hadoop_client)
                    elif int(day) == last_base_day:
                        model_base_key = tmp_xbox_base_key
                fleet.barrier_worker()

                logger.info("training a new day = {} new pass = {}".format(
                    day, pass_id))
                logger.info("Day:{}, Pass: {}, Prepare Dataset Begin.".format(
                    day, pass_id))
                begin_train = time.time()
                begin = time.time()
                dataset = self.wait_and_prepare_dataset(day, pass_id)
                end = time.time()
                read_data_cost = (end - begin) / 60.0
                logger.info("Prepare Dataset Done, using time {} mins.".format(
                    read_data_cost))

                infer_cost = 0
                infer_metric_cost = 0
                if infer_first:
                    infer_first = False
                else:
                    # opt_info = paddle.static.default_main_program()._fleet_opt
                    # opt_info["dump_fields_path"] = "dump_data/1"
                    # opt_info["dump_fields"] = ["344", "346", "sparse_embedding_342.tmp_0", "sparse_embedding_344.tmp_0",
                    #                         "sequence_pool_338.tmp_0", "linear_11.tmp_0", "linear_7.tmp_0",
                    #                         "sigmoid_0.tmp_0"]
                    # opt_info["dump_param"] = ["linear_1.w_0", "linear_3.w_0", "linear_5.w_0",
                    #                         "linear_1.b_0", "linear_3.b_0", "linear_5.b_0"]
                    logger.info("Day:{}, Pass: {}, Infering Dataset Begin.".
                                format(day, pass_id))
                    begin = time.time()
                    self.dataset_infer_loop(dataset, day, pass_id)
                    end = time.time()
                    infer_cost = (end - begin) / 60.0
                    logger.info("Infering Dataset Done, using time {} mins.".
                                format(infer_cost))
                    begin = time.time()
                    # global_auc = get_global_auc()
                    # logger.info("pass_id %d infer global auc %f" % (pass_id, global_auc))
                    metric_str = get_global_metrics_str(fluid.global_scope(),
                                                        self.metric_list, "")
                    logger.info("Day:{}, Pass: {}, Infer Global Metric: {}".
                                format(day, pass_id, metric_str))
                    clear_metrics(fluid.global_scope(), self.metric_list,
                                  self.metric_types)
                    end = time.time()
                    infer_metric_cost = (end - begin) / 60.0
                    # exit()

                logger.info("Day:{}, Pass: {}, Training Dataset Begin.".format(
                    day, pass_id))
                begin = time.time()
                self.dataset_train_loop(dataset, day, pass_id,
                                        self.need_train_dump)
                end = time.time()
                avg_cost = get_avg_cost_mins(end - begin)
                get_max_cost_mins(end - begin)
                get_min_cost_mins(end - begin)
                train_cost = avg_cost
                logger.info("Training Dataset Done, using time {} mins.".
                            format(train_cost))

                begin = time.time()
                dataset.release_memory()
                end = time.time()
                release_cost = (end - begin) / 60.0

                begin = time.time()
                # global_auc = get_global_auc()
                # logger.info(" global auc %f" % global_auc)
                metric_str = get_global_metrics_str(fluid.global_scope(),
                                                    self.metric_list, "")
                logger.info("Day:{}, Pass: {}, Train Global Metric: {}".format(
                    day, pass_id, metric_str))
                clear_metrics(fluid.global_scope(), self.metric_list,
                              self.metric_types)
                end = time.time()
                metric_cost = (end - begin) / 60
                end_train = time.time()
                total_cost = (end_train - begin_train) / 60
                other_cost = total_cost - read_data_cost - train_cost - release_cost - metric_cost - infer_cost - infer_metric_cost
                log_str = "finished train epoch %d time cost:%s min job time cost" \
                            ":[read_data:%s min][train: %s min][release: %s min][metric: %s min][other:%s min]" \
                            "[infer:%s min][infer_metric: %s min]" \
                              % (pass_id, total_cost, read_data_cost, train_cost, release_cost, metric_cost, other_cost, infer_cost, infer_metric_cost)
                logger.info(log_str)

                if self.need_infer_dump:
                    prepare_data_start_time = time.time()
                    dump_dataset = self.wait_and_prepare_infer_dataset(day,
                                                                       pass_id)
                    prepare_data_end_time = time.time()
                    logger.info(
                        "Prepare Infer Dump Dataset Done, using time {} second.".
                        format(prepare_data_end_time -
                               prepare_data_start_time))

                    dump_start_time = time.time()
                    self.dataset_infer_loop(dump_dataset, day, pass_id, True)
                    dump_end_time = time.time()
                    logger.info(
                        "Infer Dump Dataset Done, using time {} second.".
                        format(dump_end_time - dump_start_time))

                    dump_dataset.release_memory()

                if fleet.is_first_worker():
                    if pass_id % self.checkpoint_per_pass == 0:
                        save_model(self.exe, save_model_path, day, pass_id)
                        write_model_donefile(
                            output_path=save_model_path,
                            day=day,
                            pass_id=pass_id,
                            xbox_base_key=model_base_key,
                            train_local=self.train_local,
                            client=self.hadoop_client)

                    if pass_id % self.save_delta_frequency == 0:
                        last_xbox_day, last_xbox_pass, last_xbox_path, _ = get_last_save_xbox(
                            save_model_path, self.train_local,
                            self.hadoop_client)
                        if int(day) < last_xbox_day or int(
                                day) == last_xbox_day and int(
                                    pass_id) <= last_xbox_pass:
                            log_str = "delta model exists"
                            logger.info(log_str)
                        else:
                            save_xbox_model(save_model_path, day, pass_id,
                                            self.exe, self.inference_feed_vars,
                                            self.inference_target_var,
                                            self.train_local,
                                            self.hadoop_client)  # 1 delta
                            write_xbox_donefile(
                                output_path=save_model_path,
                                day=day,
                                pass_id=pass_id,
                                model_base_key=model_base_key,
                                train_local=self.train_local,
                                client=self.hadoop_client)
                fleet.barrier_worker()

            if fleet.is_first_worker():
                last_base_day, last_base_path, last_base_key = get_last_save_xbox_base(
                    save_model_path, self.hadoop_fs_name, self.hadoop_fs_ugi)
                logger.info(
                    "one epoch finishes, get_last_save_xbox, last_base_day = {}, last_base_path = {}, last_base_key = {}".
                    format(last_base_day, last_base_path, last_base_key))
                next_day = int(get_next_day(day))
                if next_day <= last_base_day:
                    model_base_key = last_base_key
                else:
                    model_base_key = int(time.time())
                    fleet.shrink(10)
                    save_xbox_model(save_model_path, next_day, -1, self.exe,
                                    self.inference_feed_vars,
                                    self.inference_target_var,
                                    self.train_local, self.hadoop_client)
                    write_xbox_donefile(
                        output_path=save_model_path,
                        day=next_day,
                        pass_id=-1,
                        model_base_key=model_base_key,
                        train_local=self.train_local,
                        client=self.hadoop_client)
                    save_batch_model(self.exe, save_model_path, next_day)
                    write_model_donefile(
                        output_path=save_model_path,
                        day=next_day,
                        pass_id=-1,
                        xbox_base_key=model_base_key,
                        train_local=self.train_local,
                        client=self.hadoop_client)
            fleet.barrier_worker()
            day = get_next_day(day)

    def dataset_train_loop(self, cur_dataset, day, pass_index,
                           need_dump=False):
        fetch_info = [
            "Day: {} Pass: {} Var {}".format(day, pass_index, var_name)
            for var_name in self.metrics
        ]
        fetch_vars = [var for _, var in self.metrics.items()]
        print_step = int(config.get("runner.print_interval"))

        debug = config.get("runner.dataset_debug", False)
        if need_dump:
            dump_fields_dir = self.config.get("runner.train_dump_fields_dir")
            dump_fields_path = "{}/{}/{}".format(dump_fields_dir, day,
                                                 pass_index)
            dump_fields = [var.name for var in self.train_dump_fields]
            dump_params = [param.name for param in self.train_dump_params]
            set_dump_config(paddle.static.default_main_program(), {
                "dump_fields_path": dump_fields_path,
                "dump_fields": dump_fields,
                "dump_param": dump_params
            })
        print(paddle.static.default_main_program()._fleet_opt)

        self.exe.train_from_dataset(
            program=paddle.static.default_main_program(),
            dataset=cur_dataset,
            fetch_list=fetch_vars,
            fetch_info=fetch_info,
            print_period=print_step,
            debug=debug)

        if need_dump:
            set_dump_config(paddle.static.default_main_program(), {
                "dump_fields_path": "",
                "dump_fields": [],
                "dump_param": []
            })

    def dataset_infer_loop(self, cur_dataset, day, pass_index,
                           need_dump=False):
        fetch_info = [
            "Day: {} Pass: {} Var {}".format(day, pass_index, var_name)
            for var_name in self.metrics
        ]
        fetch_vars = [var for _, var in self.metrics.items()]
        print_step = int(config.get("runner.print_interval"))
        debug = config.get("runner.dataset_debug", False)
        if need_dump:
            dump_fields_dir = self.config.get("runner.infer_dump_fields_dir")
            dump_fields_path = "{}/{}/{}".format(dump_fields_dir, day,
                                                 pass_index)
            dump_fields = [var.name for var in self.infer_dump_fields]
            set_dump_config(paddle.static.default_main_program(), {
                "dump_fields_path": dump_fields_path,
                "dump_fields": dump_fields
            })
        print(paddle.static.default_main_program()._fleet_opt)

        self.exe.infer_from_dataset(
            program=paddle.static.default_main_program(),
            dataset=cur_dataset,
            fetch_list=fetch_vars,
            fetch_info=fetch_info,
            print_period=print_step,
            debug=debug)

        if need_dump:
            set_dump_config(paddle.static.default_main_program(), {
                "dump_fields_path": "",
                "dump_fields": [],
            })


if __name__ == "__main__":
    paddle.enable_static()
    config = parse_args()
    # os.environ["CPU_NUM"] = str(config.get("runner.thread_num"))
    benchmark_main = Main(config)
    benchmark_main.run()
