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

import math
import paddle
import paddle.fluid as fluid

from join_net import JoinDNNLayer


class JoinStaticModel():
    def __init__(self, config):
        self.cost = None
        self.infer_target_var = None
        self.config = config
        self._init_hyper_parameters()
        self.sync_mode = config.get("runner.sync_mode")
        self._train_program = fluid.Program()
        self._startup_program = fluid.Program()

    def _init_hyper_parameters(self):
        self.is_distributed = False
        self.distributed_embedding = False

        if self.config.get("hyper_parameters.distributed_embedding", 0) == 1:
            self.distributed_embedding = True

        self.dict_dim = self.config.get("hyper_parameters.dict_dim")
        self.emb_dim = self.config.get("hyper_parameters.emb_dim")
        self.slot_num = self.config.get("hyper_parameters.slot_num")
        self.learning_rate = float(
            self.config.get("hyper_parameters.optimizer.learning_rate"))
        self.layer_sizes = self.config.get("hyper_parameters.join_layer_sizes")
        self.adjust_ins_weight_config = {
            "need_adjust": self.config.get(
                "hyper_parameters.adjust_ins_weight.need_adjust", False),
            "nid_slot": self.config.get(
                "hyper_parameters.adjust_ins_weight.nid_slot", "0"),
            "nid_adjw_threshold": self.config.get(
                "hyper_parameters.adjust_ins_weight.nid_adjw_threshold", 1000),
            "nid_adjw_ratio": self.config.get(
                "hyper_parameters.adjust_ins_weight.nid_adjw_ratio", 20)
        }

    def create_feeds(self, is_infer=False):
        with paddle.static.program_guard(self._train_program,
                                         self._startup_program):
            slot_ids = [
                paddle.static.data(
                    name=str(i), shape=[None, 1], dtype="int64", lod_level=1)
                for i in range(2, self.slot_num + 2)
            ]

            show = paddle.static.data(
                name="show", shape=[None, 1], dtype="int64", lod_level=1)
            label = paddle.static.data(
                name="click", shape=[None, 1], dtype="int64", lod_level=1)
            ins_weight = paddle.static.data(
                name="ins_weight",
                shape=[None, 1],
                dtype="float32",
                lod_level=1)

            feeds_list = [ins_weight, show, label] + slot_ids
            return feeds_list

    def net(self, input, is_infer=False):
        with paddle.static.program_guard(self._train_program,
                                         self._startup_program):
            self.ins_weight = input[0]
            self.show_input = input[1]
            self.label_input = input[2]
            self.slot_inputs = input[3:]
            # paddle.fluid.layers.Print(self.ins_weight, message="origin ins weight")

            dnn_model = JoinDNNLayer(
                self.dict_dim,
                self.emb_dim,
                self.slot_num,
                self.layer_sizes,
                sync_mode=self.sync_mode,
                adjust_ins_weight_config=self.adjust_ins_weight_config)

            self.predict, adjust_ins_weight = dnn_model.forward(
                self.show_input, self.label_input, self.ins_weight,
                self.slot_inputs)

            # paddle.fluid.layers.Print(adjust_ins_weight, message="adjust ins weight")

            # self.all_vars = input + dnn_model.all_vars
            self.all_vars = dnn_model.all_vars

            predict_2d = paddle.concat(
                x=[1 - self.predict, self.predict], axis=1)
            #label_int = paddle.cast(self.label, 'int64')

            auc, batch_auc_var, auc_stat_list = paddle.static.auc(
                input=predict_2d, label=self.label_input, slide_steps=0)
            metric_list = fluid.contrib.layers.ctr_metric_bundle(
                self.predict,
                fluid.layers.cast(
                    x=self.label_input, dtype='float32'))

            self.thread_stat_var_names = [
                auc_stat_list[2].name, auc_stat_list[3].name
            ]
            self.thread_stat_var_names += [i.name for i in metric_list]
            self.thread_stat_var_names = list(set(self.thread_stat_var_names))

            self.metric_list = list(auc_stat_list) + list(metric_list)
            self.metric_types = ["int64"] * len(
                auc_stat_list) + ["float32"] * len(metric_list)

            self.inference_feed_vars = dnn_model.inference_feed_vars
            self.inference_target_var = self.predict

            if is_infer:
                fetch_dict = {'auc': auc}
                return fetch_dict
            cost = paddle.nn.functional.log_loss(
                input=self.predict,
                label=paddle.cast(self.label_input, "float32"))
            cost = paddle.multiply(cost, adjust_ins_weight)
            avg_cost = paddle.mean(x=cost)
            # avg_cost = paddle.fluid.layers.Print(avg_cost, message="cost join")
            self._cost = avg_cost
            fetch_dict = {
                'predict': self.predict,
                'cost': avg_cost,
                'auc': auc
            }
            self.metrics = fetch_dict
            return fetch_dict

    def infer_net(self, input):
        return self.net(input, is_infer=True)
