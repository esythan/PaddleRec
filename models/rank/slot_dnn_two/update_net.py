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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
import paddle.fluid as fluid


class UpdateDNNLayer(nn.Layer):
    def __init__(self,
                 dict_dim,
                 emb_dim,
                 slot_num,
                 layer_sizes,
                 sync_mode=None,
                 adjust_ins_weight_config=None):
        super(UpdateDNNLayer, self).__init__()
        self.sync_mode = sync_mode
        self.dict_dim = dict_dim
        self.emb_dim = emb_dim
        self.slot_num = slot_num
        self.layer_sizes = layer_sizes
        self._init_range = 0.2
        self.adjust_ins_weight_config = adjust_ins_weight_config
        self.need_adjust = adjust_ins_weight_config.get("need_adjust")
        self.nid_slot = adjust_ins_weight_config.get("nid_slot")
        self.nid_adjw_threshold = adjust_ins_weight_config.get(
            "nid_adjw_threshold")
        self.nid_adjw_ratio = adjust_ins_weight_config.get("nid_adjw_ratio")

        self.entry = paddle.distributed.ShowClickEntry("show", "click")

        sizes = [(emb_dim - 2) * slot_num] + self.layer_sizes + [1]
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        scales = []
        for i in range(len(sizes[:-1])):
            scales.append(self._init_range / (sizes[i]**0.5))
        self._mlp_layers = []
        for i in range(len(layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    learning_rate=1.0,
                    initializer=paddle.nn.initializer.Normal(std=scales[i])),
                # initializer=paddle.nn.initializer.Constant(value=0.0001)),
                bias_attr=paddle.ParamAttr(
                    learning_rate=1.0,
                    initializer=paddle.nn.initializer.Normal(std=scales[i])))
            # initializer=paddle.nn.initializer.Constant(value=0.0001)))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)
                self._mlp_layers.append(act)

    def forward(self, show, click, ins_weight, slot_inputs):
        self.all_vars = []
        bows = []
        cvms = []
        self.inference_feed_vars = []
        show_clk = fluid.layers.cast(
            fluid.layers.concat(
                [show, click], axis=1), dtype='float32')
        show_clk.stop_gradient = True
        for s_input in slot_inputs:
            emb = paddle.static.nn.sparse_embedding(
                input=s_input,
                size=[self.dict_dim, self.emb_dim],
                padding_idx=0,
                entry=self.entry,
                param_attr=paddle.ParamAttr(name="embedding"))
            self.inference_feed_vars.append(emb)

            if self.need_adjust and s_input.name == self.nid_slot:
                # paddle.fluid.layers.Print(emb, message="nid emb")
                nid_show, _ = paddle.split(
                    emb, num_or_sections=[1, self.emb_dim - 1], axis=-1)
                # paddle.fluid.layers.Print(nid_show, message="nid show")
                init_weight = fluid.layers.fill_constant_batch_size_like(
                    input=ins_weight,
                    shape=[-1, 1],
                    dtype="float32",
                    value=1.0)
                weight = paddle.log(
                    math.e + (self.nid_adjw_threshold - nid_show) /
                    self.nid_adjw_threshold * self.nid_adjw_ratio)
                # paddle.fluid.layers.Print(weight, message="ins weight in net")
                weight = paddle.where(nid_show >= 0 and
                                      nid_show < self.nid_adjw_threshold,
                                      weight, init_weight)
                ins_weight = paddle.where(weight > ins_weight, weight,
                                          ins_weight)
                # paddle.fluid.layers.Print(ins_weight, message="adjust ins weight in net")
            ins_weight.stop_gradient = True

            bow = paddle.fluid.layers.sequence_pool(input=emb, pool_type='sum')
            self.all_vars.append(bow)
            # bow = paddle.fluid.layers.Print(bow, summarize=-1)
            bows.append(bow)

            cvm = fluid.layers.continuous_value_model(bow, show_clk, False)
            self.all_vars.append(cvm)
            # cvm = paddle.fluid.layers.Print(cvm, summarize=-1)
            cvms.append(cvm)

        y_dnn = paddle.concat(x=cvms, axis=1)
        self.all_vars.append(y_dnn)
        # y_dnn = paddle.fluid.layers.Print(y_dnn, summarize=-1)

        for n_layer in self._mlp_layers:
            y_dnn = n_layer(y_dnn)
            # y_dnn = paddle.fluid.layers.Print(y_dnn, summarize=-1)
            self.all_vars.append(y_dnn)

        self.predict = F.sigmoid(paddle.clip(y_dnn, min=-15.0, max=15.0))
        self.all_vars.append(self.predict)
        return self.predict, ins_weight
