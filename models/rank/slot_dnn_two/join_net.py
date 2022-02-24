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


class JoinDNNLayer(nn.Layer):
    def __init__(self,
                 dict_dim,
                 emb_dim,
                 slot_num,
                 layer_sizes,
                 sync_mode=None):
        super(JoinDNNLayer, self).__init__()
        self.sync_mode = sync_mode
        self.dict_dim = dict_dim
        self.emb_dim = emb_dim
        self.slot_num = slot_num
        self.layer_sizes = layer_sizes
        self._init_range = 0.2

        self.entry = paddle.distributed.ShowClickEntry("show", "click")

        sizes = [emb_dim * slot_num] + self.layer_sizes + [1]
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
                    initializer=paddle.nn.initializer.Normal(std=scales[i])))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)
                self._mlp_layers.append(act)

    def forward(self, show, click, slot_inputs):
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
            emb.stop_gradient = True
            self.inference_feed_vars.append(emb)

            bow = paddle.fluid.layers.sequence_pool(input=emb, pool_type='sum')
            bow.stop_gradient = True
            self.all_vars.append(bow)
            #paddle.fluid.layers.Print(bow)
            bows.append(bow)

            cvm = fluid.layers.continuous_value_model(bow, show_clk, True)
            cvm.stop_gradient = True
            self.all_vars.append(cvm)
            cvms.append(cvm)

        y_dnn = paddle.concat(x=cvms, axis=1)
        y_dnn.stop_gradient = True
        self.all_vars.append(y_dnn)

        y_dnn = paddle.static.nn.data_norm(
            input=y_dnn,
            name="data_norm",
            epsilon=1e-4,
            param_attr={
                "batch_size": 1e4,
                "batch_sum": 0.0,
                "batch_square": 1e4
            })
        self.all_vars.append(y_dnn)

        for n_layer in self._mlp_layers:
            y_dnn = n_layer(y_dnn)
            self.all_vars.append(y_dnn)

        self.predict = F.sigmoid(paddle.clip(y_dnn, min=-15.0, max=15.0))
        self.all_vars.append(self.predict)
        return self.predict
