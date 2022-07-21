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


class AutoIntLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field, num_heads,
                 interacting_layers):
        super(AutoIntLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.dense_emb_dim = self.sparse_feature_dim
        self.sparse_num_field = sparse_num_field
        self.num_heads = num_heads
        self.interacting_layers = interacting_layers
        self.init_value_ = 0.1
        use_sparse = True
        if paddle.is_compiled_with_npu():
            use_sparse = False

        # sparse coding
        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=use_sparse,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(self.sparse_feature_dim)))))

        # dense coding
        self.dense_w = paddle.create_parameter(
            shape=[1, self.dense_feature_dim, self.dense_emb_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(
                mean=0.0,
                std=self.init_value_ /
                math.sqrt(float(self.sparse_feature_dim))))

        # multi head attention
        self.multi_head_attention = paddle.nn.MultiHeadAttention(
            self.sparse_feature_dim, self.num_heads)

        # residual network weight
        self.residual_w = paddle.create_parameter(
            shape=[self.sparse_feature_dim, self.sparse_feature_dim],
            dtype="float32",
            default_initializer=paddle.nn.initializer.TruncatedNormal(
                mean=0.0,
                std=self.init_value_ /
                math.sqrt(float(self.sparse_feature_dim))))
        inter_out_dim = (self.sparse_num_field + self.dense_feature_dim
                         ) * self.sparse_feature_dim
        self.dnn_layer = paddle.nn.Linear(
            in_features=inter_out_dim,
            out_features=1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    std=1.0 / math.sqrt(float(inter_out_dim)))))

    def normalize(self, x):
        mean = paddle.mean(x, axis=-1)
        sub = paddle.subtract(x, paddle.unsqueeze(mean, axis=2))
        out = paddle.nn.functional.normalize(sub, axis=-1)
        return out

    def forward(self, sparse_inputs, dense_inputs):
        sparse_inputs_concat = paddle.concat(sparse_inputs, axis=1)
        sparse_embeddings = self.embedding(sparse_inputs_concat)
        dense_inputs_re = paddle.unsqueeze(dense_inputs, axis=2)
        dense_embeddings = paddle.multiply(dense_inputs_re, self.dense_w)
        feat_embeddings = paddle.concat([sparse_embeddings, dense_embeddings],
                                        1)
        inter_input = feat_embeddings
        for _ in range(self.interacting_layers):
            # mha
            mha_out = self.multi_head_attention(inter_input)
            # residual
            res_out = paddle.matmul(inter_input, self.residual_w)
            inter_out = paddle.add(mha_out, res_out)
            inter_out = paddle.nn.functional.relu(inter_out)
            # normalize
            inter_out = self.normalize(inter_out)
            inter_input = inter_out

        dnn_input = paddle.flatten(inter_out, start_axis=1)
        dnn_out = self.dnn_layer(dnn_input)
        pred = paddle.nn.functional.sigmoid(dnn_out)

        return pred
