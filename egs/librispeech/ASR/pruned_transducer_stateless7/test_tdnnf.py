#!/usr/bin/env python3
# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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


"""
To run this file, do:

    cd icefall/egs/librispeech/ASR
    python ./pruned_transducer_stateless7/test_tdnnf.py
"""
import torch
from tdnnf import (
    FactorizedTdnnLayer,
    FactorizedTdnnModel,
    OrthonormalLinear,
    _constrain_orthonormal_internal,
)


def test_constrain_orthonormal_internal():
    w = torch.empty(5, 5)
    w = torch.nn.init.xavier_uniform_(w)
    t = [torch.mm(w, w.t())]
    for i in range(100):
        w = _constrain_orthonormal_internal(w)
        t.append(torch.mm(w, w.t()))
    print("---test_constrain_orthonormal_internal---")
    print(t[-1])
    # You can see t[-1] is almost an identity matrix


def test_orthonormal_linear():
    layer = OrthonormalLinear(2, 5, 3)
    with torch.no_grad():
        for i in range(20):
            layer.constrain_orthonormal()
    w = layer.conv.weight
    print("---test_orthonormal_linear---")
    print(w)
    w = w.reshape(w.shape[0], -1)
    print(torch.mm(w, w.t()))  # check that it is almost an identity matrix

    x = torch.rand(10, 2, 20)  # (N, C, T)
    y = layer(x)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == 5  # out_channels
    assert y.shape[2] == 20 - 2  # there are no paddings and the kerne size is 3

    print(x.shape, y.shape)


def test_factorized_tdnnf_layer1():
    layer = FactorizedTdnnLayer(
        dim=16,
        bottleneck_dim=4,
        kernel_size=3,
        subsampling_factor=1,
    )
    x = torch.rand(5, 16, 20)  # (N, C, T)
    y = layer(x)
    assert y.shape == (5, 16, 18)


def test_factorized_tdnnf_layer2():
    layer = FactorizedTdnnLayer(
        dim=16,
        bottleneck_dim=4,
        kernel_size=1,
        subsampling_factor=3,
    )
    for T in range(20, 50):
        x = torch.rand(5, 16, T)  # (N, C, T)
        y = layer(x)
        assert y.shape == (5, 16, ((T - 1) // 3 + 1)), y.shape


def test_factorized_tdnnf_model():
    model = FactorizedTdnnModel(
        feat_dim=80,
        hidden_dim=1024,
        bottleneck_dim=128,
        num_layers=12,
        subsampling_at_layer=4,
    )
    # For the first 3 layers, each layer has a left context 1
    # For the fourth layer, it has not left ocntext
    # For the last 8 layers, each layer has a left context 1 (after subsampling)
    # So the total left context is (1+1+1) + 0 + 1*3*8 = 3 + 24 = 27
    # The right context is also 27

    x = torch.rand(10, 300 + 27 + 27, 80)
    x_lens = torch.tensor([300 + 27 + 27] * 10)
    y, y_lens = model(x, x_lens)
    print(y.shape)
    print(y_lens)

    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")


def main():
    #  test_constrain_orthonormal_internal()
    #  test_orthonormal_linear()
    #  test_factorized_tdnnf_layer1()
    #  test_factorized_tdnnf_layer2()
    test_factorized_tdnnf_model()


if __name__ == "__main__":
    main()
