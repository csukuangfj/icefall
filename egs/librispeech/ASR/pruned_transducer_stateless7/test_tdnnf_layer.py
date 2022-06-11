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
    python ./pruned_transducer_stateless7/test_tdnnf_layer.py
"""
import torch
from tdnnf_layer import _constrain_orthonormal_internal
from tdnnf_layer import OrthonormalLinear


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
    layer = OrthonormalLinear(2, 3, 3)
    with torch.no_grad():
        for i in range(10):
            layer.constrain_orthonormal()
    w = layer.conv.weight
    print(w)
    w = w.reshape(w.shape[0], -1)
    print(torch.mm(w, w.t()))


def main():
    #  test_constrain_orthonormal_internal()
    test_orthonormal_linear()


if __name__ == "__main__":
    main()
