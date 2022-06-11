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

# Content in this file is copied and modified from
# https://github.com/kaldi-asr/kaldi/blob/pybind11/egs/aishell/s10/chain/tdnnf_layer.py

import torch
import torch.nn as nn


def _constrain_orthonormal_internal(M):
    """
    Refer to
        void ConstrainOrthonormalInternal(BaseFloat scale, CuMatrixBase<BaseFloat> *M)
    from
        https://github.com/kaldi-asr/kaldi/blob/master/src/nnet3/nnet-utils.cc#L982

    Also, refer to the following paper:
    https://www.danielpovey.com/files/2018_interspeech_tdnnf.pdf

    Note that we always use the **floating** case.
    """

    # We'd like to enforce the rows of M to be orthonormal.
    # define P = M M^T.  If P is unit then M has orthonormal rows.
    assert M.ndim == 2

    num_rows = M.size(0)
    num_cols = M.size(1)

    assert num_rows <= num_cols

    # P = M * M^T
    P = torch.mm(M, M.t())
    P_PT = torch.mm(P, P.t())

    trace_P = torch.trace(P)
    trace_P_P = torch.trace(P_PT)

    scale = torch.sqrt(trace_P_P / trace_P)

    ratio = trace_P_P * num_rows / (trace_P * trace_P)
    assert ratio > 0.99

    update_speed = 0.125

    if ratio > 1.02:
        update_speed *= 0.5
        if ratio > 1.1:
            update_speed *= 0.5

    identity = torch.eye(num_rows, dtype=P.dtype, device=P.device)
    P = P - scale * scale * identity

    alpha = update_speed / (scale * scale)
    M = M - 4 * alpha * torch.mm(P, M)
    return M


class OrthonormalLinear(nn.Module):
    def __init__(self, dim: int, bottleneck_dim: int, kernel_size: int):
        """
        Args:
          dim:
            Input dimension.
          bottleneck_dim:
            Output dimension.
          kernel_size:
            Kernel size of Conv1d.
        """
        super().__init__()
        # WARNING(fangjun): kaldi uses [-1, 0] for the first linear layer
        # and [0, 1] for the second affine layer;
        # we use [-1, 0, 1] for the first linear layer if time_stride == 1

        self.kernel_size = kernel_size

        # conv requires [N, C, T]
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=bottleneck_dim,
            kernel_size=kernel_size,
            bias=False,
        )  # no paddings

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, C, T).
        Returns:
          Return a tensor of shape (N, C, T)
        """
        assert x.ndim == 3
        x = self.conv(x)
        return x

    def constrain_orthonormal(self) -> None:
        """Make the weight of self.conv to be orthonormal.

        It should be called in a torch.no_grad() context.
        """
        state_dict = self.conv.state_dict()
        w = state_dict["weight"]
        # w is of shape [out_channels, in_channels, kernel_size]
        out_channels = w.size(0)
        in_channels = w.size(1)
        kernel_size = w.size(2)

        w = w.reshape(out_channels, -1)

        num_rows = w.size(0)
        num_cols = w.size(1)

        need_transpose = False
        if num_rows > num_cols:
            w = w.t()
            need_transpose = True

        w = _constrain_orthonormal_internal(w)

        if need_transpose:
            w = w.t()

        w = w.reshape(out_channels, in_channels, kernel_size)

        state_dict["weight"] = w
        self.conv.load_state_dict(state_dict)
