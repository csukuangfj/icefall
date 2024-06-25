# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)
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

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class Tcn(nn.Module):
    """
    See https://github.com/locuslab/TCN for what TCN is.

    See also https://arxiv.org/pdf/1803.01271

    An Empirical Evaluation of Generic Convolutional and Recurrent
    Networks for Sequence Modeling

    See also https://arxiv.org/pdf/1811.07684
    Efficient keyword spotting using dilated convolutions and gating
    """

    def __init__(
        self,
        output_dim: int,
        feat_dim: int = 40,
    ):
        super().__init__()

        hidden_dim = 64
        dilations = [1, 2, 4, 8]
        kernel_size = 8

        self.in_linear = nn.Linear(feat_dim, hidden_dim)

        self.out_linear = nn.Linear(hidden_dim, output_dim)

        d_conv_list = []
        p_conv_list = []

        for i in dilations:
            d_conv_list.append(
                nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=i,
                    groups=hidden_dim,
                )
            )

            p_conv_list.append(
                nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=1,
                )
            )
        self.d_conv_list = nn.ModuleList(d_conv_list)
        self.p_conv_list = nn.ModuleList(p_conv_list)

    # TODO(fangjun): Implement streaming_forward with cache support
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
        Returns:
          Return log_probs (the output of log_softmax) of shape (N, T, C).
        """
        x = self.in_linear(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1)  # (N, T, C) -> (N, C, T)

        # Now x is (N, C, T)

        for depthwise, pointwise in zip(self.d_conv_list, self.p_conv_list):
            src = x
            padding = depthwise.dilation[0] * (depthwise.kernel_size[0] - 1)
            x = F.pad(x, pad=(padding, 0))  # default is 0 padding
            x = depthwise(x)
            x = pointwise(x)
            x = F.relu(x)
            x = x + src

        x = x.permute(0, 2, 1)  # (N, C, T) -> (N, T, C)

        # Now x is (N, T, C)
        x = self.out_linear(x)

        x = F.log_softmax(x, dim=-1)

        return x
