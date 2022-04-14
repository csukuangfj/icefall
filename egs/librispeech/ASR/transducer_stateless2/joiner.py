# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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

import torch
import torch.nn as nn


class Joiner(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_linear = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, self.input_dim).
          decoder_out:
            Output from the decoder. Its shape is (N, U, self.input_dim).
        Returns:
          Return a tensor of shape (N, T, U, self.output_dim).
        """
        assert encoder_out.ndim == decoder_out.ndim == 3
        assert encoder_out.size(0) == decoder_out.size(0)
        assert encoder_out.size(2) == self.input_dim
        assert decoder_out.size(2) == self.input_dim

        encoder_out = encoder_out.unsqueeze(2)  # (N, T, 1, C)
        decoder_out = decoder_out.unsqueeze(1)  # (N, 1, U, C)
        x = encoder_out + decoder_out  # (N, T, U, C)

        activations = torch.tanh(x)

        logits = self.output_linear(activations)

        return logits