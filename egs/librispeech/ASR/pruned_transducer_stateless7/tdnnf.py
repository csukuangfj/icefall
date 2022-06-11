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

import warnings
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_interface import EncoderInterface


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


class FactorizedTdnnLayer(nn.Module):
    """
    This class implements the following topology in kaldi:
      tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1

    References:
        - http://danielpovey.com/files/2018_interspeech_tdnnf.pdf
        - ConstrainOrthonormalInternal() from
          https://github.com/kaldi-asr/kaldi/blob/master/src/nnet3/nnet-utils.cc#L982

    Note: We replace BatcNnorm with LayerNorm
    """

    def __init__(
        self,
        dim: int,
        bottleneck_dim: int,
        kernel_size: int,
        subsampling_factor: int,
        bypass_scale: float = 0.66,
    ):
        """
        Args:
          dim:
            Input and output dimension.
          bottleneck_dim:
            The bottleneck dimension.
          kernel_size:
            Kernel size used in Conv1d.
          subsampling_factor:
            Number of output frames is equal to
            num_input_frames//subsampling_factor.
          bypass_scale:
            The scale used in skip connection.
        """
        super().__init__()
        assert abs(bypass_scale) <= 1
        self.bypass_scale = bypass_scale
        self.s = subsampling_factor

        # It requires (N, C, T)
        self.linear = OrthonormalLinear(
            dim=dim,
            bottleneck_dim=bottleneck_dim,
            kernel_size=kernel_size,
        )

        # affine requires [N, C, T]
        # WARNING(fangjun): we do not use nn.Linear here
        # since we want to use `stride`
        self.affine = nn.Conv1d(
            in_channels=bottleneck_dim,
            out_channels=dim,
            kernel_size=1,
            stride=subsampling_factor,
        )

        # LayerNorm requires (N, T, C)
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:
            A tensor of shape (N, C, T)
        Returns:
          Return a tensor of shape (N, C, T)
        """
        assert x.ndim == 3

        # save it for skip connection
        input_x = x

        x = self.linear(x)  # (N, C, T)
        x = self.affine(x)  # (N, C, T)
        x = F.relu(x)  # (N, C, T)

        x = x.permute(0, 2, 1)  # (N, T, C)
        x = self.layernorm(x)  # (N, T, C)
        x = x.permute(0, 2, 1)  # (N, C, T)

        s = self.s

        if self.linear.kernel_size > 1:
            # Note: Usually kernel_size is 3 and subsampling_factor is 1
            #
            # s:-s:s means to select indexes s, 2s, 3s, 4s, ..., (n-2)s, ...,
            # up to -s. That is, first remove s elements from head and tail
            # and then subsample the remaining elements by a factor of s
            x = self.bypass_scale * input_x[:, :, s:-s:s] + x
        else:
            # Note: Usually kernel_size is 1 and subsampling_factor is 3
            x = self.bypass_scale * input_x[:, :, ::s] + x

        return x


def constrain_orthonormal_hook(model, unused_x):
    if not model.training:
        return

    model.ortho_constrain_count = (model.ortho_constrain_count + 1) % 2
    if model.ortho_constrain_count != 0:
        return

    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, "constrain_orthonormal"):
                m.constrain_orthonormal()


class FactorizedTdnnModel(EncoderInterface):
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 1024,
        bottleneck_dim: int = 128,
        num_layers: int = 12,
        subsampling_at_layer: int = 4,  # index starts from 1
        #  kernel_size_list=[3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3],
        #  subsampling_factor_list=[1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1],
    ):
        """
        Args:
          feat_dim:
            Input feature dim.
          hidden_dim:
            Input feature dim is first transformed into this dim before doing
            any processing.
          bottleneck_dim:
            Bottleneck dimension used in the OrthonormalLinear layer.
          num_layers:
            Number of FactorizedTdnnLayer to use.
          subsampling_at_layer:
            Apply subsampling with factor 3 at this layer. Layer index is
            1 based.
        """
        super().__init__()
        assert 1 <= subsampling_at_layer <= num_layers
        self.num_layers = num_layers
        self.subsampling_at_layer = subsampling_at_layer

        self.input_linear = nn.Linear(
            in_features=feat_dim,
            out_features=hidden_dim,
        )
        self.input_layernorm = nn.LayerNorm(hidden_dim)

        factorized_tdnns = []
        for i in range(num_layers):
            if i != subsampling_at_layer - 1:
                kernel_size = 3
                subsampling_factor = 1
            else:
                kernel_size = 1
                subsampling_factor = 3
            layer = FactorizedTdnnLayer(
                dim=hidden_dim,
                bottleneck_dim=bottleneck_dim,
                kernel_size=kernel_size,
                subsampling_factor=subsampling_factor,
            )

            factorized_tdnns.append(layer)

        self.factorized_tdnns = nn.Sequential(*factorized_tdnns)

        self.ortho_constrain_count = 0
        self.register_forward_pre_hook(constrain_orthonormal_hook)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A tensor of shape (batch_size, input_seq_len, num_features)
            containing the input features.
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames
            in `x` before padding.
        Returns:
          Return a tuple containing two tensors:
            - encoder_out, a tensor of (batch_size, out_seq_len, output_dim)
              containing unnormalized probabilities, i.e., the output of a
              linear layer.
            - encoder_out_lens, a tensor of shape (batch_size,) containing
              the number of frames in `encoder_out` before padding.
        """
        x = self.input_linear(x)  # (N, T, C)
        x = self.input_layernorm(x)  # (N, T, C)

        x = x.permute(0, 2, 1)  # (N, C, T)
        x = self.factorized_tdnns(x)  # (N, C, T)
        x = x.permute(0, 2, 1)  # (N, T, C)

        # For each layer before the subsampling_at_layer, T is reduced by 2.
        # Hence we have (self.subsampling_at_layer - 1)*2
        #
        # For the subsampling_at_layer, it output is (T-1)//3+1
        #
        # For the remaining layers, each layer reduces T by 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_lens = (
                ((x_lens - (self.subsampling_at_layer - 1) * 2) - 1) // 3
                + 1
                - (self.num_layers - self.subsampling_at_layer) * 2
            )

        return x, x_lens
