#!/usr/bin/env python3
#
# Copyright 2021-2022 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Liyong Guo,
#                                                 Quandong Wang,
#                                                 Zengwei Yao)
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
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sentencepiece as spm
import torch
import torch.nn as nn
from aishell import AishellAsrDataModule
from beam_search import greedy_search_batch, modified_beam_search
from lhotse import set_caching_enabled
from lhotse.cut import Cut
from train_bbpe_cr_ctc import add_model_arguments, get_model, get_params

from icefall import smart_byte_decode
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)

LOG_EPS = math.log(1e-10)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - modified_beam_search
        """,
    )

    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="""Maximum number of symbols per frame.
        Used only when --decoding-method is greedy_search""",
    )

    parser.add_argument(
        "--skip-scoring",
        type=str2bool,
        default=False,
        help="""Skip scoring, but still save the ASR output (for eval sets).""",
    )

    add_model_arguments(parser)

    return parser


def get_decoding_params() -> AttributeDict:
    """Parameters for decoding."""
    params = AttributeDict(
        {
            "frame_shift_ms": 10,
            "search_beam": 20,  # for k2 fsa composition
            "output_beam": 8,  # for k2 fsa composition
            "min_active_states": 30,
            "max_active_states": 10000,
            "use_double_scores": True,
            "beam": 4,  # for prefix-beam-search
        }
    )
    return params


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    bpe_model: Optional[spm.SentencePieceProcessor],
    batch: dict,
) -> Dict[str, List[List[str]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:
    - key: It indicates the setting used for decoding. For example,
           if no rescoring is used, the key is the string `no_rescore`.
           If LM rescoring is used, the key is the string `lm_scale_xxx`,
           where `xxx` is the value of `lm_scale`. An example key is
           `lm_scale_0.7`
    - value: It contains the decoding result. `len(value)` equals to
             batch size. `value[i]` is the decoding result for the i-th
             utterance in the given batch.

    Args:
      model:
        The neural model.
      bpe_model:
        The BPE model. Used only when params.decoding_method is ctc-decoding.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict. Note: If it decodes to nothing, then return None.
    """
    device = params.device
    feature = batch["inputs"]
    assert feature.ndim == 3
    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    encoder_out, encoder_out_lens = model.forward_encoder(feature, feature_lens)

    if params.decoding_method == "greedy_search" and params.max_sym_per_frame == 1:
        hyp_tokens = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
        )

        hyps = []
        for hyp in bpe_model.decode(hyp_tokens):
            hyps.append(smart_byte_decode(hyp).split())

        key = "greedy-search"
        return {key: hyps}
    elif params.decoding_method == "modified_beam_search":
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
        )
        hyps = []
        for hyp in bpe_model.decode(hyp_tokens):
            hyps.append(smart_byte_decode(hyp).split())

        key = "modified-beam-search"
        return {key: hyps}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    bpe_model: Optional[spm.SentencePieceProcessor],
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      bpe_model:
        The BPE model. Used only when params.decoding_method is ctc-decoding.
    Returns:
      Return a dict, whose key may be "no-rescore" if no LM rescoring
      is used, or it may be "lm_scale_0.7" if LM rescoring is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            bpe_model=bpe_model,
            batch=batch,
        )

        for name, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                ref_words = "".join(ref_text.split())
                this_batch.append((cut_id, ref_words, hyp_words))

            results[name].extend(this_batch)

        num_cuts += len(texts)

        if batch_idx % 100 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def save_asr_output(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    """
    Save text produced by ASR.
    """
    for key, results in results_dict.items():

        recogs_filename = params.res_dir / f"recogs-{test_set_name}-{params.suffix}.txt"

        results = sorted(results)
        store_transcripts(filename=recogs_filename, texts=results)

        logging.info(f"The transcripts are stored in {recogs_filename}")


def save_wer_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    if params.decoding_method in (
        "attention-decoder-rescoring-with-ngram",
        "whole-lattice-rescoring",
    ):
        # Set it to False since there are too many logs.
        enable_log = False
    else:
        enable_log = True

    test_set_wers = dict()
    for key, results in results_dict.items():
        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = params.res_dir / f"errs-{test_set_name}-{params.suffix}.txt"
        with open(errs_filename, "w", encoding="utf8") as fd:
            wer = write_error_stats(
                fd, f"{test_set_name}_{key}", results, enable_log=enable_log
            )
            test_set_wers[key] = wer

        logging.info(f"Wrote detailed error stats to {errs_filename}")

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])

    wer_filename = params.res_dir / f"wer-summary-{test_set_name}-{params.suffix}.txt"

    with open(wer_filename, "w", encoding="utf8") as fd:
        print("settings\tWER", file=fd)
        for key, val in test_set_wers:
            print(f"{key}\t{val}", file=fd)

    s = f"\nFor {test_set_name}, WER of different settings are:\n"
    note = f"\tbest for {test_set_name}"
    for key, val in test_set_wers:
        s += f"{key}\t{val}{note}\n"
        note = ""
    logging.info(s)


@torch.no_grad()
def main():
    parser = get_parser()
    AishellAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)

    params = get_params()
    # add decoding params
    params.update(get_decoding_params())
    params.update(vars(args))

    # enable AudioCache
    set_caching_enabled(True)  # lhotse

    assert params.decoding_method in (
        "greedy_search",
        "modified_beam_search",
    ), params.decoding_method

    params.res_dir = params.exp_dir / params.decoding_method

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}_avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}_avg-{params.avg}"

    if params.use_averaged_model:
        params.suffix += "_use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    params.device = device

    logging.info(f"Device: {device}")
    logging.info(params)

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    num_classes = sp.get_piece_size()

    params.vocab_size = num_classes
    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = 0
    params.eos_id = 1
    params.sos_id = 1

    logging.info("About to create model")
    model = get_model(params)

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )

    model.to(device)
    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    aishell = AishellAsrDataModule(args)

    def remove_short_utt(c: Cut):
        T = ((c.num_frames - 7) // 2 + 1) // 2
        if T <= 0:
            logging.warning(
                f"Exclude cut with ID {c.id} from decoding, num_frames : {c.num_frames}."
            )
        return T > 0

    dev_cuts = aishell.valid_cuts()
    dev_cuts = dev_cuts.filter(remove_short_utt)
    dev_dl = aishell.valid_dataloaders(dev_cuts)

    test_cuts = aishell.test_cuts()
    test_cuts = test_cuts.filter(remove_short_utt)
    test_dl = aishell.test_dataloaders(test_cuts)

    test_sets = ["dev", "test"]
    test_dls = [dev_dl, test_dl]

    for test_set, test_dl in zip(test_sets, test_dls):
        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            bpe_model=sp,
        )

        save_asr_output(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
        )

        if not params.skip_scoring:
            save_wer_results(
                params=params,
                test_set_name=test_set,
                results_dict=results_dict,
            )

    logging.info("Done!")


if __name__ == "__main__":
    main()
