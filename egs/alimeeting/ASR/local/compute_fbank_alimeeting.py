#!/usr/bin/env python3
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


"""
This file computes fbank features of the aishell dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from lhotse import ChunkedLilcomHdf5Writer, CutSet, Fbank, FbankConfig
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def compute_fbank_alimeeting(num_mel_bins: int = 80):
    src_dir = Path("data/manifests/alimeeting")
    output_dir = Path("data/fbank")
    num_jobs = min(15, os.cpu_count())

    dataset_parts = (
        "train",
        "eval",
        "test",
    )
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        suffix="jsonl.gz",
    )
    assert manifests is not None

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    with get_executor() as ex:  # Initialize the executor only once.
        for partition, m in manifests.items():
            if (output_dir / f"cuts_{partition}.json.gz").is_file():
                logging.info(f"{partition} already exists - skipping.")
                continue
            logging.info(f"Processing {partition}")
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )
            if "train" in partition:
                cut_set = (
                    cut_set
                    + cut_set.perturb_speed(0.9)
                    + cut_set.perturb_speed(1.1)
                )
            cur_num_jobs = num_jobs if ex is None else 80
            cur_num_jobs = min(cur_num_jobs, len(cut_set))

            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/feats_{partition}",
                # when an executor is specified, make more partitions
                num_jobs=cur_num_jobs,
                executor=ex,
                storage_type=ChunkedLilcomHdf5Writer,
            )

            logging.info("About splitting cuts into smaller chunks")
            cut_set = cut_set.trim_to_supervisions(
                keep_overlapping=False,
                min_duration=None,
            )
            cut_set.to_json(output_dir / f"cuts_{partition}.json.gz")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-mel-bins",
        type=int,
        default=80,
        help="""The number of mel bins for Fbank""",
    )

    return parser.parse_args()


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    compute_fbank_alimeeting(num_mel_bins=args.num_mel_bins)