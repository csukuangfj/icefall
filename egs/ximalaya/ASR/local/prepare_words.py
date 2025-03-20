#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright    2021  Xiaomi Corp.        (authors: Mingshuang Luo)
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
This script takes as input words.txt without ids:
    - words_no_ids.txt
and generates the new words.txt with related ids.
    - words.txt
"""


import argparse

from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="Prepare words.txt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-file",
        default="data/lang_char/words_no_ids.txt",
        type=str,
        help="the words file without ids for WenetSpeech",
    )
    parser.add_argument(
        "--output-file",
        default="data/lang_char/words.txt",
        type=str,
        help="the words file with ids for WenetSpeech",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    words = set()

    with open(input_file, encoding="utf-8") as f:
        lines = f.readlines()

    for i in tqdm(range(len(lines))):
        line = lines[i].strip()
        words.update(line)
    words = list(words)
    words.sort()
    print(f"number of words: {len(words)}")

    add_words = ["<eps> 0", "!SIL 1", "<SPOKEN_NOISE> 2", "<UNK> 3"]
    with open(output_file, "w", encoding="utf-8") as f:
        for w in add_words:
            f.write(f"{w}\n")
        for i, w in enumerate(words, 4):
            f.write(f"{w} {i}\n")

        f.write(f"#0 {i+1}\n")
        f.write(f"<s> {i+2}\n")
        f.write(f"</s> {i+3}\n")


if __name__ == "__main__":
    main()
