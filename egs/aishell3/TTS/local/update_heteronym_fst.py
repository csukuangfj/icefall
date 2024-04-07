#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: Fangjun Kuang)


"""
This file generates a new heteronym fst for phrases that are not covered in
./generate_heteronym_fst.py
"""

import argparse
import logging

import pynini
from pynini import cdrewrite
from pynini.lib import utf8
from pypinyin import load_phrases_dict
from tokenizer import Tokenizer
from new_phrases import new_phrases


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--tokens",
        type=str,
        default="data/tokens.txt",
        help="""Path to vocabulary.""",
    )

    parser.add_argument(
        "--lexicon",
        type=str,
        default="data/lexicon.txt",
        help="""Path to save the generated lexicon.""",
    )
    return parser


# https://raw.githubusercontent.com/mozillazg/python-pinyin/master/pypinyin/phrases_dict.py
def main():
    args = get_parser().parse_args()
    tokenizer = Tokenizer(args.tokens)

    sigma = utf8.VALID_UTF8_CHAR.star

    load_phrases_dict(new_phrases)

    phrases_map = []

    for p in new_phrases:
        phrases_map.append([p, f"#$|{p}|$#"])

    with open(args.lexicon, "a", encoding="utf-8") as f:
        for p in new_phrases:
            tokens = tokenizer.text_to_tokens(p)[1:-1]
            tokens = " ".join(tokens)
            f.write(f"{p} {tokens}\n")

    filename = "new_heteronym.fst"
    fst = pynini.string_map(phrases_map)
    fst = fst.optimize()
    rule = cdrewrite(fst, "", "", sigma)

    logging.info(f"Saving to {filename}")
    rule.write(filename)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
