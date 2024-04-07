#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
This file generates the file lexicon.txt that contains pronunciations of all
words and phrases.
"""
import argparse
import logging
from pathlib import Path

from pypinyin import phrases_dict, pinyin_dict
from tokenizer import Tokenizer


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

    parser.add_argument(
        "--heteronym-fst",
        type=str,
        default="data/heteronym.fst",
        help="""Path to save the heteronym rule fst.""",
    )
    return parser


def main():
    args = get_parser().parse_args()
    tokens = args.tokens
    tokenizer = Tokenizer(tokens)

    word_dict = pinyin_dict.pinyin_dict
    phrases = phrases_dict.phrases_dict

    if not Path(args.lexicon).is_file():
        logging.info(f"Creating {args.lexicon}")
        with open(args.lexicon, "w", encoding="utf-8") as f:
            for key in word_dict:
                if not (0x4E00 <= key <= 0x9FFF):
                    continue

                w = chr(key)

                # 1 to remove the initial sil
                # :-1 to remove the final eos
                tokens = tokenizer.text_to_tokens(w)[1:-1]

                tokens = " ".join(tokens)
                f.write(f"{w} {tokens}\n")

            for key in phrases:
                # 1 to remove the initial sil
                # :-1 to remove the final eos
                tokens = tokenizer.text_to_tokens(key)[1:-1]
                tokens = " ".join(tokens)
                f.write(f"{key} {tokens}\n")
    else:
        logging.info(f"Skipping generating {args.lexicon}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
