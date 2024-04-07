#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
This file generates the file lexicon.txt that contains pronunciations of all
words and phrases.
"""

import functools
import logging
from pathlib import Path

import pynini
from pynini import cdrewrite
from pynini.lib import utf8
from pypinyin import phrases_dict


def main():
    sigma = utf8.VALID_UTF8_CHAR.star

    phrases = phrases_dict.phrases_dict

    if True:
        phrases_map = []
        for p in phrases:
            phrases_map.append([p, f"#$|{p}|$#"])

        num = len(phrases_map)
        logging.info(f"number of phrases: {num}")
        chunk_len = 1000
        num_chunks = (num + chunk_len - 1) // chunk_len
        start = 0

        for i in range(num_chunks):
            filename = f"heteronym_{i}.fst"
            if Path(filename).is_file():
                continue

            logging.info(f"Processing chunk {i}/{num_chunks}")
            start = i * chunk_len
            end = start + chunk_len

            this_chunk = phrases_map[start:end]
            logging.info(f"This chunk len: {len(this_chunk)}")
            fst = pynini.string_map(this_chunk)
            fst = fst.optimize()
            rule = cdrewrite(fst, "", "", sigma)

            logging.info(f"Saving to {filename}")
            rule.write(filename)
        logging.info(
            "\nHint: You can use\n\n\tfarcreate *.fst rule.far\n\n"
            "to combine all FSTs into a single FST archive file rule.far"
        )
    else:
        # this branch runs very slowly
        fst_list = []
        for p in phrases:
            fst = pynini.cross(p, f"#$|{p}|$#")
            # more words in a phrase -> smaller weight -> higher priority
            fst.set_final(fst.num_states() - 1, 1 / len(p))
            fst_list.append(fst)
        fst = functools.reduce(lambda a, b: a | b, fst_list)
        fst = fst.optimize()
        rule = cdrewrite(fst, "", "", sigma)

        filename = "heteronym.fst"
        logging.info(f"Saving to {filename}")
        rule.write(filename)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    logging.info("Started! May take 1 hour and 24 minutes")
    main()
