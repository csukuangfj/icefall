#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
This file generates the file lexicon.txt that contains pronunciations of all
words and phrases.
"""

import functools

import pynini
from pynini import cdrewrite
from pynini.lib import utf8
from pypinyin import phrases_dict, pinyin_dict


def main():
    sigma = utf8.VALID_UTF8_CHAR.star

    phrases = phrases_dict.phrases_dict

    if False:
        phrases_map = []
        for p in phrases:
            phrases_map.append([p, f"#$|{p}|$#"])
        phrases_map = phrases_map[:1000]
        logging.info(f"number of phrases: {len(phrases_map)}")
        fst = pynini.string_map(phrases_map)
        fst = fst.optimize()
        rule = cdrewrite(fst, "", "", sigma)
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

    rule = generate_heteronym_rule_fst()
    rule.write(args.heteronym_fst)

    return rule


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
