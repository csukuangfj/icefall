#!/usr/bin/env python3

import sentencepiece as spm
from icefall import tokenize_by_CJK_char, byte_encode


def main():
    model_file = "./data/lang_bbpe_500/bbpe.model"
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)
    text = "你好世界丹尼尔"
    text = tokenize_by_CJK_char(text)
    print(text)  # 你 好
    text = byte_encode(text)
    print(text)
    tokens = sp.encode_as_pieces(text)
    print(tokens)


if __name__ == "__main__":
    main()
