#!/usr/bin/env python3

import glob
import random

import lhotse
import time
from asr_datamodule import XimalayaAsrDataModule


def test_generate_filenames(wer="0.1"):
    names = glob.glob(
        f"/mnt/bos-multimodal/multi-modal/audio/NGK/data2/ximalaya/wer/{wer}/cutset/*"
    )
    cut_set_filenames = []
    for n in names:
        cut_set_filenames += glob.glob(f"{n}/*")

    random.shuffle(cut_set_filenames)
    with open(f"cutset-all-{wer}.txt", "w") as f:
        for n in cut_set_filenames:
            f.write(f"{n}\n")

    print(f"{len(cut_set_filenames)}")

    cut_set_filenames = cut_set_filenames[:800]
    with open(f"cutset-random-800-{wer}.txt", "w") as f:
        for n in cut_set_filenames:
            f.write(f"{n}\n")

    start = time.time()
    cuts_train = lhotse.combine(lhotse.load_manifest_lazy(p) for p in cut_set_filenames)
    end = time.time()

    elapsed_seconds = end - start

    print(f"{elapsed_seconds} seconds")


def test_train():
    m = XimalayaAsrDataModule(None)
    train = m.train_cuts()
    print(train)
    # CutSet(len=15058764) [underlying data type: <class 'lhotse.lazy.LazyIteratorChain'>]


def test_valid():
    m = XimalayaAsrDataModule(None)
    valid = m.valid_cuts()
    print(valid)


def test_test():
    m = XimalayaAsrDataModule(None)
    test = m.test_cuts().subset(first=100)
    print(test)


def test_generate_filenames_aishell():
    cut_set_filenames = glob.glob(
        "/star-oss/fangjun/data/aishell/my-split-100/*.jsonl.gz"
    )

    random.shuffle(cut_set_filenames)

    with open(f"cutset-all-aishell.txt", "w") as f:
        for n in cut_set_filenames:
            f.write(f"{n}\n")

    print(f"{len(cut_set_filenames)}")

    start = time.time()
    cuts_train = lhotse.combine(lhotse.load_manifest_lazy(p) for p in cut_set_filenames)
    end = time.time()

    elapsed_seconds = end - start

    print(f"{elapsed_seconds} seconds")


def test_generate_filenames_zhvoice():
    cut_set_filenames = glob.glob(
        "/star-oss/fangjun/data/zhvoice/my-split-100/*.jsonl.gz"
    )

    random.shuffle(cut_set_filenames)

    with open(f"cutset-all-zhvoice.txt", "w") as f:
        for n in cut_set_filenames:
            f.write(f"{n}\n")

    print(f"{len(cut_set_filenames)}")

    start = time.time()
    cuts_train = lhotse.combine(lhotse.load_manifest_lazy(p) for p in cut_set_filenames)
    end = time.time()

    elapsed_seconds = end - start

    print(f"{elapsed_seconds} seconds")


def test_generate_filenames_aishell2():
    cut_set_filenames = glob.glob(
        "/star-oss/fangjun/data/aishell2/my-split-100/*.jsonl.gz"
    )

    random.shuffle(cut_set_filenames)

    with open(f"cutset-all-aishell2.txt", "w") as f:
        for n in cut_set_filenames:
            f.write(f"{n}\n")

    print(f"{len(cut_set_filenames)}")

    start = time.time()
    cuts_train = lhotse.combine(lhotse.load_manifest_lazy(p) for p in cut_set_filenames)
    end = time.time()

    elapsed_seconds = end - start

    print(f"{elapsed_seconds} seconds")


def test_generate_filenames_wenetspeech():
    cut_set_filenames = glob.glob(
        "/star-oss/fangjun/data/wenetspeech/my-splits-1000/*.jsonl.gz"
    )

    random.shuffle(cut_set_filenames)
    with open(f"cutset-all-wenetspeech.txt", "w") as f:
        for n in cut_set_filenames:
            f.write(f"{n}\n")

    print(f"{len(cut_set_filenames)}")

    start = time.time()
    cuts_train = lhotse.combine(lhotse.load_manifest_lazy(p) for p in cut_set_filenames)
    end = time.time()

    elapsed_seconds = end - start

    print(f"{elapsed_seconds} seconds")


def test_generate_filenames_kespeech():
    cut_set_filenames = glob.glob(
        "/star-oss/fangjun/data/kespeech/my-split-100/*.jsonl.gz"
    )

    random.shuffle(cut_set_filenames)
    with open(f"cutset-all-kespeech.txt", "w") as f:
        for n in cut_set_filenames:
            f.write(f"{n}\n")

    print(f"{len(cut_set_filenames)}")

    start = time.time()
    cuts_train = lhotse.combine(lhotse.load_manifest_lazy(p) for p in cut_set_filenames)
    end = time.time()

    elapsed_seconds = end - start

    print(f"{elapsed_seconds} seconds")


def test_generate_filenames_wenetspeech4tts():
    cut_set_filenames = glob.glob(
        "/star-data/kangwei/icefall/egs/wenetspeech4tts/ASR/data/fbank/splits/wenetspeech4tts_cuts_Basic.*.jsonl.gz"
    )

    random.shuffle(cut_set_filenames)
    with open(f"cutset-all-wenetspeech4tts.txt", "w") as f:
        for n in cut_set_filenames:
            f.write(f"{n}\n")

    print(f"{len(cut_set_filenames)}")

    start = time.time()
    cuts_train = lhotse.combine(lhotse.load_manifest_lazy(p) for p in cut_set_filenames)
    end = time.time()

    elapsed_seconds = end - start

    print(f"{elapsed_seconds} seconds")


def get_dataset(filename):
    with open(filename) as f:
        for line in f:
            yield line.strip()


def generate_combined():
    ximalaya = list(get_dataset("./cutset-all-0.3.txt"))
    aishell = list(
        get_dataset(
            "/star-oss/fangjun/data/normalized/aishell/cutset-all-aishell-normalized.txt"
        )
    )
    aishell2 = list(
        get_dataset(
            "/star-oss/fangjun/data/normalized/aishell2/cutset-all-aishell2-normalized.txt"
        )
    )
    kespeech = list(
        get_dataset(
            "/star-oss/fangjun/data/normalized/kespeech/cutset-all-kespeech-normalized.txt"
        )
    )
    wenetspeech = list(
        get_dataset(
            "/star-oss/fangjun/data/normalized/wenetspeech/cutset-all-wenetspeech-normalized.txt"
        )
    )
    wenetspeech4tts = list(
        get_dataset(
            "/star-oss/fangjun/data/normalized/wenetspeech4tts/cutset-all-wenetspeech4tts-normalized.txt"
        )
    )
    zhvoice = list(
        get_dataset(
            "/star-oss/fangjun/data/normalized/zhvoice/cutset-all-zhvoice-normalized.txt"
        )
    )

    all_data = (
        ximalaya
        + aishell
        + aishell2
        + kespeech
        + wenetspeech
        + wenetspeech4tts
        + zhvoice
    )
    random.shuffle(all_data)

    with open("cutset-all-7-combined.txt", "w") as f:
        for n in all_data:
            f.write(f"{n}\n")


def main():
    print("started")
    generate_combined()
    #  test_generate_filenames_zhvoice()
    #  test_generate_filenames_aishell2()
    #  test_generate_filenames_aishell()
    #  test_generate_filenames_kespeech()
    #  test_generate_filenames_wenetspeech4tts()
    #  test_generate_filenames_wenetspeech()
    #  test_generate_filenames(wer="0.3")
    #  test_train()
    #  test_valid()
    #  test_test()


if __name__ == "__main__":
    main()
