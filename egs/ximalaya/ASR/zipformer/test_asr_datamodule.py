#!/usr/bin/env python3

import glob
import random

import lhotse
import time
from asr_datamodule import XimalayaAsrDataModule


def test_generate_filenames():
    names = glob.glob(
        "/mnt/bos-multimodal/multi-modal/audio/NGK/data/ximalaya/wer/0.1/cutset/*"
    )
    cut_set_filenames = []
    for n in names:
        cut_set_filenames += glob.glob(f"{n}/*")
        print(len(cut_set_filenames))
    with open("cutset-all.txt", "w") as f:
        for n in cut_set_filenames:
            f.write(f"{n}\n")

    random.shuffle(cut_set_filenames)

    cut_set_filenames = cut_set_filenames[:400]
    with open("cutset-random-400.txt", "w") as f:
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
    test = m.test_cuts()
    print(test)


def main():
    print("started")
    #  test_generate_filenames()
    #  test_train()
    test_valid()
    test_test()


if __name__ == "__main__":
    main()
