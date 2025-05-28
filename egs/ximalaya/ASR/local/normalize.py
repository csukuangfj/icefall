#!/usr/bin/env python3
import glob
import random
import re
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path

from lhotse import CutSet
from tqdm import tqdm


def process_text(text: str, pattern=re.compile(r"[^\u4e00-\u9fff]")):
    return pattern.sub("", text)


def process_cutset(f: str):
    cut_set = CutSet.from_file(f)
    ans = []
    for c in cut_set:
        c.supervisions[0].text = process_text(c.supervisions[0].text)
        ans.append(c)

    return CutSet(ans)


def process_dataset(input_filename, out_cut_dir, out_filename):
    out_cut_dir = Path(out_cut_dir)
    out_cut_dir.mkdir(exist_ok=True, parents=True)

    files = []
    with open(input_filename) as f:
        for line in f:
            line = line.strip()
            files.append(line)

    for f in tqdm(files):
        name = f.split("/")[-1]
        if (out_cut_dir / name).is_file():
            print(f"{out_cut_dir / name} exists - skipping")
            continue

        out_cut_set = process_cutset(f)

        out_cut_set.to_file(out_cut_dir / name)

        print(f"Saved to {out_cut_dir / name}")

    cut_set_filenames = glob.glob(f"{out_cut_dir}/*.jsonl.gz")
    random.shuffle(cut_set_filenames)

    with open(f"{out_cut_dir}/../{out_filename}", "w") as f:
        for n in cut_set_filenames:
            f.write(f"{n}\n")


def main():
    process_dataset(
        "./cutset-all-aishell.txt",
        "/star-oss/fangjun/data/normalized/aishell/my-split-100",
        "./cutset-all-aishell-normalized.txt",
    )

    process_dataset(
        "./cutset-all-aishell2.txt",
        "/star-oss/fangjun/data/normalized/aishell2/my-split-100",
        "./cutset-all-aishell2-normalized.txt",
    )

    process_dataset(
        "./cutset-all-wenetspeech.txt",
        "/star-oss/fangjun/data/normalized/wenetspeech/my-split-1000",
        "./cutset-all-wenetspeech-normalized.txt",
    )

    process_dataset(
        "./cutset-all-kespeech.txt",
        "/star-oss/fangjun/data/normalized/kespeech/my-split-100",
        "./cutset-all-kespeech-normalized.txt",
    )

    process_dataset(
        "./cutset-all-wenetspeech4tts.txt",
        "/star-oss/fangjun/data/normalized/wenetspeech4tts/my-split-400",
        "./cutset-all-wenetspeech4tts-normalized.txt",
    )

    process_dataset(
        "./cutset-all-zhvoice.txt",
        "/star-oss/fangjun/data/normalized/zhvoice/my-split-100",
        "./cutset-all-zhvoice-normalized.txt",
    )


if __name__ == "__main__":
    main()
