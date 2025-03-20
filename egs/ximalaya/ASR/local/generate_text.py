#!/usr/bin/env python3

"""
This script generates a text file from all cutset.
The text file contains all transcript in the cutset
"""

import glob
import logging
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path

from lhotse import CutSet
from tqdm import tqdm

from icefall.utils import setup_logger


def process_file(f):
    cut_set = CutSet.from_file(f)
    ans = []
    for c in cut_set:
        ans.append(c.supervisions[0].text)
    return ans


def process_dataset(name):
    cut_dir = Path(
        "/mnt/bos-multimodal/multi-modal/audio/NGK/data/ximalaya/normalized/wer/0.2/cutset"
    )
    files = glob.glob(str(cut_dir / name / "*.jsonl.gz"))

    num_jobs = 50

    if len(files) == 0:
        logging.warning(f"Skip {name} since there are no manifest files")
        return

    ans = []
    with ProcessPoolExecutor(num_jobs) as ex:
        futures = []
        for i, f in enumerate(files):
            futures.append(ex.submit(process_file, f))

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Processing {name}",
            leave=False,
        ):
            text_list = future.result()
            ans.extend(text_list)
    return ans


def main():
    Path("data").mkdir(exist_ok=True)
    if Path("data/text").is_file():
        logging.warning("data/text exists - skip")
        return
    names_file = "/mnt/data-ssd/user/omni/fj/open-source/large-dataset/names.txt"
    names = []
    with open(names_file) as f:
        for line in f:
            if line[0].isupper():
                continue
            names.append(line.strip())

    text_list = []
    for i, name in enumerate(names):
        logging.warning(f"Processing {name}, {i}/{len(names)}")
        this_text_list = process_dataset(name)
        if this_text_list is not None:
            text_list += this_text_list

    with open("./data/text", "w", encoding="utf-8") as f:
        for text in text_list:
            f.write(f"{text}\n")

    logging.warning("Saved to data/text")


if __name__ == "__main__":
    setup_logger("./log/generate-text")
    main()
