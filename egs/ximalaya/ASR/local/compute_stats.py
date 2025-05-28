#!/usr/bin/env python3

import logging
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from lhotse import CutSet
from lhotse.cut.describe import CutSetStatistics
from tqdm import tqdm


def process_file(filename):
    logging.info(f"Processing {filename}")
    cut_set = CutSet.from_file(filename)

    stats = CutSetStatistics(full=True)
    total_seconds = np.array(stats.accumulate(cut_set).cut_durations).sum().item()
    return total_seconds


# "./cutset-all-0.2.txt" 352.685k
# "./cutset-all-0.3.txt" 374.605k (3990 files)
# ./cutset-all.txt 82.3k (1162 files)

# ./cutset-all-wenetspeech.txt, 30.217k (1001 files)
# ./cutset-all-wenetspeech4tts.txt, 7.2k (394 files)
# ./cutset-all-2-ximalaya-wenetspeech.txt,
# ./cutset-all-aishell.txt, 455 hours (100 files)
# ./cutset-all-aishell2.txt, 1000 hours (100 files)
# ./cutset-all-kespeech.txt, 1396 hours (200 files)
# ./cutset-all-zhvoice.txt,  888.4 hours (100 files)


def main():
    filenames = []
    #  with open("./cutset-all-0.3.txt") as f:
    #  with open("./cutset-random-first-400.txt") as f:
    #  with open("./cutset-all.txt") as f:
    #  with open("./cutset-all-wenetspeech.txt") as f:
    #  with open("./cutset-all-2-ximalaya-wenetspeech.txt") as f:
    #  with open("./cutset-all-aishell.txt") as f:
    #  with open("./cutset-all-aishell2.txt") as f:
    #  with open("./cutset-all-wenetspeech4tts.txt") as f:
    #  with open("./cutset-all-kespeech.txt") as f:
    with open("./cutset-all-zhvoice.txt") as f:
        for line in f:
            line = line.strip()
            filenames.append(line)

    logging.info("started")
    logging.info(f"Number of files {len(filenames)}")

    num_jobs = 20
    ans = []
    with ProcessPoolExecutor(num_jobs) as ex:
        futures = []
        for i, f in enumerate(filenames):
            futures.append(ex.submit(process_file, f))

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Processing {Path(f).stem}",
            leave=False,
        ):
            total_seconds = future.result()
            ans.append(total_seconds)
    total_hours = sum(ans) / 3600.0
    logging.info(f"Total hours: {total_hours:.4f}")  # 82309.5509 hours


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
