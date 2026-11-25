"""Download the Shakespeare <-> modern English parallel corpus.

Source: https://github.com/harsh19/Shakespearizing-Modern-English (folder data/)
Please cite in your report:
  - Jhamtani et al. (2017) "Shakespearizing Modern Language Using Copy-Enriched
    Sequence-to-Sequence Models"
  - Xu et al. (2012) "Paraphrasing for Style"

Run:  python download_data.py
"""
import os
import urllib.request

import config

BASE_URL = "https://raw.githubusercontent.com/harsh19/Shakespearizing-Modern-English/master/data/"
FILES = [
    f"{split}.{side}.nltktok"
    for split in ["train", "valid", "test"]
    for side in ["original", "modern"]
]


def main():
    os.makedirs(config.CORPUS_DIR, exist_ok=True)
    for name in FILES:
        path = os.path.join(config.CORPUS_DIR, name)
        if os.path.exists(path):
            print(f"already have {name}")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(BASE_URL + name, path)
        with open(path, encoding="utf-8") as f:
            print(f"  {sum(1 for _ in f)} lines")
    print("Done. Files are in", config.CORPUS_DIR)


if __name__ == "__main__":
    main()
