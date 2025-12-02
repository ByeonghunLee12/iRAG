#!/usr/bin/env python
"""
Pair query / retrieved 이미지를 폴더 구조로 복사·정리.

예)
python pair_imgs.py \
  --results_txt /path/to/results_80_train_down.txt \
  --out_dir CODEC_VALID_80 \
  --topk 5
"""

import argparse
import os
import shutil
from typing import List


def copy_with_rank(query_path: str,
                   retrieved_paths: List[str],
                   dst_root: str,
                   topk: int = None) -> None:
    """query 1장 + retrieved K장을 dst_root/<query_id> 하위에 복사한다."""
    query_id = os.path.basename(query_path)
    query_dir = os.path.join(dst_root, f"DIV2K_HR_patch512_val_{query_id}")
    os.makedirs(query_dir, exist_ok=True)

    # ① Query 이미지
    if os.path.exists(query_path):
        shutil.copy(query_path, os.path.join(query_dir, f"DIV2K_query_{query_id}"))
    else:
        print(f"[WARN] Query image not found: {query_path}")

    # ② Retrieved 이미지들
    for rank, img_path in enumerate(retrieved_paths[:topk], start=1):
        if os.path.exists(img_path):
            new_name = f"Flickr2K_R{rank}_{os.path.basename(img_path)}"
            shutil.copy(img_path, os.path.join(query_dir, new_name))
        else:
            print(f"[WARN] Retrieved image not found: {img_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-organise retrieval results into easy-to-browse folders."
    )
    parser.add_argument("--results_txt", required=True,
                        help="Path to results_*.txt produced by ContraHash.")
    parser.add_argument("--out_dir", required=True,
                        help="Destination root directory for paired data.")
    parser.add_argument("--topk", type=int, default=None,
                        help="Copy only top-K retrieved images (default: copy all).")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.results_txt, "r") as f:
        for line in f:
            if not line.strip():
                continue
            query_path, retrieved_str = line.rstrip().split("\t")
            retrieved_imgs = [p.strip() for p in retrieved_str.split(",") if p.strip()]
            copy_with_rank(query_path, retrieved_imgs, args.out_dir, args.topk)

    print("✅ Data pairing complete!")


if __name__ == "__main__":
    main()
