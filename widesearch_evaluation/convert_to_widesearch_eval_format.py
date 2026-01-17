#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path

SEED = 3

INPUT_JSONL = "/mnt/mnt/public/zhangruize/MAS/repo/ASearcher/widesearch_evaluation/result_fixed/widesearch_asearcher_asearcher_seed3_20260116_121140.jsonl"

# 输出目录：默认与输入文件同目录；需要改路径就改这里
OUTPUT_DIR = "/mnt/mnt/public/zhangruize/MAS/repo/ASearcher/widesearch_evaluation/widesearch_format_result"

FILENAME_TEMPLATE = "ASearcher-qwen2.5-7b_{instance_id}_{SEED}_response.jsonl"


def sanitize_filename_part(s: str) -> str:
    """将 instance_id 清理为安全的文件名片段：仅保留字母数字._-，其他替换为下划线"""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s)


def main():
    in_path = Path(INPUT_JSONL)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 若同一个 instance_id 重复出现，避免覆盖：追加 _dupN
    dup_counter = {}

    with in_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Line {line_no} JSON decode failed: {e}") from e

            if "instance_id" not in obj:
                raise KeyError(f"Line {line_no} missing 'instance_id'")
            if "response" not in obj:
                raise KeyError(f"Line {line_no} missing 'response'")

            instance_id_raw = str(obj["instance_id"])
            instance_id = sanitize_filename_part(instance_id_raw)

            out_obj = {
                "instance_id": obj["instance_id"],
                "response": obj["response"],
            }

            filename = FILENAME_TEMPLATE.format(instance_id=instance_id, SEED=SEED)

            if filename in dup_counter:
                dup_counter[filename] += 1
                stem = Path(filename).stem
                suffix = Path(filename).suffix
                filename = f"{stem}_dup{dup_counter[filename]}{suffix}"
            else:
                dup_counter[filename] = 0

            out_path = out_dir / filename

            with out_path.open("w", encoding="utf-8") as wf:
                wf.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"Done. Output dir: {out_dir}")


if __name__ == "__main__":
    main()