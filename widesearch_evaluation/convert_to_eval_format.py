#!/usr/bin/env python3
"""
Convert fixed results to WideSearch evaluation format.

Input: result_fixed/{seed_dir}/ws_*.json
Output: result_fixed/{seed_name}.jsonl
"""

import json
from pathlib import Path
from tqdm import tqdm


def convert_to_eval_format(input_root, output_dir):
    """Convert JSON files to jsonl format for WideSearch evaluation"""

    input_root = Path(input_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each seed directory
    seed_dirs = sorted(input_root.glob("widesearch_asearcher_asearcher_seed*"))

    for seed_dir in seed_dirs:
        if not seed_dir.is_dir():
            continue

        print(f"\nProcessing {seed_dir.name}...")

        # Output file name: same as seed directory name
        output_file = output_dir / f"{seed_dir.name}.jsonl"

        # Collect all JSON files
        json_files = sorted(seed_dir.glob("ws_*.json"))

        # Write to jsonl
        with open(output_file, 'w', encoding='utf-8') as f:
            for json_file in tqdm(json_files, desc=f"  Converting"):
                # Read JSON file
                with open(json_file, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)

                # Extract required fields
                output_data = {
                    "instance_id": data.get("instance_id"),
                    "response": data.get("response", ""),
                    "message_history": data.get("message_history", [])
                }

                # Write as single line JSON
                f.write(json.dumps(output_data, ensure_ascii=False) + '\n')

        print(f"  ✓ Created {output_file}")
        print(f"  Total instances: {len(json_files)}")

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Output directory: {output_dir}")
    print(f"Total seed files created: {len(seed_dirs)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert fixed results to WideSearch eval format")
    parser.add_argument(
        "--input-root",
        type=str,
        default="/mnt/mnt/public/zhangruize/MAS/repo/ASearcher/widesearch_evaluation/result_fixed",
        help="Input directory containing fixed result folders"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/mnt/public/zhangruize/MAS/repo/ASearcher/widesearch_evaluation/result_fixed",
        help="Output directory for jsonl files (default: same as input)"
    )

    args = parser.parse_args()

    convert_to_eval_format(args.input_root, args.output_dir)
