#!/usr/bin/env python3
"""
Fix WideSearch response files using LLM with async concurrency.
Much faster than the serial version.
"""

import json
import os
import re
import argparse
import asyncio
from pathlib import Path
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import sys

# Try to load config
sys.path.insert(0, str(Path(__file__).parent))
try:
    from config_loader import load_config_and_set_env
    load_config_and_set_env()
except Exception as e:
    print(f"Note: Could not load config file: {e}")

# Initialize async OpenAI client
client = None

FIX_PROMPT = """You are a data cleaning assistant. Your task is to extract and format markdown tables from conversation responses.

Given a response text or message history, extract the markdown table and format it correctly.

Requirements:
1. Extract ONLY the markdown table (lines starting with |)
2. Output format must be: ```markdown\\n[table content]\\n```
3. Remove any <answer>, </answer>, <markdown>, </markdown> tags
4. Do NOT include any explanatory text before or after the table
5. The first line after ```markdown should start with |
6. Ensure the table is complete and well-formatted

Input will be either:
- A "response" field that needs fixing
- A "message_history" array where you need to find the table in the last assistant message

Output ONLY the correctly formatted response with ```markdown wrapper."""


def has_markdown_table(text):
    """Check if text contains a markdown table"""
    if not text:
        return False
    pipe_positions = [m.start() for m in re.finditer(r"\|", text)]
    return len(pipe_positions) >= 4


def needs_fixing(data):
    """Check if a JSON entry needs fixing"""
    response = data.get("response", "")

    # Case 1: Empty response
    if not response or response.strip() == "":
        return True

    # Case 2: Has <markdown> tags instead of ```markdown
    if "<markdown>" in response or "</markdown>" in response:
        return True

    # Case 3: Has <answer> tags that shouldn't be inside ```markdown
    if "```markdown" in response and "<answer>" in response:
        markdown_blocks = re.findall(r"```markdown(.*?)```", response, re.DOTALL)
        for block in markdown_blocks:
            if "<answer>" in block or "</answer>" in block:
                return True

    # Case 4: Has markdown table but no ```markdown wrapper
    if has_markdown_table(response) and "```markdown" not in response:
        return True

    return False


async def fix_response_with_llm(data, model="gpt-4o-mini", semaphore=None):
    """Use LLM to fix the response field"""
    async with semaphore:
        response = data.get("response", "")
        message_history = data.get("message_history", [])

        # Prepare input for LLM
        if response and response.strip():
            user_input = f"Response to fix:\n{response}"
        else:
            # Extract from message history
            assistant_messages = []
            for msg in message_history:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if has_markdown_table(content):
                        assistant_messages.append(content)

            if not assistant_messages:
                return None

            user_input = f"Message history to extract table from:\n{assistant_messages[-1]}"

        try:
            # Call LLM
            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": FIX_PROMPT},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.1,
                max_tokens=4096
            )

            fixed_response = completion.choices[0].message.content.strip()

            # Validate the output
            if "```markdown" in fixed_response and fixed_response.count("```") >= 2:
                return fixed_response
            else:
                return None

        except Exception as e:
            print(f"\nError calling LLM: {e}")
            return None


async def process_json_file(input_path, output_path, model="gpt-4o-mini", semaphore=None):
    """Process a single JSON file"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        modified = False

        if needs_fixing(data):
            fixed_response = await fix_response_with_llm(data, model, semaphore)
            if fixed_response:
                data["response"] = fixed_response
                modified = True

        # Save to output path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return modified, None

    except Exception as e:
        return False, str(e)


async def process_seed_dir(seed_dir, output_seed_dir, model, semaphore, pbar):
    """Process all files in a seed directory"""
    json_files = sorted(seed_dir.glob("ws_*.json"))

    tasks = []
    for json_file in json_files:
        output_file = output_seed_dir / json_file.name
        task = process_json_file(json_file, output_file, model, semaphore)
        tasks.append(task)

    results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        pbar.update(1)

    return results


async def main_async(args):
    """Main async function"""
    global client

    # Initialize async OpenAI client
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not provided!")
        print("Please provide via --api-key argument or OPENAI_API_KEY environment variable")
        sys.exit(1)

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.api_base
    )

    result_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(args.concurrency)

    # Get all seed directories
    seed_dirs = sorted(result_root.glob("widesearch_asearcher_asearcher_seed*"))

    total_files = sum(len(list(d.glob("ws_*.json"))) for d in seed_dirs if d.is_dir())

    print(f"\nProcessing {total_files} files from {len(seed_dirs)} seed directories...")
    print(f"Concurrency: {args.concurrency}")
    print(f"Model: {args.model}\n")

    stats = {
        "total_files": 0,
        "files_fixed": 0,
        "files_failed": 0
    }

    # Process all seed directories with progress bar
    with tqdm(total=total_files, desc="Processing files") as pbar:
        for seed_dir in seed_dirs:
            if not seed_dir.is_dir():
                continue

            output_seed_dir = output_root / seed_dir.name
            output_seed_dir.mkdir(parents=True, exist_ok=True)

            results = await process_seed_dir(seed_dir, output_seed_dir, args.model, semaphore, pbar)

            for modified, error in results:
                stats["total_files"] += 1
                if error:
                    stats["files_failed"] += 1
                elif modified:
                    stats["files_fixed"] += 1

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {stats['total_files']}")
    print(f"Files fixed: {stats['files_fixed']}")
    print(f"Files failed: {stats['files_failed']}")
    print(f"Output directory: {output_root}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Fix WideSearch response files using LLM (async version)")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--api-base", type=str, default="https://api.openai.com/v1",
                        help="OpenAI API base URL")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model to use for fixing")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Number of concurrent requests")
    parser.add_argument("--input-dir", type=str,
                        default="/mnt/mnt/public/zhangruize/MAS/repo/ASearcher/widesearch_evaluation/result",
                        help="Input directory containing result folders")
    parser.add_argument("--output-dir", type=str,
                        default="/mnt/mnt/public/zhangruize/MAS/repo/ASearcher/widesearch_evaluation/result_fixed",
                        help="Output directory for fixed results")
    args = parser.parse_args()

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
