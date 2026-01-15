"""
WideSearch Evaluation Script
Generates trajectories for WideSearch data without LLM-as-judge evaluation.
Output format is compatible with SearchR1 trajectory format.
"""

import argparse
import json
import os
import time
from datetime import datetime
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Dict, Any, Optional, List

import asyncio
from agent import make_agent
from tools.search_utils import make_search_client
from config_loader import load_config_and_set_env
from llm_utils import get_sglang_llm


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class CompatibleLLMResponse:
    def __init__(self, text: str, input_len: Optional[int] = None,
                 input_tokens: Optional[List[int]] = None,
                 output_len: Optional[int] = None,
                 output_tokens: Optional[List[int]] = None,
                 output_logprobs: Optional[List[float]] = None,
                 output_versions: Optional[List[int]] = None):
        self.text = text
        self.input_len = input_len
        self.input_tokens = input_tokens or []
        self.output_len = output_len
        self.output_tokens = output_tokens or []
        self.output_logprobs = output_logprobs or []
        self.output_versions = output_versions or []


PROMPT_TYPES = {
    "asearcher-reasoning": "{question}",
    "asearcher": """A conversation between User and Assistant. The user asks a question, and the Assistant answers it. The Assistant analyzes the given question and information in the mind, retains important relevant information, calls a search engine to find necessary information, accesses web pages with certain urls, and provides the user with the answer. The Assistant conducts search by <search> query </search>, access certain url by <access> url </access>, and the top search results and url page will be returned between <information> and </information>.  The reasoning processes are enclosed within <think> </think>. Finally, the Assistant provides answer inside <answer> and </answer>, i.e. <answer> answer here </answer>. If there are multiple queries, ensure all answers are enclosed within <answer> </answer>, seperated with comma. \n\nNote: the question is a valid question and you should try to find a correct answer. \n\nUser: {question}\n\nAssistant: \n<think>""",
    "local-rag": """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}
<|im_end|>
<|im_start|>assistant""",
    "search-r1": """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}?
""",
}


def make_prompt(question: str, prompt_type: str) -> str:
    if prompt_type not in PROMPT_TYPES:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    return PROMPT_TYPES[prompt_type].format(question=question)


def load_widesearch_data(data_path: str) -> List[Dict]:
    """Load WideSearch data from JSONL file."""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse line: {line[:100]}...")
    return data


def prepare_widesearch_data(args) -> List[Dict]:
    """Prepare WideSearch data for evaluation."""
    data = load_widesearch_data(args.data_path)

    # Assign IDs
    for idx, item in enumerate(data):
        if "id" not in item:
            item["id"] = item.get("instance_id", str(idx))

    # Shuffle if needed
    if args.shuffle:
        random.shuffle(data)

    # Limit number of samples
    if args.num_test_sample > 0 and args.num_test_sample < len(data):
        data = data[:args.num_test_sample]

    # Apply start/end range
    if args.start > 0:
        data = data[args.start:]
    if args.end > 0 and args.end < len(data):
        data = data[:args.end]

    # Prepare prompts
    for item in data:
        question = item.get("query", item.get("question", ""))
        item["question"] = question
        item["prompt"] = make_prompt(question, args.prompt_type)
        # Keep original fields for output
        item["instance_id"] = item.get("instance_id", item["id"])

    return data


def parse_args():
    parser = argparse.ArgumentParser(description="WideSearch Trajectory Generation")
    parser.add_argument("--data_path", default="/mnt/mnt/public/zhangruize/MAS/data/widesearch/widesearch.jsonl", type=str,
                        help="Path to WideSearch data file")
    parser.add_argument("--model_name_or_path", default="/storage/openpsi/models/Qwen__Qwen3-1.7B/", type=str)
    parser.add_argument("--output_dir", default="./result", type=str)
    parser.add_argument("--prompt_type", default="asearcher", type=str)
    parser.add_argument("--agent-type", default="asearcher", type=str)
    parser.add_argument("--search-client-type", default="async-web-search-access", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0.6, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--top_k", default=-1, type=int)
    parser.add_argument("--max-tokens-per-call", default=4096, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-jina", action="store_true", help="Use Jina to get webpage content")
    parser.add_argument("--jina-api-key", type=str, help="Jina API key")
    parser.add_argument("--concurrent", type=int, default=64, help="Concurrent requests")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--output_prefix", type=str, default="widesearch", help="Output file prefix")
    args = parser.parse_args()

    # Adjust sampling parameters
    if args.temperature == 0:
        args.top_p = 1
        args.top_k = -1

    available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    args.data_parallel_size = len(available_gpus) // args.tensor_parallel_size
    return args


def truncate_at_first_complete_tool_call(text: str) -> str:
    """Truncate text at the first complete tool call"""
    import re

    patterns = [
        r'(<search>.*?</search>)',
        r'(<access>.*?</access>)',
        r'(<answer>.*?</answer>)'
    ]

    earliest_end = len(text)
    found_tool_call = False

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            tool_call_end = match.end()
            if tool_call_end < earliest_end:
                earliest_end = tool_call_end
                found_tool_call = True

    return text[:earliest_end] if found_tool_call else text


def convert_agent_tool_calls_to_dict(agent_tool_calls):
    """Convert agent tool calls to dict format"""
    import re

    dict_tool_calls = []

    for tool_call_str in agent_tool_calls:
        # Parse <search>...</search>
        search_match = re.search(r'<search>(.*?)</search>', tool_call_str, re.DOTALL)
        if search_match:
            dict_tool_calls.append({"type": "search", "query": search_match.group(1).strip()})
            continue

        # Parse <access>...</access>
        access_match = re.search(r'<access>(.*?)</access>', tool_call_str, re.DOTALL)
        if access_match:
            dict_tool_calls.append({"type": "access", "url": access_match.group(1).strip()})
            continue

        # Parse <answer>...</answer>
        answer_match = re.search(r'<answer>(.*?)</answer>', tool_call_str, re.DOTALL)
        if answer_match:
            dict_tool_calls.append({"type": "answer", "content": answer_match.group(1).strip()})
            continue

    return dict_tool_calls


async def process_single_llm_query(llm, tokenizer, prompt: str, sampling_params: Dict, args, qid=None) -> CompatibleLLMResponse:
    """Process a single LLM query."""
    sampling_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=sampling_params.get("max_new_tokens", args.max_tokens_per_call),
        n=1,
        stop_token_ids=(
            [151645, 151643]
            if "qwen2" in args.model_name_or_path.lower()
            else [tokenizer.pad_token_id, tokenizer.eos_token_id]
        ),
    )

    # Handle agent's stop strings
    if sampling_params.get("stop") and isinstance(sampling_params["stop"], list):
        stop_strings = sampling_params["stop"]

        if stop_strings == ["</think>"]:
            tokens = tokenizer.encode("</think>", add_special_tokens=False)
            existing_stops = sampling_kwargs.get("stop_token_ids", [])
            sampling_kwargs["stop_token_ids"] = existing_stops + tokens
            sampling_kwargs["stop"] = ["</think>"]

    try:
        output = await llm.async_generate(prompt, sampling_kwargs)
    except ValueError as e:
        print(f"ValueError when handling query {qid}")
        raise e

    text = output['text'] if isinstance(output, dict) else output

    # Post-process: truncate at first complete tool call
    if sampling_params.get("stop") and sampling_params["stop"] != ["</think>"]:
        text = truncate_at_first_complete_tool_call(text)

    input_tokens = tokenizer.encode(prompt) if tokenizer else None
    output_tokens = tokenizer.encode(text) if tokenizer and text else None

    print(f"[DEBUG] input length: {len(input_tokens) if input_tokens else 'N/A'}", flush=True)

    return CompatibleLLMResponse(
        text=text,
        input_len=len(input_tokens) if input_tokens else None,
        input_tokens=input_tokens,
        output_len=len(output_tokens) if output_tokens else None,
        output_tokens=output_tokens,
    )


async def process_single_search_query(search_client, query: str, topk: int = 5) -> Any:
    """Process a single search query."""
    req_meta = {
        "queries": [query],
        "topk": topk,
        "return_scores": False
    }
    results = await search_client.query_async(req_meta)
    return results if results else None


async def process_single_access_query(search_client, url: str) -> Any:
    """Process a single URL access query."""
    results = await search_client.access_async([url])
    return results if results else None


def format_search_result_for_message(query: str, documents: List, urls: List) -> str:
    """Format search result for message history (SearchR1 compatible format)."""
    result_text = f"Search query: {query}\nResult: "
    for i, (doc, url) in enumerate(zip(documents, urls)):
        if isinstance(doc, dict):
            doc_text = doc.get("text", doc.get("content", str(doc)))
        else:
            doc_text = str(doc)
        result_text += f"[Doc {i+1}]({url}):\n{doc_text}\n\n"
    return result_text.strip()


async def process_single_work_item(semaphore, agent_type, llm, tokenizer, search_client, args, out_dir, process):
    """Process a single work item and generate trajectory."""
    async with semaphore:
        # Initialize message history in SearchR1 format
        message_history = []

        # Add initial user message
        message_history.append({
            "role": "user",
            "content": process["prompt"]
        })

        if "history" not in process:
            process["history"] = []
            process["running"] = True
            process["num_turns"] = 0

        # Create fresh agent instance
        agent = make_agent(agent_type)
        agent.initialize_with_prompt(process)

        if hasattr(agent, 'set_tokenizer'):
            agent.set_tokenizer(tokenizer)

        final_response = ""

        while process["running"] and agent.num_turns < agent.max_turns:
            if agent.is_finished:
                process["running"] = False
                break

            try:
                prompt_or_messages, sampling_params = agent.prepare_llm_query()

                if agent.is_finished:
                    process["running"] = False
                    break

                if isinstance(prompt_or_messages, str):
                    prompt = prompt_or_messages

                    llm_response = await process_single_llm_query(
                        llm, tokenizer, prompt, sampling_params, args, qid=process["id"]
                    )
                    completion_text = llm_response.text

                tool_calls_raw = agent.consume_llm_response(llm_response, completion_text)
                tool_calls = convert_agent_tool_calls_to_dict(tool_calls_raw)

                if agent.is_finished:
                    # Add final assistant message (only if non-empty)
                    if completion_text and completion_text.strip():
                        message_history.append({
                            "role": "assistant",
                            "content": completion_text
                        })
                        final_response = completion_text
                    process["running"] = False
                    break

                # Log progress
                if tool_calls:
                    print(f"Process {process['id']}: {', '.join([tc['type'] for tc in tool_calls])}")

                # Add assistant message to history (only if non-empty)
                if completion_text and completion_text.strip():
                    message_history.append({
                        "role": "assistant",
                        "content": completion_text
                    })

                # Add to internal history
                process["history"].append({
                    "type": "llm_response",
                    "text": completion_text,
                    "tool_calls": tool_calls
                })

                # Process each tool call
                for tool_call in tool_calls:
                    if tool_call["type"] == "search":
                        search_result = await process_single_search_query(search_client, tool_call["query"])
                        if search_result:
                            if isinstance(search_result, dict):
                                documents = search_result.get("documents", []) or []
                                urls = search_result.get("urls", []) or []
                            elif isinstance(search_result, list):
                                documents = []
                                urls = []
                                for result in search_result:
                                    if isinstance(result, dict):
                                        documents.extend(result.get("documents", []) or [])
                                        urls.extend(result.get("urls", []) or [])
                            else:
                                documents = []
                                urls = []

                            documents = documents or []
                            urls = urls or []

                            tool_response = {
                                "type": "search",
                                "documents": documents,
                                "urls": urls
                            }
                            agent.consume_tool_response(tool_response)

                            # Add tool message to message_history (SearchR1 format)
                            tool_content = format_search_result_for_message(tool_call["query"], documents, urls)
                            message_history.append({
                                "role": "tool",
                                "content": tool_content
                            })

                            process["history"].append({
                                "type": "search_result",
                                "query": tool_call["query"],
                                "documents": documents,
                                "urls": urls
                            })

                    elif tool_call["type"] == "access":
                        access_result = await process_single_access_query(search_client, tool_call["url"])
                        if access_result:
                            if isinstance(access_result, dict):
                                page = access_result.get("page", "") or ""
                            elif isinstance(access_result, str):
                                page = access_result or ""
                            else:
                                page = str(access_result) if access_result else ""

                            page = page or ""

                            tool_response = {
                                "type": "access",
                                "page": page
                            }
                            agent.consume_tool_response(tool_response)

                            # Add tool message to message_history (SearchR1 format)
                            tool_content = f"URL: {tool_call['url']}\nContent: {page[:2000]}"
                            message_history.append({
                                "role": "tool",
                                "content": tool_content
                            })

                            process["history"].append({
                                "type": "page_access",
                                "url": tool_call["url"],
                                "page": page
                            })

                    elif tool_call["type"] == "answer":
                        process["pred_answer"] = tool_call["content"]
                        final_response = completion_text
                        process["running"] = False
                        break

                process["num_turns"] = agent.num_turns

            except Exception as e:
                print(f"Error processing work item {process['id']}: {e}")
                import traceback
                traceback.print_exc()
                process["running"] = False
                process["error"] = str(e)
                break

            # Save intermediate state
            intermediate_output = {
                "instance_id": process["instance_id"],
                "response": final_response,
                "message_history": message_history,
                "internal_history": process["history"]
            }
            with open(os.path.join(out_dir, f"{process['id']}.json"), "w") as f:
                json.dump(intermediate_output, f, ensure_ascii=False, indent=2)

        # Get final answer if not already set
        if "pred_answer" not in process and hasattr(agent, 'get_answer'):
            final_answer = agent.get_answer()
            if final_answer:
                process["pred_answer"] = final_answer
            else:
                process["pred_answer"] = ""

        # Extract response from final answer
        if not final_response and "pred_answer" in process:
            final_response = process.get("pred_answer", "")

        # Build final output in SearchR1-compatible format
        output = {
            "instance_id": process["instance_id"],
            "response": final_response,
            "message_history": message_history,
        }

        # Save final state
        with open(os.path.join(out_dir, f"{process['id']}.json"), "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        return output


async def eval_widesearch(semaphore, llm, args):
    """Main evaluation function for WideSearch."""
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )

    search_client = make_search_client(args.search_client_type, args.use_jina, args.jina_api_key)

    # Prepare data
    processes = prepare_widesearch_data(args)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        args.output_dir,
        f"{args.output_prefix}_{args.agent_type}_{args.prompt_type}_seed{args.seed}_{timestamp}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Check for existing results if not overwriting
    if not args.overwrite:
        existing_ids = set()
        for fname in os.listdir(out_dir):
            if fname.endswith('.json'):
                existing_ids.add(fname.replace('.json', ''))
        processes = [p for p in processes if p["id"] not in existing_ids]
        print(f"Skipping {len(existing_ids)} existing results, processing {len(processes)} remaining")

    start_time = time.time()

    # Create tasks
    tasks = [
        process_single_work_item(semaphore, args.agent_type, llm, tokenizer, search_client, args, out_dir, p)
        for p in processes
    ]

    results = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating trajectories"):
        results.append(await f)

    # Write final output JSONL (SearchR1-compatible format)
    out_file = out_dir + ".jsonl"

    # Load all results (including existing ones)
    all_results = []
    for fname in os.listdir(out_dir):
        if fname.endswith('.json'):
            with open(os.path.join(out_dir, fname), 'r') as f:
                all_results.append(json.load(f))

    # Sort by instance_id for consistency
    all_results.sort(key=lambda x: x.get("instance_id", ""))

    with open(out_file, "w") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    time_use = time.time() - start_time
    print(f"\nTime used: {int(time_use // 60)}:{int(time_use % 60):02d}")
    print(f"Output saved to: {out_file}")
    print(f"Total trajectories: {len(all_results)}")

    return results


async def main(args):
    print("Loading configuration...")
    load_config_and_set_env()

    print(f"Loading model from {args.model_name_or_path}")
    llm = get_sglang_llm(args)

    try:
        semaphore = asyncio.Semaphore(args.concurrent)
        await eval_widesearch(semaphore, llm, args)
    finally:
        # Cleanup
        if llm is not None:
            try:
                if hasattr(llm, 'shutdown'):
                    llm.shutdown()
                elif hasattr(llm, 'close'):
                    llm.close()
                del llm

                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"Warning: Error while releasing GPU memory: {e}")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    asyncio.run(main(args))
