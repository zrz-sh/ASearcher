# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple evaluation metrics for WideSearch trajectory generation.
LLM-as-judge functionality has been removed.
"""

import re
import string


def normalize_answer(s):
    """Normalize answer string for comparison."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    """Check exact match between prediction and golden answers."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    """Check substring exact match between prediction and golden answers."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def bool_mapping(s):
    """Map boolean strings to yes/no."""
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s


def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    if len(matches) <= 0:
        return None

    return matches[-1].group(1).strip()


def compute_score_em(answer, ground_truth, method='strict', format_score=0., score=1.):
    """Compute exact match score."""
    if answer is None:
        return None, 0
    else:
        if em_check(answer, ground_truth):
            return score
        else:
            return format_score


def compute_score_subem(answer, ground_truth, method='strict', format_score=0., score=1.):
    """Compute substring exact match score."""
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth):
            return score
        else:
            return format_score


def normalize_text(text: str) -> str:
    """Preprocess text for scoring."""
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    return text


def f1_score(answer_content, gt):
    """Compute F1 score between answer and ground truth."""
    answer_content = normalize_text(answer_content)
    gt = normalize_text(gt)

    pred_tokens = set(answer_content.split())
    gt_tokens = set(gt.split())

    if not gt_tokens:
        return 0
    if not pred_tokens:
        return 0

    common_tokens = pred_tokens & gt_tokens

    precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
    recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0

    f1 = 0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)

    return f1


def compute_score_f1(answer, ground_truth, method='strict', format_score=0., score=1.):
    """Compute F1 score."""
    if answer is None:
        return None, 0
    else:
        ret_score = f1_score(answer, ground_truth)
        return ret_score


def cover_exact_match_score_1(prediction, ground_truth):
    """Compute cover exact match score."""
    pre_list = normalize_answer(bool_mapping(prediction)).split(" ")
    ground_list = normalize_answer(bool_mapping(ground_truth)).split(" ")
    return float(all(ground in pre_list for ground in ground_list))
