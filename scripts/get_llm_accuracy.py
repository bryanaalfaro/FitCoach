# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""LLM Auto-Eval Script."""

import argparse
import ast
import json
import random
import re
from typing import Any, List

import evaluate
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = (
    "You are an intelligent chatbot designed for evaluating feedbacks provided by a virtual "
    "fitness coach to a person. You always provide your responses as a python dictionary string.\n"
    "Your task is to compare the accuracy of the the predicted feedback with the ground truth "
    "feedback. Here is how you can accomplish this:\n"
    "-The predicted feedback must be factually accurate, relevant and align with the ground truth "
    "feedback.\n"
    "-Consider synonyms or paraphrases as valid matches.\n"
    "-Take into account repetition counts that can expressed both in numeric form or in words."
)

USER_CONTENT = (
    "Please evaluate the following predicted feedback:\n"
    "-Ground truth feedback: <1>\n"
    "-Predicted feedback: <2>\n\n"
    "Provide your evaluation as a python dictionary string with the accuracy score where the "
    "score is an integer value between 1 and 5, with 5 indicating the highest level of accuracy."
    "Generate the response only in the form of a Python dictionary string with keys 'score', "
    "where its value is the accuracy score in INTEGER, not STRING."
    "For example, your response should look like this: {'score': int(score)}."
    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION."
)


def get_parser() -> argparse.ArgumentParser:
    """Parse command line arguments."""
    _parser = argparse.ArgumentParser(
        description="LLM Accuracy Evaluator",
        usage="python scripts/get_llm_accuracy.py [--options]",
    )
    _parser.add_argument(
        "--results_file",
        default=None,
        required=True,
        help="Path to results json file. First run scripts/evaluate_baseline.py .",
    )
    _parser.add_argument(
        "--llm_model_path",
        required=True,
        help="Path to directory containing LLaMA-3-70B-Instruct weights.",
    )
    _parser.add_argument(
        "--llm_tokenizer_path",
        required=False,
        default=None,
        help="Path to directory containing LLaMA-3-70B-Instruct tokenizer.",
    )

    return _parser


def extract_substrings_in_curly_braces(text: str) -> List[str]:
    """Extracts substrings enclosed in curly braces using regular expressions."""
    pattern = r"\{(.*?)\}"  # Matches content between curly braces (non-greedy)
    matches = re.findall(pattern, text)
    return matches


def fill_template(template: str, fillers: List[str]) -> str:
    """Fills a string template (here the LLM prompts) with content."""
    for idx, filler in enumerate(fillers):
        placeholder = f"<{idx + 1}>"
        template = template.replace(placeholder, filler)
    return template


def load_json_from_file(file_path: str) -> Any:
    """Loads JSON data from a file and returns it as a Python object."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None


if __name__ == "__main__":
    # Parse CLI arguments
    parser = get_parser()
    args = parser.parse_args()

    preds = load_json_from_file(args.results_file)

    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_tokenizer_path if args.llm_tokenizer_path is not None else args.llm_model_path,
        padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(args.llm_model_path, load_in_8bit=True)

    bert_score = evaluate.load("bertscore")
    rouge_score = evaluate.load("rouge")
    meteor_score = evaluate.load("meteor")
    scores, meteor_scores, rouge_scores, bert_scores = [], [], [], []

    random.shuffle(preds)
    for eval_item in tqdm(preds):
        messages = [
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                "content": fill_template(USER_CONTENT, [eval_item["GT"], eval_item["Pred"]]),
            },
        ]

        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            outputs = model.generate(
                input_ids,
                max_new_tokens=64,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

        response = outputs[0][input_ids.shape[-1] :]
        response = tokenizer.decode(response, skip_special_tokens=True)

        print("=" * 40)
        print(f"GT: {eval_item['GT']}\nPred: {eval_item['Pred']}\n\n")
        score_dicts = extract_substrings_in_curly_braces(response)
        if len(score_dicts) == 1:
            try:
                score_dict = ast.literal_eval("{" + score_dicts[0] + "}")
                scores.append(float(score_dict["score"]))
            except (SyntaxError, TypeError, ValueError):
                pass
        print("Running Mean LLM Accuracy: ", sum(scores) / (len(scores) + 1e-12))
