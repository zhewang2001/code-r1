"""
Preprocess LeetCode problems (newfacade/LeetCodeDataset) to parquet format.
"""

import os
from datasets import load_dataset, concatenate_datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def ground_truth_tests(example):
    return f"""{example['test']}

check({example['entry_point'].strip()})"""


def question_prompt(example):
    return f"""Please solve the programming task below using a self-contained code snippet in a markdown code block.

{example['meta']['query'].strip()}
"""


SYSTEM_PROMPT = """You are a helpful programming assistant. The user will ask you a question, and you as the assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively. For example:
<think>
Let's think step by step to solve the programming task in high quality...
{thinking processing...}
</think>
<answer>
The code below solves the task and is vulnerability-free...
```{language}
{code...}
```
</answer>"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/leetcode")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    test_dataset = load_dataset(
        "json", data_files="LeetCodeDataset/data/LeetCodeDataset-v2-test-problems.jsonl"
    )["train"]
    print("Test set:", test_dataset)

    train_dataset = concatenate_datasets(
        [
            load_dataset(
                "json",
                data_files="LeetCodeDataset/data/LeetCodeDataset-v2-rl-problems.jsonl",
            )["train"],
            load_dataset(
                "json",
                data_files="LeetCodeDataset/data/LeetCodeDataset-v2-sft-problems.jsonl",
            )["train"],
        ]
    ).filter(
        lambda example: example["meta"]["question_id"]
        not in set([d["question_id"] for d in test_dataset["meta"]])
    )
    print("Before deduplication - Training set:", train_dataset)

    first_time_idx = []
    seen_question_ids = set()
    for i, example in enumerate(train_dataset):
        if example["meta"]["question_id"] not in seen_question_ids:
            first_time_idx.append(i)
            seen_question_ids.add(example["meta"]["question_id"])
    train_dataset = train_dataset.select(first_time_idx)

    print("After deduplication - Training set:", train_dataset)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            data = {
                "data_source": "code",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question_prompt(example)},
                ],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth_tests(example),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "reference": example["completion"],
                    "dataset": "LeetCodeDataset",
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
