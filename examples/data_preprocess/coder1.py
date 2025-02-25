"""
Preprocess LeetCode problems (newfacade/LeetCodeDataset) to parquet format.
"""

import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

from datasets import load_dataset, concatenate_datasets
from rich.rule import Rule
import rich

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.coder1 import code_exec, remote_check_stdio, _ERROR_MSG_PREFIX


def minimize_stdio(inputs, outputs, max_n_tests=8):
    stdin_list = []
    stdout_list = []
    for stdin, stdout in zip(inputs, outputs):
        if isinstance(stdin, list):
            stdin = "\n".join(stdin)
        if isinstance(stdout, list):
            stdout = "\n".join(stdout)
        if sys.getsizeof(stdin) > 4 * 1024:
            continue
        stdout.replace("\r\n", "\n")
        stdin_list.append(stdin)
        stdout_list.append(stdout)

    zipped = sorted(zip(stdin_list, stdout_list), key=lambda x: sys.getsizeof(x[0]))

    if not zipped:
        print("No tests found!")
        return [], []

    sorted_stdin, sorted_stdout = zip(*zipped)
    return list(sorted_stdin[:max_n_tests]), list(sorted_stdout[:max_n_tests])


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

PY_IMPORTS = "import heapq\nfrom math import floor, gcd\nimport random\nimport sys\nfrom typing import *\nfrom functools import *\nimport collections\nfrom collections import *\nfrom itertools import *\nfrom heapq import *\nfrom bisect import *\nfrom string import *\nimport math\nimport datetime\ninf = float('inf')\n"


# this dataset is super noisy and needs code execution to verify the tasks
def taco(max_n_tests=8):
    rich.print(Rule("Loading likaixin/TACO-verified..."))
    dataset = load_dataset("likaixin/TACO-verified", trust_remote_code=True)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            oracle = json.loads(example["input_output"])
            source = example["source"]

            # skip poorly formatted examples
            if source in ["geeksforgeeks", "leetcode"]:
                return None

            # too short description
            if len("".join([c for c in example["question"] if c.isalnum()])) < 100:
                return None

            # no image
            if "image" in example["question"].lower() or "\n![" in example["question"]:
                return None

            prompt_pieces = [
                "Solve the programming task below in a Python markdown code block.",
                example["question"].strip(),
            ]
            if example["starter_code"].strip():
                prompt_pieces.append("Also feel free to reuse/extend the following starter code:")
                prompt_pieces.append(f"```python\n{example['starter_code'].strip()}\n```")

            ##
            ## Customization
            ##
            if "fn_name" in oracle:  # the dataset is too noisy
                fn_name = oracle["fn_name"]
                if source == "leetcode":
                    fn_name = "Solution()." + fn_name

                test_code = f"""\
_inputs = {oracle["inputs"]}
_outputs = {oracle["outputs"]}
import math
def _deep_eq(a, b, tol=1e-5):
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b): return False
        return all(_deep_eq(x, y, tol) for x, y in zip(a, b))
    return a == b

for i, o in zip(_inputs, _outputs):
"""

                if source in ["leetcode", "hackerrank"]:
                    test_code += f"    assert _deep_eq({fn_name}(*i), o)"
                elif source == "codewars":
                    test_code += f"    assert _deep_eq({fn_name}(*i), o[0])"
                else:
                    raise ValueError(f"Unknown source: {source}")

                _check_test = example["solutions"][-1] + "\n" + test_code
                if source in ["leetcode"]:
                    _check_test = PY_IMPORTS + _check_test
                if not code_exec(_check_test)[0]:
                    rich.print("[bold red]Test code failed!")
                    print(_check_test)
                    print(source)
                    return None
                oracle = json.dumps({"functional": test_code})
                assert example["starter_code"].strip() != ""
            elif "inputs" in oracle and "outputs" in oracle:
                stdin_list, stdout_list = minimize_stdio(oracle["inputs"], oracle["outputs"], max_n_tests)
                if len(stdin_list) == 0:
                    return None

                with ThreadPoolExecutor(max_workers=min(len(stdin_list), 8)) as executor:
                    futures = []
                    for stdin, stdout in zip(stdin_list, stdout_list):
                        futures.append(executor.submit(
                            remote_check_stdio,
                            example["solutions"][-1],
                            stdin,
                            stdout,
                        ))
                    for future in as_completed(futures):
                        exec_succ, output, stdin, stdout = future.result()
                        pass_test = exec_succ and output.strip() == stdout.strip()
                        if not pass_test:
                            rich.print(f"[bold red]Test code failed for {source}")
                            print(example["solutions"][-1])
                            print(f"{exec_succ = }")
                            print(f"{repr(stdin) = }", f"{repr(stdout) = }")
                            if output.startswith(_ERROR_MSG_PREFIX):
                                print(f"{output = }")
                            else:
                                print(f"{repr(output) = }")
                            return None

                oracle = json.dumps({"inputs": stdin_list, "outputs": stdout_list})
            else:
                raise ValueError(f"Unknown ground truth format: {oracle}")

            prompt = "\n".join(prompt_pieces)
            data = {
                "data_source": "code",
                "prompt": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": oracle,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "prompt": prompt,
                    "reference": (example["solutions"][0] if example["solutions"] else ""),
                    "dataset": "likaixin/TACO-verified",
                },
            }
            return data

        return process_fn

    dataset = dataset.map(function=make_map_fn("train"), with_indices=True,
                          num_proc=min(64, cpu_count())).filter(lambda x: x is not None)

    N_TESTSET = 1000
    splits = dataset["train"].train_test_split(test_size=N_TESTSET, seed=666)
    train_dataset = splits["train"]
    test_dataset = splits["test"]

    return train_dataset, test_dataset


# Some tests are very broken and needs verification
def codecontests(max_n_tests=8):
    rich.print(Rule("Loading deepmind/code_contests..."))
    dataset = load_dataset("deepmind/code_contests")
    train_dataset = dataset["train"]
    test_dataset = dataset["valid"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            if "<image>" in example["description"]:
                print("Description includes image, skipping...")
                return None

            stdin_list = (example["public_tests"]["input"] + example["private_tests"]["input"] +
                          example["generated_tests"]["input"])
            stdout_list = (example["public_tests"]["output"] + example["private_tests"]["output"] +
                           example["generated_tests"]["output"])

            stdin_list, stdout_list = minimize_stdio(stdin_list, stdout_list, max_n_tests)
            assert len(stdin_list) == len(stdout_list)
            if len(stdin_list) == 0:
                return None

            prompt = ("Solve the programming task below in a Python markdown code block. "
                      "Each time, given inputs through STDIN (like those in the 'Input' section), the program "
                      "produces outputs through STDOUT (like those in the 'Output' section)."
                      f"\n\n{example['description'].strip()}")
            return {
                "data_source": "code",
                "prompt": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps({
                        "inputs": stdin_list,
                        "outputs": stdout_list
                    }),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "prompt": prompt,
                    "reference": (example["solutions"]["solution"][0] if example["solutions"]["solution"] else ""),
                    "dataset": "deepmind/code_contests",
                },
            }

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True).filter(lambda x: x is not None)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    from IPython import embed
    embed()
    return train_dataset, test_dataset


def leetcode2k():
    rich.print(Rule("Loading LeetCodeDataset..."))
    test_dataset = load_dataset("json",
                                data_files="LeetCodeDataset/data/LeetCodeDataset-v2-test-problems.jsonl")["train"]
    print("Test set:", test_dataset)

    train_dataset = concatenate_datasets([
        load_dataset(
            "json",
            data_files="LeetCodeDataset/data/LeetCodeDataset-v2-rl-problems.jsonl",
        )["train"],
        load_dataset(
            "json",
            data_files="LeetCodeDataset/data/LeetCodeDataset-v2-sft-problems.jsonl",
        )["train"],
    ]).filter(
        lambda example: example["meta"]["question_id"] not in set([d["question_id"] for d in test_dataset["meta"]]))
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
            prompt = f"Please solve the programming task below using a self-contained code snippet in a markdown code block.\n\n{example['meta']['query'].strip()}"
            return {
                "data_source": "code",
                "prompt": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "ability": "coding",
                "reward_model": {
                    "style":
                        "rule",
                    "ground_truth":
                        json.dumps({"functional": f"{example['test']}\n\ncheck({example['entry_point'].strip()})"}),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "reference": example["completion"],  # C++?
                    "prompt": prompt,
                    "dataset": "LeetCodeDataset",
                },
            }

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    return train_dataset, test_dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="./data/")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    root_dir = args.root_dir
    hdfs_dir = args.hdfs_dir

    train_datasets = []
    test_datasets = []

    for train, test in [leetcode2k(), taco()]:
        train_datasets.append(train)
        test_datasets.append(test)

    train_dataset = concatenate_datasets(train_datasets).shuffle(seed=666)
    test_dataset = concatenate_datasets(test_datasets)

    rich.print(Rule("Saving the final dataset"))
    print("Train set:", train_dataset)
    print("Test set:", test_dataset)

    local_dir = os.path.join(root_dir, f"code-r1-{round(len(train_dataset) / 1000)}k")
    rich.print(f"[bold green]Saving to {local_dir}...")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=root_dir, dst=hdfs_dir)
