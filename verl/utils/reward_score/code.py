# TODO: CodeGuru
# TODO: CodeQL
# TODO: Languages beyond Python

import re
import os
from traceback import format_exc
from typing import Tuple, Optional
import time
import requests
import json

def exec_test(
    server: str, code: str, timeout: int = 30, timeout_on_client: bool = False
) -> Tuple[bool, str]:
    assert isinstance(timeout, int), "Timeout needs to be an integer"
    while True:  # loop for server downtime
        try:
            headers = {"Content-Type": "application/json"}
            r = requests.post(
                server + "/py_exec",
                data=json.dumps({"code": code, "timeout": timeout}),
                timeout=(timeout * 1.5) if timeout_on_client else None,
                headers=headers,
            )
            resp, outs = r.text.split("\n", 1)
            assert resp == "0" or resp == "1"
            return resp == "0", outs
        except Exception:
            if not check_executor_alive(server): # check if the server is alive
                print("Request rejected, waiting 3 seconds and then retrying...")
                time.sleep(3)
                continue

            return False, "Failed to execute program: " + format_exc()

def check_executor_alive(executor):
    try:
        r = requests.get(executor + "/")
        return r.status_code == 200 or r.status_code == 404
    except Exception:
        return False

# https://github.com/Unakar/Logic-RL/blob/main/verl/utils/reward_score/kk.py
def validate_response_structure(processed_str: str) -> bool:
    validation_log = ["\n[Structure Validation]"]
    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        validation_log.append(f"  {tag_str}: count={count}, position={pos}")

        if count != expected_count:
            validation_log.append(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            return False, "\n".join(validation_log)

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        validation_log.append("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        return False, "\n".join(validation_log)
    else:
        validation_log.append("  Tag sequence validation passed")

    return True, "\n".join(validation_log)

# https://github.com/Unakar/Logic-RL/blob/main/verl/utils/reward_score/kk.py
def try_extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if matches:
        final_answer = matches[-1].group(1).strip()
        return final_answer

    return solution_str

CODE_PATTERN = re.compile(r'```(?:\w+)?\n(.*?)\n```', re.DOTALL)
def extract_code_from_string(solution_str):
    solution_str = try_extract_solution(solution_str)
    code_blocks = CODE_PATTERN.findall(solution_str)
    return '\n'.join(code_blocks).strip()

CODE_RPC_URL = os.getenv("CODE_RPC_URL", "http://localhost:8000")
def _compute_score(solution_str, ground_truth, format_reward=0.5, answer_reward=1.):
    reward_log = "-"*16 + "Model Output" + "-"*16 + "\n" + solution_str + "\n"

    # ground_truth is not code, but tests
    pass_fmt, validation_log = validate_response_structure(solution_str)
    solution_code = extract_code_from_string(solution_str)

    if not pass_fmt or len(solution_code) == 0:
        reward_log += "-"*16 + "Bad format detected!" + "-"*16 + "\n"
        reward_log += validation_log + "\n"
        return - answer_reward - format_reward, reward_log

    reward_log += "-"*16 + "Code Execution" + "-"*16 + "\n" + solution_code + "\n"
    pass_test, output = exec_test(CODE_RPC_URL, solution_code + "\n" + ground_truth)
    if not pass_test:
        reward_log += "-"*16 + "Code Execution Failed! (Exception)" + "-"*16 + "\n" + output + "\n"
        return format_reward, reward_log

    reward_log += "-"*16 + "Code Execution Passed! (Output)" + "-"*16 + "\n" + output + "\n"
    return format_reward + answer_reward, reward_log

def compute_score(solution_str, ground_truth, format_reward=0.5, answer_reward=1.):
    score, reward_log = _compute_score(solution_str, ground_truth, format_reward, answer_reward)
    reward_log = "="*16 + "Reward Calculation" + "="*16 + "\n"+  reward_log + "\n"+ "="*16 + f"Final Rward = {score}" + "="*16
    print(reward_log)
    return score
