import re
from typing import Tuple, Optional
import time
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .utils import _ERROR_MSG_PREFIX

_MAX_CHAR_DISPLAY = 2048
CODER1_EXEC = os.environ.get("CODER1_EXEC", "firejail")

if CODER1_EXEC == "firejail":
    from .firejail_exec import code_exec_firejail
    code_exec = code_exec_firejail
elif CODER1_EXEC == "ces":
    from .ces_exec import remote_code_exec_ces
    code_exec = remote_code_exec_ces
elif CODER1_EXEC == "kira":
    from .kira_exec import remote_code_exec_kira
    code_exec = remote_code_exec_kira
else:
    raise ValueError(f"Unknown CODER1_EXEC: {CODER1_EXEC}")


def remote_check_stdio(code, stdin, stdout):
    succ, output = code_exec(code=code, stdin=stdin)
    return succ, output, stdin, stdout


# https://github.com/Unakar/Logic-RL/blob/main/verl/utils/reward_score/kk.py
def validate_response_structure(processed_str: str) -> bool:
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
        positions[tag_name] = processed_str.find(tag_str)

        if count != expected_count:
            return False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or positions['think_end'] > positions['answer_start'] or
            positions['answer_start'] > positions['answer_end']):
        return False

    return True


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


def _compute_score(solution_str, ground_truth, extra_info, format_reward=0.1, answer_reward=1.):
    reward_log = []

    # ground_truth is not code, but tests
    pass_fmt = validate_response_structure(solution_str)
    solution_code = extract_code_from_string(solution_str)

    if not pass_fmt or len(solution_code) == 0:  # only print full output when there is an error
        reward_log.append("-" * 16 + "Bad format detected!" + "-" * 16)
        reward_log.append("-" * 16 + "Original Model Output" + "-" * 16)
        reward_log.append(solution_str)
        return -answer_reward - format_reward, "\n".join(reward_log)

    reward_log.append("-" * 16 + "Extracted Code to Execute" + "-" * 16)
    ground_truth = json.loads(ground_truth)

    t_start = time.time()

    # log code
    if "functional" in ground_truth:
        reward_log.append(solution_code + "\n" + ground_truth["functional"])
    else:
        reward_log.append(solution_code)

    if "functional" in ground_truth:
        succ, output = code_exec(solution_code + "\n" + ground_truth["functional"])
        if not succ:
            reward_log.append("!" * 16 + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s" + "!" * 16)
            reward_log.append(output[:_MAX_CHAR_DISPLAY])
            reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
            reward_log.append(extra_info["prompt"].replace("\n\n", "\n"))
            return format_reward, "\n".join(reward_log)
    elif "inputs" in ground_truth and "outputs" in ground_truth:
        stdin_list: str = ground_truth["inputs"]
        stdout_list: str = ground_truth["outputs"]

        # for stdin, stdout in zip(stdin_list, stdout_list):
        #     succ, output, stdin, stdout = remote_check_stdio(code=solution_code, stdin=stdin, stdout=stdout)
        #     if not succ or output.strip() != stdout.strip():
        #         reward_log.append("!" * 16 + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s" + "!" * 16)
        #         reward_log.append(f"🔎Input: {repr(stdin)}")
        #         reward_log.append(f"✅Expected: {repr(stdout.strip())}")
        #         reward_log.append(
        #             f"❌Actual: {output if output.startswith(_ERROR_MSG_PREFIX) else repr(output.strip())}")
        #         reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
        #         reward_log.append(extra_info["prompt"].replace("\n\n", "\n"))
        #         return format_reward, "\n".join(reward_log)

        # Add parallelism
        with ThreadPoolExecutor(max_workers=min(8, len(stdin_list))) as executor:
            futures = [
                executor.submit(remote_check_stdio, solution_code, stdin, stdout)
                for stdin, stdout in zip(stdin_list, stdout_list)
            ]
            for future in as_completed(futures):
                succ, output, stdin, stdout = future.result()
                if not succ or output.strip() != stdout.strip():
                    output = output[:_MAX_CHAR_DISPLAY]  # truncate output to print
                    reward_log.append("!" * 16 + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s" + "!" * 16)
                    reward_log.append(f"🔎Input: {repr(stdin)}")
                    reward_log.append(f"✅Expected: {repr(stdout.strip())}")
                    reward_log.append(
                        f"❌Actual: {output if output.startswith(_ERROR_MSG_PREFIX) else repr(output.strip())}")
                    reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
                    reward_log.append(extra_info["prompt"].replace("\n\n", "\n"))
                    return format_reward, "\n".join(reward_log)
    else:
        raise ValueError(
            f"Current supports for ground-truth are ['functional', 'inputs/outputs'] -- No idea what's: {ground_truth = }"
        )

    reward_log.append("+" * 16 + "Test Execution Passed! (Output)" + "+" * 16)
    reward_log.append(output)
    return format_reward + answer_reward, "\n".join(reward_log)


def compute_score(solution_str, ground_truth, extra_info, format_reward=0.1, answer_reward=1.):
    if isinstance(extra_info, np.ndarray):
        extra_info = extra_info.item()
    score, reward_log = _compute_score(solution_str,
                                       ground_truth,
                                       extra_info=extra_info,
                                       format_reward=format_reward,
                                       answer_reward=answer_reward)
    marker = "✅" if score == (format_reward + answer_reward) else "❌"
    reward_log = marker * 16 + "Reward Calculation" + marker * 16 + "\n" + reward_log + "\n" + marker * 16 + f"Final Rward = {score}" + marker * 16
    print(reward_log + "\n\n")
    return score
