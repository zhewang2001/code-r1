# Code-R1: Reproducing R1 for Code with Reliable Rewards

This repository includes implementations to reproduce the R1 pipeline for code generation:

* **Result:** 2K Code-R1 samples + `Qwen2.5-7B-Instruct-1M` beats `Qwen2.5-Coder-7B-Instruct` (even better w/ 12K samples).
* **Finding:** Quality of rewards matters. False positives in datasets and execution confuse the model.
* **Implementation:** A reliable, scalable, and sandboxed pipeline to minimize reward false positives in datasets and execution.

More results and findings to come...

## Setup

### Environment

```bash
# For training
pip install -e .
pip install vllm==0.7.3
pip install flash-attn --no-build-isolation
pip install wandb IPython matplotlib gpustat # utility
```

### Sandboxing

I tried multiple ways for sandboxing including calling code execution servers, running dockerized Python, calling paid services, etc.
`firejail` is the approach I found to meet all the three:

1. Reliability -- False positive comes when "autograders" have concurrency issue (timeouts), violating OS limits (`OSError`), etc.
2. Scalability -- e.g., dockerized Python run is generally reliable but too slow (e.g., 20 samples/s on 192 cores).
3. Security -- ... otherwise the school IT will email you and stop your server...

```bash
sudo add-apt-repository ppa:deki/firejail
sudo apt-get update
sudo apt-get install firejail firejail-profiles
```

### Datasets

The current version has 12K RL samples (prompt + tests):
* [2K LeetCode data](https://github.com/newfacade/LeetCodeDataset) where the tests are generally reliable
* 10K verified data filtered from 26K [TACO](https://huggingface.co/datasets/BAAI/TACO) data. 

In general, it's suggesgted to test data & sandbox on every dataset & environment before training code RL.
Directly using noisy data and mismatched envornments can lead to reward false positives, confusing the model.
These noise could come from (i) wrong tests, (ii) unsolvable prompts (e.g., images tags), and (iii) execution environment mismatch.

Run the command below to produce validated RL data for code:

```bash
python examples/data_preprocess/coder1.py
```

## Code-R1 Zero based on 7B models

Starting model and our RL-trained model (no distillation):

|                    Model                       |     LCB (v5)  |   HumanEval+   |    MBPP+    | **Average** |
|------------------------------------------------|---------------|----------------|-------------|------------:|
| Qwen2.5-7B-Instruct-1M                         |     24.0      |     80.5       |    66.7     |   57.1      |
| + Code-R1-Zero (2k  - 1088s GRPO)               |     28.6      |     84.8       |    70.1     |   61.2      |
| + Code-R1-Zero (12k -  832s GRPO)               |     29.7      |     83.5       |    74.3     | 🌟**62.5**  |

* 2K leetcode training samples can already show promising results without any additional SFT or distillation.
* Adding it to 12K data (10K more verified data from TACO) can further improve the performance.

Some existing models:

|                    Model                       |     LCB (v5)  |   HumanEval+   |    MBPP+    | **Average** |
|------------------------------------------------|---------------|----------------|-------------|------------:|
| Qwen2.5-Coder-7B-Instruct                      |     31.1      |     82.3       |    69.6     |  61.0       |
| Eurus-2-7B-PRIME                               |     23.8      |     65.9       |    29.9     |  39.9       |
| Sky-T1-7B                                      |     21.3      |     54.3       |    50.8     |  42.1       |

* Qwen2.5-Coder-7B-Instruct, despite released months back, is still very performant as the best baseline, but we don't know where the improvement comes from.
* Eurus-2-7B-PRIME starts from Qwen2.5-Math-7B-Instruct and is RL only. Its training data includes (unfiltered) extensive coding datasets, including APPS, CodeContests, TACO, and Codeforces. Code-R1-Zero outperforms it significantly despite using fewer data, likely because we use validated datasets and sandboxes.
* Sky-T1-7B uses a combination of RL and SFT/distillation steps. Its RL partially uses PRIME but its training data does not (seem to) include coding datasets.

## Citation

If you find this work helpful...

```bibtex
@article{code-r1,
  title={Code-R1: Reproducing R1 for Code with Reliable Rewards},
  author={Liu, Jiawei and Zhang, Lingming},
  howpublished={\url{https://github.com/ganler/code-r1}},
  year={2025}
}
```

## Acknowledgements

* [Verl](https://github.com/volcengine/verl)
* [Logic-RL](https://github.com/Unakar/Logic-RL)

## License

Apache-2.0. See [LICENSE.code-r1](LICENSE.code-r1) for more details.
