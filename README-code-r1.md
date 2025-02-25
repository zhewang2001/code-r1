# Code R1

## Setup

```bash
# For sandboxing
sudo add-apt-repository ppa:deki/firejail
sudo apt-get update
sudo apt-get install firejail firejail-profiles

# For training
pip install vllm==0.6.3 torch==2.4.0 ray
pip install flash-attn --no-build-isolation
pip install -e .  # For verl integration
pip install wandb IPython matplotlib gpustat
```