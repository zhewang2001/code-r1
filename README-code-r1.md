Use the setup script below to avoid environment issues:

```bash
pip install vllm==0.6.3 torch==2.4.0 ray
pip3 install flash-attn --no-build-isolation
pip install -e .  # For verl integration
pip install wandb IPython matplotlib gpustat
```