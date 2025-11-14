
Install transformers with commit ce40ca0d4c7d

```bash
# If you already have transformers installed, please uninstall it first, or create a new Python environment
# pip uninstall transformers
git clone https://github.com/huggingface/transformers
cd transformers
git checout ce40ca0d4c7d
uv pip install . 

uv pip install accelerate
uv pip install flash-attn --no-build-isolation
```


Run qwen3omni on a single card (~80G mem)
```
export CUDA_VISIBLE_DEVICES=0
python test_infer.py
```

TODOs:
- [ ] audio input raises audioread.ffdec.NotInstalledError
