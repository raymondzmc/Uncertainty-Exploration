# GPU Float16/Float32 Fix Applied

## Problem
RuntimeError: expected scalar type Float but found Half

This occurred when running DeBERTa NLI models on GPU with float16 precision.

## Root Cause
DeBERTa models have known issues with mixed precision (float16) inference. The attention computation expects float32 but receives float16 tensors.

## Solution Applied

### 1. NLI Model Always Uses Float32
```python
# DeBERTa models are kept in float32 even on GPU
if torch_dtype is not None and "deberta" not in model_name.lower():
    kwargs["torch_dtype"] = torch_dtype
```

### 2. Disabled Autocast for NLI
```python
# Ensure float32 computation for NLI
with torch.autocast(device_type=device, enabled=False):
    logits = self.model(**inputs).logits[0].float()
```

### 3. LM Model Can Still Use Float16
The main language model (Llama-3.2-1B) can still use float16 to save memory, only the NLI model uses float32.

## Memory Impact
- **Llama-3.2-1B (float16)**: ~2-3 GB VRAM
- **DeBERTa-base NLI (float32)**: ~1.5 GB VRAM
- **Total**: ~4-5 GB VRAM (fits on most GPUs)

## To Run on GPU

```bash
# Quick test
python run_full_evaluation.py --config fast

# Full benchmark
python run_full_evaluation.py --config comprehensive

# Or use the script
chmod +x run_gpu_eval.sh
./run_gpu_eval.sh comprehensive
```

The fix is now integrated - just run the commands above! 