# AUROC NaN Issue - Fixed

## Problem
The Predictive Entropy AUROC was showing as `nan` in evaluation results.

## Root Cause
AUROC (Area Under ROC Curve) requires both positive and negative examples to compute. When all examples belong to a single class (all should_abstain=True OR all should_abstain=False), ROC curves cannot be computed, resulting in NaN.

This happened because:
1. GSM8K dataset has abstention examples starting very late (index ~1213)
2. With only 5 random samples, we were unlikely to get both classes
3. Scikit-learn correctly returns NaN when only one class is present

## Solutions Implemented

### 1. **Balanced Sampling** (abstention_semantic_entropy.py)
Added balanced sampling to try to get equal numbers of abstain/answer examples:
```python
def load_abstention_dataset(..., balanced_sampling: bool = True):
    # Now attempts to sample equally from both classes when available
```

### 2. **Better Parsing** (run_full_evaluation.py)
- Fixed parsing to correctly detect "N/A" values
- Updated summary generation to handle None values gracefully
- Shows informative message: "N/A (insufficient class diversity)"

### 3. **Configuration Updates**
- Changed "fast" config from 5 to 20 examples
- Switched from GSM8K to SQuAD2 (better class balance)
- This increases likelihood of getting both classes

### 4. **Documentation**
Added explanation in README_semantic_abstention.md about:
- Why AUROC might be N/A
- Dataset characteristics  
- Minimum recommended sample sizes

## Expected Behavior

### With Small Samples (< 20)
- AUROC may show as "N/A" - this is correct behavior
- Indicates insufficient class diversity
- Solution: Increase sample size

### With Adequate Samples (20+)
- Should get valid AUROC scores
- Balanced sampling helps ensure both classes are represented
- Meaningful uncertainty quantification metrics

## Dataset Notes

| Dataset | Abstention Distribution | Min Samples for AUROC |
|---------|------------------------|----------------------|
| GSM8K | Late (index 1213+) | 50+ recommended |
| SQuAD2 | Mixed throughout | 20+ usually sufficient |
| FalseQA | All abstain | N/A (single class) |
| SelfAware | Mixed | 20+ usually sufficient |
| WorldSense | Mixed | 20+ usually sufficient |

## Testing

To verify the fix works:
```bash
# Quick test with better balance
python run_full_evaluation.py --config fast

# Should now show either:
# - Valid AUROC scores (if both classes present)
# - "N/A (insufficient class diversity)" (if single class)
```

## Key Takeaway

**NaN AUROC is not a bug** - it's the mathematically correct result when computing ROC curves with only one class. The fix ensures this is handled gracefully and communicated clearly to users. 