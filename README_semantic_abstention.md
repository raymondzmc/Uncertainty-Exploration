# Semantic Entropy for AbstentionBench

This project integrates **Semantic Entropy** (Kuhn et al., ICLR 2023) with **AbstentionBench** to quantify model uncertainty and evaluate its effectiveness for abstention decisions.

## Overview

The integration combines:
- **Semantic Entropy**: A method that clusters semantically equivalent model outputs to compute uncertainty
- **AbstentionBench**: A benchmark for evaluating when LLMs should abstain from answering

### Key Hypothesis
Models with high semantic entropy (high uncertainty) should correlate with cases where abstention is appropriate. This integration tests whether semantic entropy can be used as a reliable signal for abstention decisions.

## Architecture

```
Question → System Prompt → Llama-3.2-1B → M Samples
                                           ↓
                                    NLI Clustering
                                           ↓
                                    Semantic Entropy
                                           ↓
                                    Compare with
                                    should_abstain
```

## Installation

### 1. Install Dependencies

```bash
# Install core requirements
pip install -r requirements_semantic_abstention.txt

# For GPU support (recommended), install PyTorch with CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 2. Set up AbstentionBench

Make sure the AbstentionBench repository is available in the same directory.

## Usage

### Quick Start

```bash
# Make the script executable
chmod +x run_semantic_abstention.sh

# Run with default settings (GSM8K dataset, 50 examples)
./run_semantic_abstention.sh

# Run with custom dataset and size
./run_semantic_abstention.sh squad2 100
```

### Python API

```python
python abstention_semantic_entropy.py \
    --dataset gsm8k \
    --subset_size 100 \
    --model "meta-llama/Llama-3.2-1B-Instruct" \
    --num_samples 10 \
    --temperature 0.7 \
    --use_system_prompt
```

### Available Datasets

- `gsm8k`: Math word problems with/without context
- `squad2`: Reading comprehension with unanswerable questions
- `falseqa`: Questions with false premises
- `self_aware`: Self-knowledge questions
- `world_sense`: Common sense reasoning

## Experiment Design

### 1. System Prompt Integration
The script uses AbstentionBench's system prompt which includes explicit abstention instructions:
- "If a question is unanswerable... respond 'I don't know'"
- Instructions for handling unclear, ambiguous, or false-premise questions

### 2. Semantic Entropy Computation
For each question:
1. Generate M samples (default: 10) from the model
2. Cluster samples by bidirectional entailment using an NLI model
3. Compute entropy over the cluster distribution
4. Higher entropy = higher uncertainty

### 3. Evaluation Metrics
- **AUROC**: Area Under ROC curve for using entropy to predict `should_abstain`
- **Entropy Statistics**: Compare entropy distributions for abstain vs. answer cases
- **Majority Vote Accuracy**: Accuracy of abstention based on majority of samples

## Expected Results

### Ideal Outcomes
- **Higher semantic entropy** for questions that should trigger abstention
- **Lower semantic entropy** for answerable questions
- **AUROC > 0.7** indicating good discrimination

### Interpretation
- **AUROC ≈ 0.5**: Semantic entropy is not predictive of abstention needs
- **AUROC > 0.7**: Good correlation - entropy can guide abstention
- **AUROC > 0.8**: Strong signal - entropy is highly predictive

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_samples` | 10 | Samples per question for entropy |
| `--temperature` | 0.7 | Sampling temperature |
| `--max_new_tokens` | 64 | Maximum response length |
| `--subset_size` | 100 | Examples to evaluate |
| `--use_system_prompt` | True | Use AbstentionBench prompt |

## Model Configuration

### Language Model
- **Default**: `meta-llama/Llama-3.2-1B-Instruct`
- Small enough for single GPU (< 8GB VRAM)
- Chat-tuned for instruction following

### NLI Model  
- **Default**: `microsoft/deberta-v3-base-mnli`
- Used for semantic clustering via entailment
- Smaller than deberta-large for memory efficiency

## Memory Requirements

- **GPU**: ~6-8 GB VRAM for default models
- **CPU**: Possible but significantly slower
- **Optimization**: Reduce `num_samples` or `subset_size` if needed

## Important Notes

### AUROC Scores and Class Balance

**Why AUROC might show as "N/A":**
- AUROC requires both positive (should abstain) and negative (should answer) examples
- Some datasets (like GSM8K) have imbalanced distributions where abstention examples appear later
- With small sample sizes (e.g., 5 examples in "fast" mode), you might only get one class
- This is expected behavior - increase `subset_size` to get more diverse examples

**Dataset Characteristics:**
- **GSM8K**: Abstention examples start around index 1213 (context-removed questions)
- **SQuAD2**: Has unanswerable questions mixed throughout
- **FalseQA**: All questions have false premises (should abstain)
- **SelfAware**: Mix of answerable and unanswerable self-knowledge questions

For meaningful AUROC scores, use at least 20-50 examples per dataset.

## Troubleshooting

### Out of Memory
```python
# Use smaller models
--model "facebook/opt-350m"
--nli_model "roberta-base-mnli"

# Reduce batch processing
--num_samples 5
--subset_size 25
```

### Slow Performance
- Ensure CUDA is available: `torch.cuda.is_available()`
- Use GPU: `--device cuda`
- Reduce number of samples

## Scientific Soundness

This experiment maintains scientific rigor by:

1. **Proper Prompting**: Uses the same system prompt as AbstentionBench
2. **Fair Comparison**: Evaluates on datasets with known abstention labels
3. **Multiple Metrics**: AUROC, accuracy, and statistical analysis
4. **Reproducibility**: Fixed seeds and documented parameters
5. **Diverse Datasets**: Tests across different abstention scenarios

## References

- **Semantic Entropy**: Kuhn, Gal, and Farquhar. "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation." ICLR 2023.
- **AbstentionBench**: Kirichenko et al. "AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions." 2025.

## Citation

If you use this integration, please cite both papers:

```bibtex
@inproceedings{kuhn2023semantic,
  title={Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation},
  author={Kuhn, Lorenz and Gal, Yarin and Farquhar, Sebastian},
  booktitle={ICLR},
  year={2023}
}

@misc{kirichenko2025abstentionbench,
  title={AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions}, 
  author={Polina Kirichenko and Mark Ibrahim and Kamalika Chaudhuri and Samuel J. Bell},
  year={2025},
  eprint={2506.09038},
  archivePrefix={arXiv}
}
``` 