#!/bin/bash
# GPU Evaluation Script for Semantic Entropy + AbstentionBench

echo "=========================================="
echo "Running GPU Evaluation with Llama-3.2-1B"
echo "=========================================="

# Check if running on GPU
if ! nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: nvidia-smi not found. Are you on a GPU machine?"
fi

# Configuration
CONFIG=${1:-comprehensive}  # Default to comprehensive, or pass 'fast' for testing

echo ""
echo "Configuration: $CONFIG"
echo "Model: meta-llama/Llama-3.2-1B-Instruct"
echo ""

# Run the evaluation
python run_full_evaluation.py --config $CONFIG

# Alternative: Direct run with specific parameters
# python abstention_semantic_entropy.py \
#     --dataset gsm8k \
#     --subset_size 100 \
#     --model "meta-llama/Llama-3.2-1B-Instruct" \
#     --nli_model "microsoft/deberta-v3-base-mnli" \
#     --num_samples 15 \
#     --temperature 0.7 \
#     --max_new_tokens 64 \
#     --device cuda \
#     --use_system_prompt \
#     --seed 42

echo ""
echo "Evaluation complete!"
echo "Check results/ directory for outputs" 