"""
Semantic Entropy for AbstentionBench
=====================================

This script integrates semantic entropy with AbstentionBench to quantify
uncertainty and evaluate its effectiveness for abstention decisions.

Key features:
- Uses AbstentionBench datasets and system prompt
- Computes semantic entropy for each question
- Evaluates correlation between entropy and abstention labels
- Supports Llama-3.2-1B model for single GPU execution

Usage:
------
python abstention_semantic_entropy.py \
    --dataset gsm8k \
    --subset_size 100 \
    --num_samples 10 \
    --temperature 0.7

"""

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# Add AbstentionBench to path
sys.path.append(str(Path(__file__).parent / "AbstentionBench"))

from recipe.abstention_datasets.abstract_abstention_dataset import Prompt
from recipe.abstention_datasets.gsm8k import GSM8K
from recipe.abstention_datasets.squad import Squad2Dataset
from recipe.abstention_datasets.false_qa import FalseQADataset
from recipe.abstention_datasets.self_aware import SelfAwareDataset
from recipe.abstention_datasets.world_sense import WorldSenseDataset
from recipe.system_prompt import SYSTEM_PROMPT


# ----------------------------
# Utility functions
# ----------------------------

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_whitespace(s: str) -> str:
    """Normalize whitespace in a string."""
    return re.sub(r"\s+", " ", s.strip())


# ----------------------------
# Language Model Wrapper
# ----------------------------

@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = True


class CausalLMWithSystemPrompt:
    """Causal LM wrapper that uses AbstentionBench system prompt."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        device: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
        max_seq_len: int = 2048,
        use_system_prompt: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.use_system_prompt = use_system_prompt
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        kwargs = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
            
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.to(device)
        self.model.eval()
        self.max_seq_len = max_seq_len
    
    def format_prompt(self, question: str) -> str:
        """Format question with system prompt using chat template."""
        messages = []
        
        if self.use_system_prompt:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        
        messages.append({"role": "user", "content": question})
        
        # Try to apply chat template, fallback to simple format if not available
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted
        except:
            # Fallback for models without chat templates (e.g., GPT-2)
            if self.use_system_prompt:
                return f"{SYSTEM_PROMPT}\n\nQuestion: {question}\nAnswer:"
            else:
                return f"Question: {question}\nAnswer:"
    
    @torch.inference_mode()
    def generate_samples(
        self,
        question: str,
        num_samples: int,
        gen_cfg: GenerationConfig,
    ) -> Tuple[List[str], List[float]]:
        """Generate multiple samples and compute log probabilities."""
        
        # Format with system prompt
        prompt = self.format_prompt(question)
        
        # Tokenize
        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len
        ).to(self.device)
        
        samples = []
        logprobs = []
        
        for _ in range(num_samples):
            # Generate
            outputs = self.model.generate(
                **prompt_ids,
                max_new_tokens=gen_cfg.max_new_tokens,
                temperature=gen_cfg.temperature,
                top_p=gen_cfg.top_p,
                top_k=gen_cfg.top_k,
                do_sample=gen_cfg.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )
            
            # Extract completion
            full_ids = outputs.sequences[0]
            prompt_len = prompt_ids["input_ids"].shape[1]
            completion_ids = full_ids[prompt_len:]
            completion_text = self.tokenizer.decode(
                completion_ids, 
                skip_special_tokens=True
            ).strip()
            
            samples.append(completion_text)
            
            # Compute log probability
            concat_ids = self.tokenizer(
                prompt + completion_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len,
            )["input_ids"].to(self.device)
            
            logits = self.model(concat_ids).logits
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Sum log probs over completion tokens
            lp = 0.0
            for t in range(prompt_len, concat_ids.shape[1]):
                token_id = concat_ids[0, t].item()
                lp += log_probs[0, t - 1, token_id].item()
            
            logprobs.append(lp)
        
        return samples, logprobs


# ----------------------------
# NLI Model for Semantic Clustering
# ----------------------------

class NLIModel:
    """NLI model for bidirectional entailment."""
    
    def __init__(
        self, 
        model_name: str = "microsoft/deberta-v3-base-mnli",  # Smaller than large
        device: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        kwargs = {}
        # Force float32 for NLI models to avoid dtype issues
        # DeBERTa models have issues with float16
        if torch_dtype is not None and "deberta" not in model_name.lower():
            kwargs["torch_dtype"] = torch_dtype
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, **kwargs
        )
        self.model.to(device)
        self.model.eval()
        self.device = device
        
        # Find entailment index
        id2label = self.model.config.id2label
        self.label2id = {v.lower(): int(k) for k, v in id2label.items()}
        self.entail_idx = self.label2id.get("entailment", 2)
    
    @torch.inference_mode()
    def entailment_prob(self, premise: str, hypothesis: str) -> float:
        """Compute entailment probability."""
        inputs = self.tokenizer(
            premise, 
            hypothesis, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Move to device and ensure float32
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)
        
        # Get logits and ensure float32
        with torch.autocast(device_type=self.device.split(':')[0] if ':' in self.device else self.device, enabled=False):
            logits = self.model(**inputs).logits[0].float()
        
        probs = torch.softmax(logits, dim=-1)
        return float(probs[self.entail_idx].item())


# ----------------------------
# Semantic Clustering
# ----------------------------

@dataclass
class SemanticCluster:
    indices: List[int]


def cluster_by_bidirectional_entailment(
    context: str,
    answers: List[str],
    nli: NLIModel,
    entail_threshold: float = 0.5,
) -> List[SemanticCluster]:
    """Cluster answers by bidirectional entailment."""
    clusters: List[SemanticCluster] = []
    
    def format_context_answer(ans: str) -> str:
        if context.strip():
            return f"{context}\nAnswer: {ans.strip()}"
        else:
            return f"Answer: {ans.strip()}"
    
    for i, ans in enumerate(answers):
        placed = False
        for cluster in clusters:
            rep_idx = cluster.indices[0]
            rep_ans = answers[rep_idx]
            
            # Bidirectional entailment check
            p1 = nli.entailment_prob(
                format_context_answer(rep_ans),
                format_context_answer(ans)
            )
            p2 = nli.entailment_prob(
                format_context_answer(ans),
                format_context_answer(rep_ans)
            )
            
            if p1 >= entail_threshold and p2 >= entail_threshold:
                cluster.indices.append(i)
                placed = True
                break
        
        if not placed:
            clusters.append(SemanticCluster(indices=[i]))
    
    return clusters


# ----------------------------
# Entropy Calculations
# ----------------------------

def predictive_entropy(logprobs: List[float]) -> float:
    """Compute predictive entropy."""
    if len(logprobs) == 0:
        return float("nan")
    return -float(np.mean(logprobs))


def semantic_entropy(
    clusters: List[SemanticCluster],
    logprobs: List[float],
) -> float:
    """Compute semantic entropy over clusters."""
    if len(clusters) == 0:
        return float("nan")
    
    # Convert log probs to probabilities
    p = np.exp(np.array(logprobs, dtype=np.float64))
    
    # Aggregate probability per cluster
    p_clusters = []
    for cluster in clusters:
        cluster_prob = float(p[cluster.indices].sum())
        p_clusters.append(cluster_prob)
    
    p_clusters = np.array(p_clusters, dtype=np.float64)
    
    # Avoid log(0)
    eps = 1e-300
    p_clusters = np.clip(p_clusters, eps, 1.0)
    
    # Compute entropy
    se = -float(np.mean(np.log(p_clusters)))
    return se


# ----------------------------
# Abstention Detection
# ----------------------------

def contains_abstention_keywords(response: str) -> bool:
    """Check if response contains abstention keywords."""
    abstention_phrases = [
        "i don't know",
        "i do not know",
        "i'm not sure",
        "i am not sure",
        "i cannot",
        "i can't",
        "unable to",
        "insufficient",
        "unclear",
        "uncertain",
        "not enough information",
        "cannot determine",
        "cannot answer",
        "need more information",
        "ambiguous",
        "it depends",
    ]
    
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in abstention_phrases)


# ----------------------------
# Dataset Loading
# ----------------------------

def load_abstention_dataset(
    dataset_name: str,
    subset_size: int,
    seed: int = 42,
    balanced_sampling: bool = True
) -> List[Dict]:
    """Load AbstentionBench dataset and format for evaluation.
    
    Args:
        dataset_name: Name of the dataset to load
        subset_size: Number of examples to sample
        seed: Random seed for reproducibility
        balanced_sampling: If True, try to balance abstain/answer examples
    """
    
    # Initialize dataset
    if dataset_name == "gsm8k":
        dataset = GSM8K(split="test", max_num_samples=subset_size * 4)
    elif dataset_name == "squad2":
        dataset = Squad2Dataset(split="validation", max_num_samples=subset_size * 4)
    elif dataset_name == "falseqa":
        dataset = FalseQADataset(max_num_samples=subset_size * 4)
    elif dataset_name == "self_aware":
        dataset = SelfAwareDataset(max_num_samples=subset_size * 4)
    elif dataset_name == "world_sense":
        dataset = WorldSenseDataset(max_num_samples=subset_size * 4)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    rng = np.random.default_rng(seed)
    
    if balanced_sampling:
        # Try to get balanced samples
        abstain_indices = []
        answer_indices = []
        
        for i in range(len(dataset)):
            if dataset[i].should_abstain:
                abstain_indices.append(i)
            else:
                answer_indices.append(i)
        
        # Calculate how many of each type to sample
        n_abstain = len(abstain_indices)
        n_answer = len(answer_indices)
        
        if n_abstain > 0 and n_answer > 0:
            # Both types exist, try to balance
            target_each = min(subset_size // 2, n_abstain, n_answer)
            remaining = subset_size - (target_each * 2)
            
            # Sample from each category
            sampled_abstain = rng.choice(abstain_indices, size=target_each, replace=False)
            sampled_answer = rng.choice(answer_indices, size=target_each, replace=False)
            
            # Add remaining samples from whichever pool has more
            indices = list(sampled_abstain) + list(sampled_answer)
            
            if remaining > 0:
                all_remaining = list(set(abstain_indices + answer_indices) - set(indices))
                if all_remaining:
                    extra = rng.choice(all_remaining, size=min(remaining, len(all_remaining)), replace=False)
                    indices.extend(extra)
        else:
            # Only one type exists, sample normally
            indices = rng.choice(len(dataset), size=min(subset_size, len(dataset)), replace=False)
    else:
        # Random sampling
        indices = rng.choice(len(dataset), size=min(subset_size, len(dataset)), replace=False)
    
    data = []
    for idx in indices:
        prompt = dataset[int(idx)]
        data.append({
            "question": prompt.question,
            "references": prompt.reference_answers or [],
            "should_abstain": prompt.should_abstain,
            "metadata": prompt.metadata,
        })
    
    # Shuffle the data to mix abstain and answer examples
    rng.shuffle(data)
    
    return data


# ----------------------------
# Main Evaluation
# ----------------------------

@dataclass
class EvaluationResult:
    question: str
    should_abstain: bool
    samples: List[str]
    semantic_entropy: float
    predictive_entropy: float
    num_clusters: int
    abstained_samples: int
    majority_abstains: bool


def evaluate_with_semantic_entropy(
    dataset_name: str,
    lm: CausalLMWithSystemPrompt,
    nli: NLIModel,
    subset_size: int,
    num_samples: int,
    temperature: float,
    max_new_tokens: int,
    seed: int = 42,
) -> None:
    """Evaluate semantic entropy for abstention prediction."""
    
    set_seed(seed)
    
    # Load dataset
    print(f"\nLoading {dataset_name} dataset...")
    data = load_abstention_dataset(dataset_name, subset_size, seed)
    
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=1.0,
        top_k=0,
        do_sample=True,
    )
    
    results = []
    
    for example in tqdm(data, desc="Processing examples"):
        question = example["question"]
        should_abstain = example["should_abstain"]
        
        # Generate samples
        samples, logprobs = lm.generate_samples(
            question,
            num_samples,
            gen_cfg
        )
        
        # Compute entropies
        pe = predictive_entropy(logprobs)
        
        # Semantic clustering
        context = f"Question: {normalize_whitespace(question)}"
        clusters = cluster_by_bidirectional_entailment(
            context=context,
            answers=samples,
            nli=nli,
            entail_threshold=0.5,
        )
        se = semantic_entropy(clusters, logprobs)
        
        # Check abstention in samples
        abstained = [contains_abstention_keywords(s) for s in samples]
        abstained_count = sum(abstained)
        majority_abstains = abstained_count > len(samples) / 2
        
        results.append(EvaluationResult(
            question=question,
            should_abstain=should_abstain,
            samples=samples,
            semantic_entropy=se,
            predictive_entropy=pe,
            num_clusters=len(clusters),
            abstained_samples=abstained_count,
            majority_abstains=majority_abstains,
        ))
    
    # Analyze results
    analyze_results(results, dataset_name, num_samples)


def analyze_results(
    results: List[EvaluationResult],
    dataset_name: str,
    num_samples: int
) -> None:
    """Analyze and print evaluation results."""
    
    print("\n" + "="*60)
    print(f"RESULTS: {dataset_name}")
    print("="*60)
    
    # Separate by ground truth
    should_abstain_results = [r for r in results if r.should_abstain]
    should_answer_results = [r for r in results if not r.should_abstain]
    
    print(f"\nDataset Statistics:")
    print(f"  Total examples: {len(results)}")
    print(f"  Should abstain: {len(should_abstain_results)}")
    print(f"  Should answer: {len(should_answer_results)}")
    
    # Compute entropy statistics
    if should_abstain_results:
        se_abstain = [r.semantic_entropy for r in should_abstain_results if not math.isnan(r.semantic_entropy)]
        pe_abstain = [r.predictive_entropy for r in should_abstain_results if not math.isnan(r.predictive_entropy)]
        clusters_abstain = [r.num_clusters for r in should_abstain_results]
        
        print(f"\nShould Abstain - Entropy Statistics:")
        print(f"  Semantic Entropy: {np.mean(se_abstain):.3f} ± {np.std(se_abstain):.3f}")
        print(f"  Predictive Entropy: {np.mean(pe_abstain):.3f} ± {np.std(pe_abstain):.3f}")
        print(f"  Avg Clusters: {np.mean(clusters_abstain):.2f}")
    
    if should_answer_results:
        se_answer = [r.semantic_entropy for r in should_answer_results if not math.isnan(r.semantic_entropy)]
        pe_answer = [r.predictive_entropy for r in should_answer_results if not math.isnan(r.predictive_entropy)]
        clusters_answer = [r.num_clusters for r in should_answer_results]
        
        print(f"\nShould Answer - Entropy Statistics:")
        print(f"  Semantic Entropy: {np.mean(se_answer):.3f} ± {np.std(se_answer):.3f}")
        print(f"  Predictive Entropy: {np.mean(pe_answer):.3f} ± {np.std(pe_answer):.3f}")
        print(f"  Avg Clusters: {np.mean(clusters_answer):.2f}")
    
    # AUROC evaluation
    y_true = [1 if r.should_abstain else 0 for r in results]
    
    # Use entropy as uncertainty score (higher entropy = more likely to abstain)
    se_scores = [r.semantic_entropy if not math.isnan(r.semantic_entropy) else 0 for r in results]
    pe_scores = [r.predictive_entropy if not math.isnan(r.predictive_entropy) else 0 for r in results]
    cluster_scores = [r.num_clusters for r in results]
    
    # Majority vote abstention
    majority_preds = [1 if r.majority_abstains else 0 for r in results]
    
    print("\n" + "-"*40)
    print("AUROC Scores (higher is better):")
    print("-"*40)
    
    try:
        auroc_se = roc_auc_score(y_true, se_scores)
        print(f"  Semantic Entropy: {auroc_se:.3f}")
    except:
        print(f"  Semantic Entropy: N/A")
    
    try:
        auroc_pe = roc_auc_score(y_true, pe_scores)
        print(f"  Predictive Entropy: {auroc_pe:.3f}")
    except:
        print(f"  Predictive Entropy: N/A")
    
    try:
        auroc_clusters = roc_auc_score(y_true, cluster_scores)
        print(f"  Number of Clusters: {auroc_clusters:.3f}")
    except:
        print(f"  Number of Clusters: N/A")
    
    # Accuracy of majority vote
    majority_accuracy = sum([1 for pred, true in zip(majority_preds, y_true) if pred == true]) / len(y_true)
    print(f"\nMajority Vote Abstention Accuracy: {majority_accuracy:.3f}")
    
    # Sample outputs
    print("\n" + "-"*40)
    print("Sample Outputs (first 3 examples):")
    print("-"*40)
    
    for i, result in enumerate(results[:3]):
        print(f"\nExample {i+1}:")
        print(f"  Question: {result.question[:100]}...")
        print(f"  Should Abstain: {result.should_abstain}")
        print(f"  Semantic Entropy: {result.semantic_entropy:.3f}")
        print(f"  Clusters: {result.num_clusters}")
        print(f"  Abstained Samples: {result.abstained_samples}/{num_samples}")
        print(f"  Sample Responses:")
        for j, sample in enumerate(result.samples[:2]):
            print(f"    {j+1}: {sample[:100]}...")


# ----------------------------
# CLI Interface
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate semantic entropy for AbstentionBench"
    )
    
    parser.add_argument(
        "--dataset",
        choices=["gsm8k", "squad2", "falseqa", "self_aware", "world_sense"],
        default="gsm8k",
        help="AbstentionBench dataset to evaluate"
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=100,
        help="Number of examples to evaluate"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Language model to use"
    )
    parser.add_argument(
        "--nli_model",
        type=str,
        default="microsoft/deberta-v3-base-mnli",
        help="NLI model for semantic clustering"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples per question"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--use_system_prompt",
        action="store_true",
        default=True,
        help="Use AbstentionBench system prompt"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set dtype based on device
    # Use float16 for LM on GPU to save memory, but NLI will use float32
    torch_dtype = torch.float16 if args.device.startswith("cuda") else None
    
    print(f"Loading models...")
    print(f"  LM: {args.model}")
    print(f"  NLI: {args.nli_model}")
    print(f"  Device: {args.device}")
    
    # Initialize models
    lm = CausalLMWithSystemPrompt(
        model_name=args.model,
        device=args.device,
        torch_dtype=torch_dtype,
        use_system_prompt=args.use_system_prompt,
    )
    
    nli = NLIModel(
        model_name=args.nli_model,
        device=args.device,
        torch_dtype=torch_dtype,
    )
    
    # Run evaluation
    evaluate_with_semantic_entropy(
        dataset_name=args.dataset,
        lm=lm,
        nli=nli,
        subset_size=args.subset_size,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )


if __name__ == "__main__":
    main() 