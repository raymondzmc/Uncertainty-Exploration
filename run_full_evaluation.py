#!/usr/bin/env python
"""
Full Evaluation Script for Semantic Entropy + AbstentionBench
==============================================================

This script runs comprehensive evaluations of semantic entropy
as a predictor for abstention decisions across multiple datasets.

Usage:
    python run_full_evaluation.py --config default
    python run_full_evaluation.py --config fast
    python run_full_evaluation.py --config comprehensive
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Configuration presets
CONFIGS = {
    "fast": {
        "description": "Quick test run (20 examples per dataset)",
        "datasets": ["squad2"],  # SQuAD2 has better mix of answerable/unanswerable
        "subset_size": 20,
        "num_samples": 5,
        "models": {
            "lm": "gpt2",  # Small model for testing
            "nli": "cross-encoder/nli-MiniLM2-L6-H768"
        },
        "temperature": 0.7,
        "max_new_tokens": 32
    },
    
    "default": {
        "description": "Standard evaluation (50 examples per dataset)",
        "datasets": ["gsm8k", "squad2", "falseqa"],
        "subset_size": 50,
        "num_samples": 10,
        "models": {
            "lm": "meta-llama/Llama-3.2-1B-Instruct",
            "nli": "microsoft/deberta-v3-base-mnli"
        },
        "temperature": 0.7,
        "max_new_tokens": 64
    },
    
    "comprehensive": {
        "description": "Full evaluation (100 examples, all datasets)",
        "datasets": ["gsm8k", "squad2", "falseqa", "self_aware", "world_sense"],
        "subset_size": 100,
        "num_samples": 15,
        "models": {
            "lm": "meta-llama/Llama-3.2-1B-Instruct",
            "nli": "microsoft/deberta-large-mnli"
        },
        "temperature": 0.7,
        "max_new_tokens": 64
    },
    
    "gpu_optimized": {
        "description": "GPU-optimized with larger models",
        "datasets": ["gsm8k", "squad2", "falseqa", "self_aware", "world_sense"],
        "subset_size": 200,
        "num_samples": 20,
        "models": {
            "lm": "meta-llama/Llama-3.2-3B-Instruct",
            "nli": "microsoft/deberta-xlarge-mnli"
        },
        "temperature": 0.8,
        "max_new_tokens": 128
    }
}


class EvaluationRunner:
    """Manages the execution of semantic entropy evaluations."""
    
    def __init__(self, config_name: str, output_dir: Optional[str] = None):
        self.config = CONFIGS.get(config_name, CONFIGS["default"])
        self.config_name = config_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(f"results/semantic_entropy_{self.timestamp}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def check_environment(self) -> bool:
        """Check if the environment is properly set up."""
        print("🔍 Checking environment...")
        
        # Check PyTorch
        try:
            import torch
            print(f"  ✓ PyTorch {torch.__version__}")
            
            # Check device availability
            if torch.cuda.is_available():
                print(f"  ✓ CUDA available (GPU: {torch.cuda.get_device_name(0)})")
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print(f"  ✓ MPS available (Apple Silicon)")
                self.device = "mps"
            else:
                print(f"  ⚠️  No GPU detected, using CPU (will be slower)")
                self.device = "cpu"
        except ImportError:
            print("  ✗ PyTorch not installed")
            return False
        
        # Check Transformers
        try:
            import transformers
            print(f"  ✓ Transformers {transformers.__version__}")
        except ImportError:
            print("  ✗ Transformers not installed")
            return False
        
        # Check sklearn
        try:
            import sklearn
            print(f"  ✓ Scikit-learn {sklearn.__version__}")
        except ImportError:
            print("  ✗ Scikit-learn not installed")
            return False
        
        # Check AbstentionBench
        abstention_path = Path(__file__).parent / "AbstentionBench"
        if abstention_path.exists():
            print(f"  ✓ AbstentionBench found")
        else:
            print(f"  ✗ AbstentionBench not found at {abstention_path}")
            return False
        
        return True
    
    def run_single_evaluation(self, dataset: str) -> Dict:
        """Run evaluation on a single dataset."""
        print(f"\n📊 Evaluating on {dataset}...")
        
        # Prepare command
        cmd = [
            sys.executable,
            "abstention_semantic_entropy.py",
            "--dataset", dataset,
            "--subset_size", str(self.config["subset_size"]),
            "--model", self.config["models"]["lm"],
            "--nli_model", self.config["models"]["nli"],
            "--num_samples", str(self.config["num_samples"]),
            "--temperature", str(self.config["temperature"]),
            "--max_new_tokens", str(self.config["max_new_tokens"]),
            "--device", self.device,
            "--use_system_prompt",
            "--seed", "42"
        ]
        
        # Run evaluation
        print(f"  Command: {' '.join(cmd[:5])}...")
        
        output_file = self.output_dir / f"{dataset}_output.txt"
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            # Save output
            with open(output_file, 'w') as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n\nSTDERR:\n")
                    f.write(result.stderr)
            
            if result.returncode == 0:
                print(f"  ✓ Completed successfully")
                # Parse results from output
                return self.parse_results(result.stdout)
            else:
                print(f"  ✗ Failed with error code {result.returncode}")
                print(f"  See {output_file} for details")
                return {"status": "failed", "error": result.stderr}
                
        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout exceeded")
            return {"status": "timeout"}
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return {"status": "error", "error": str(e)}
    
    def parse_results(self, output: str) -> Dict:
        """Parse evaluation results from output."""
        results = {
            "status": "completed",
            "metrics": {}
        }
        
        # Extract AUROC scores
        if "AUROC Scores" in output:
            for line in output.split('\n'):
                if "Semantic Entropy:" in line and "AUROC" in line:
                    try:
                        # Handle both numeric values and "N/A"
                        value_str = line.split(':')[-1].strip()
                        if value_str == "N/A":
                            results["metrics"]["semantic_entropy_auroc"] = None
                        else:
                            results["metrics"]["semantic_entropy_auroc"] = float(value_str)
                    except:
                        pass
                elif "Predictive Entropy:" in line and "AUROC" in line:
                    try:
                        # Handle both numeric values and "N/A"
                        value_str = line.split(':')[-1].strip()
                        if value_str == "N/A":
                            results["metrics"]["predictive_entropy_auroc"] = None
                        else:
                            results["metrics"]["predictive_entropy_auroc"] = float(value_str)
                    except:
                        pass
                elif "Majority Vote Abstention Accuracy:" in line:
                    try:
                        value = float(line.split(':')[-1].strip())
                        results["metrics"]["majority_vote_accuracy"] = value
                    except:
                        pass
        
        # Extract dataset statistics
        if "Total examples:" in output:
            for line in output.split('\n'):
                if "Total examples:" in line:
                    try:
                        results["total_examples"] = int(line.split(':')[-1].strip())
                    except:
                        pass
                elif "Should abstain:" in line and "Summary" not in line:
                    try:
                        results["should_abstain"] = int(line.split(':')[-1].strip())
                    except:
                        pass
                elif "Should answer:" in line:
                    try:
                        results["should_answer"] = int(line.split(':')[-1].strip())
                    except:
                        pass
        
        return results
    
    def run_all_evaluations(self):
        """Run evaluations on all configured datasets."""
        print("\n" + "="*70)
        print(f"RUNNING FULL EVALUATION: {self.config_name}")
        print("="*70)
        print(f"Configuration: {self.config['description']}")
        print(f"Output directory: {self.output_dir}")
        
        # Save configuration
        config_file = self.output_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump({
                "config_name": self.config_name,
                "config": self.config,
                "timestamp": self.timestamp,
                "device": self.device
            }, f, indent=2)
        
        # Run evaluations
        for dataset in self.config["datasets"]:
            self.results[dataset] = self.run_single_evaluation(dataset)
        
        # Save results
        results_file = self.output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate a summary report of all evaluations."""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        summary = []
        summary.append(f"Configuration: {self.config_name}")
        summary.append(f"Timestamp: {self.timestamp}")
        summary.append(f"Device: {self.device}")
        summary.append("")
        
        # Results table
        summary.append("Results by Dataset:")
        summary.append("-" * 50)
        
        for dataset, result in self.results.items():
            summary.append(f"\n{dataset.upper()}:")
            
            if result.get("status") == "completed":
                metrics = result.get("metrics", {})
                summary.append(f"  Status: ✓ Completed")
                
                if "semantic_entropy_auroc" in metrics:
                    value = metrics['semantic_entropy_auroc']
                    if value is None:
                        summary.append(f"  Semantic Entropy AUROC: N/A (insufficient class diversity)")
                    else:
                        summary.append(f"  Semantic Entropy AUROC: {value:.3f}")
                if "predictive_entropy_auroc" in metrics:
                    value = metrics['predictive_entropy_auroc']
                    if value is None:
                        summary.append(f"  Predictive Entropy AUROC: N/A (insufficient class diversity)")
                    else:
                        summary.append(f"  Predictive Entropy AUROC: {value:.3f}")
                if "majority_vote_accuracy" in metrics:
                    summary.append(f"  Majority Vote Accuracy: {metrics['majority_vote_accuracy']:.3f}")
            else:
                summary.append(f"  Status: ✗ {result.get('status', 'unknown')}")
                if "error" in result:
                    summary.append(f"  Error: {result['error'][:100]}...")
        
        summary.append("")
        summary.append("="*70)
        
        # Print summary
        summary_text = "\n".join(summary)
        print(summary_text)
        
        # Save summary
        summary_file = self.output_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary_text)
        
        print(f"\n📁 Results saved to: {self.output_dir}")
        print(f"   - config.json: Configuration details")
        print(f"   - results.json: Parsed results")
        print(f"   - summary.txt: This summary")
        print(f"   - [dataset]_output.txt: Raw outputs")


def main():
    parser = argparse.ArgumentParser(
        description="Run semantic entropy evaluation for AbstentionBench"
    )
    
    parser.add_argument(
        "--config",
        choices=list(CONFIGS.keys()),
        default="default",
        help="Configuration preset to use"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: auto-generated)"
    )
    
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configurations and exit"
    )
    
    args = parser.parse_args()
    
    # List configurations if requested
    if args.list_configs:
        print("\nAvailable Configurations:")
        print("="*50)
        for name, config in CONFIGS.items():
            print(f"\n{name}:")
            print(f"  {config['description']}")
            print(f"  Datasets: {', '.join(config['datasets'])}")
            print(f"  Examples: {config['subset_size']} per dataset")
            print(f"  Samples: {config['num_samples']} per question")
        return
    
    # Run evaluation
    runner = EvaluationRunner(args.config, args.output_dir)
    
    # Check environment
    if not runner.check_environment():
        print("\n❌ Environment check failed. Please install required packages:")
        print("   pip install -r requirements_cpu_macos.txt  # For CPU")
        print("   pip install -r requirements_gpu_ubuntu.txt # For GPU")
        sys.exit(1)
    
    # Run evaluations
    try:
        runner.run_all_evaluations()
        print("\n✅ Evaluation completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 