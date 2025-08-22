#!/bin/bash

# ============================================================
# Semantic Entropy + AbstentionBench Evaluation Runner
# ============================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Default configuration
CONFIG="default"
OUTPUT_DIR=""
CONDA_ENV="uncertainty"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config <name>       Configuration preset (fast/default/comprehensive/gpu_optimized)"
            echo "  --output-dir <path>   Output directory for results"
            echo "  --conda-env <name>    Conda environment name (default: uncertainty)"
            echo "  --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --config fast                    # Quick test run"
            echo "  $0 --config default                  # Standard evaluation"
            echo "  $0 --config comprehensive            # Full evaluation"
            echo "  $0 --config gpu_optimized            # GPU-optimized evaluation"
            echo ""
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Header
echo ""
echo "========================================================================"
echo "       Semantic Entropy + AbstentionBench Evaluation Runner"
echo "========================================================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed or not in PATH"
    exit 1
fi

# Activate conda environment
print_info "Activating conda environment: $CONDA_ENV"
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV
if [ $? -ne 0 ]; then
    print_error "Failed to activate conda environment: $CONDA_ENV"
    print_info "Create it with: conda create -n $CONDA_ENV python=3.12"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
print_info "Python version: $PYTHON_VERSION"

# Check required packages
print_info "Checking required packages..."

python -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    print_warning "PyTorch not installed. Install with:"
    echo "  pip install torch"
    exit 1
fi

python -c "import transformers" 2>/dev/null
if [ $? -ne 0 ]; then
    print_warning "Transformers not installed. Install with:"
    echo "  pip install transformers"
    exit 1
fi

python -c "import sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    print_warning "Scikit-learn not installed. Install with:"
    echo "  pip install scikit-learn"
    exit 1
fi

print_success "All required packages found"

# Check GPU availability
print_info "Checking compute device..."
DEVICE=$(python -c "
import torch
if torch.cuda.is_available():
    print('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('mps')
else:
    print('cpu')
" 2>/dev/null)

case $DEVICE in
    cuda)
        GPU_INFO=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        print_success "CUDA GPU detected: $GPU_INFO"
        ;;
    mps)
        print_success "Apple Silicon GPU (MPS) detected"
        ;;
    cpu)
        print_warning "No GPU detected, using CPU (evaluation will be slower)"
        if [ "$CONFIG" = "gpu_optimized" ]; then
            print_error "GPU-optimized config selected but no GPU available"
            print_info "Switching to 'default' configuration"
            CONFIG="default"
        fi
        ;;
esac

# Show configuration
echo ""
print_info "Configuration: $CONFIG"

# Build command
CMD="python run_full_evaluation.py --config $CONFIG"

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
    print_info "Output directory: $OUTPUT_DIR"
else
    print_info "Output directory: auto-generated"
fi

# Show what will be run
echo ""
echo "------------------------------------------------------------------------"
python run_full_evaluation.py --list-configs | grep -A 4 "^$CONFIG:"
echo "------------------------------------------------------------------------"
echo ""

# Confirmation prompt
read -p "Do you want to proceed with the evaluation? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Evaluation cancelled by user"
    exit 0
fi

# Run evaluation
echo ""
print_info "Starting evaluation..."
echo "Command: $CMD"
echo ""

# Execute the evaluation
$CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    print_success "Evaluation completed successfully!"
    echo ""
    
    # Show results location
    if [ -z "$OUTPUT_DIR" ]; then
        # Find the most recent results directory
        LATEST_RESULTS=$(ls -td results/semantic_entropy_* 2>/dev/null | head -1)
        if [ -n "$LATEST_RESULTS" ]; then
            print_info "Results saved to: $LATEST_RESULTS"
            echo ""
            echo "View summary with:"
            echo "  cat $LATEST_RESULTS/summary.txt"
            echo ""
            echo "View detailed results with:"
            echo "  cat $LATEST_RESULTS/results.json"
        fi
    fi
else
    echo ""
    print_error "Evaluation failed!"
    echo "Check the error messages above for details"
    exit 1
fi

echo ""
echo "========================================================================"
print_success "Done!"
echo "========================================================================" 