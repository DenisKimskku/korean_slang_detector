#!/bin/bash

# Multi-Model Drug Detection Evaluation Script
# This script runs evaluation on multiple KLUE models and saves results in separate directories

set -e  # Exit on any error

# Configuration
PYTHON_SCRIPT="ablation_plain.py"
BATCH_SIZE=16
MAX_LENGTH=512

# Models to evaluate
MODELS=(
    # "klue/roberta-small"
    # "klue/roberta-base" 
    # "klue/roberta-large"
    "klue/bert-base"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
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

# Function to check if required files exist
check_requirements() {
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        print_error "Python script '$PYTHON_SCRIPT' not found!"
        exit 1
    fi
    
    if ! command -v python &> /dev/null; then
        print_error "Python not found! Please install Python."
        exit 1
    fi
    
    print_success "Requirements check passed"
}

# Function to check GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        if [ "$GPU_INFO" -gt 1000 ]; then
            print_success "GPU available with ${GPU_INFO}MB free memory"
            return 0
        else
            print_warning "GPU has low memory (${GPU_INFO}MB). Consider reducing batch size."
            return 1
        fi
    else
        print_warning "nvidia-smi not found. Running on CPU (will be slower)."
        return 1
    fi
}

# Function to estimate time and memory requirements
estimate_requirements() {
    local model=$1
    local estimated_time="Unknown"
    local estimated_memory="Unknown"
    
    case $model in
        "klue/roberta-small")
            estimated_time="~10-15 minutes"
            estimated_memory="~2-4 GB"
            ;;
        "klue/roberta-base")
            estimated_time="~15-25 minutes" 
            estimated_memory="~4-6 GB"
            ;;
        "klue/roberta-large")
            estimated_time="~25-40 minutes"
            estimated_memory="~8-12 GB"
            ;;
        "klue/bert-base")
            estimated_time="~15-25 minutes"
            estimated_memory="~4-6 GB"
            ;;
    esac
    
    echo "Estimated time: $estimated_time, Memory: $estimated_memory"
}

# Function to run evaluation for a single model
run_model_evaluation() {
    local model=$1
    local model_name=$(echo $model | sed 's/klue\///g' | sed 's/-/_/g')
    
    print_status "Starting evaluation for model: $model"
    estimate_requirements $model
    
    # Create timestamp for this run
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local log_file="evaluation_${model_name}_${timestamp}.log"
    
    # Run the evaluation
    if python "$PYTHON_SCRIPT" \
        --model "$model" \
        --batch_size "$BATCH_SIZE" \
        --max_length "$MAX_LENGTH" \
        2>&1 | tee "$log_file"; then
        
        print_success "Completed evaluation for $model"
        
        # Move log file to appropriate directory
        if [ -d "logs_${model_name}" ]; then
            mv "$log_file" "logs_${model_name}/"
        fi
        
        return 0
    else
        print_error "Failed evaluation for $model"
        return 1
    fi
}

# Function to create summary report
create_summary_report() {
    local summary_file="evaluation_summary_$(date '+%Y%m%d_%H%M%S').txt"
    
    print_status "Creating summary report: $summary_file"
    
    {
        echo "Multi-Model Drug Detection Evaluation Summary"
        echo "============================================="
        echo "Evaluation Date: $(date)"
        echo "Batch Size: $BATCH_SIZE"
        echo "Max Length: $MAX_LENGTH"
        echo ""
        
        for model in "${MODELS[@]}"; do
            local model_name=$(echo $model | sed 's/klue\///g' | sed 's/-/_/g')
            local results_dir="results_${model_name}"
            
            echo "Model: $model"
            echo "Results Directory: $results_dir"
            
            if [ -d "$results_dir" ]; then
                local latest_result=$(ls -t "$results_dir"/*.json 2>/dev/null | head -1)
                if [ -n "$latest_result" ]; then
                    echo "Latest Results File: $(basename $latest_result)"
                    # Extract key metrics if jq is available
                    if command -v jq &> /dev/null; then
                        echo "Metrics:"
                        jq -r '.test_metrics | "  Accuracy: \(.accuracy | tonumber | . * 100 | round / 100)%, F1: \(.f1 | tonumber | . * 100 | round / 100)%, AUC: \(.auc | tonumber | . * 100 | round / 100)%"' "$latest_result" 2>/dev/null || echo "  Unable to parse metrics"
                    fi
                else
                    echo "No results found"
                fi
            else
                echo "Results directory not found"
            fi
            echo ""
        done
        
        echo "Directory Structure:"
        for model in "${MODELS[@]}"; do
            local model_name=$(echo $model | sed 's/klue\///g' | sed 's/-/_/g')
            echo "  logs_${model_name}/       - Log files"
            echo "  results_${model_name}/    - JSON result files"  
            echo "  predictions_${model_name}/ - Detailed predictions"
        done
        
    } > "$summary_file"
    
    print_success "Summary report created: $summary_file"
}

# Function to cleanup temporary files
cleanup() {
    print_status "Cleaning up temporary files..."
    # Remove any temporary log files in current directory
    rm -f evaluation_*.log 2>/dev/null || true
    print_success "Cleanup completed"
}

# Main execution
main() {
    echo "======================================================"
    echo "Multi-Model Drug Detection Evaluation"
    echo "======================================================"
    echo "Models to evaluate: ${MODELS[*]}"
    echo "Batch size: $BATCH_SIZE"
    echo "Max length: $MAX_LENGTH"
    echo "======================================================"
    
    # Check requirements
    check_requirements
    
    # Check GPU
    check_gpu
    
    # Initialize counters
    local total_models=${#MODELS[@]}
    local successful_runs=0
    local failed_runs=0
    local start_time=$(date +%s)
    
    print_status "Starting evaluation of $total_models models..."
    
    # Run evaluation for each model
    for i in "${!MODELS[@]}"; do
        local model="${MODELS[$i]}"
        local current=$((i + 1))
        
        echo ""
        print_status "Progress: [$current/$total_models] Evaluating $model"
        echo "======================================================"
        
        if run_model_evaluation "$model"; then
            ((successful_runs++))
        else
            ((failed_runs++))
            print_error "Evaluation failed for $model"
        fi
        
        # Show progress
        local elapsed=$(($(date +%s) - start_time))
        local avg_time_per_model=$((elapsed / current))
        local remaining_models=$((total_models - current))
        local estimated_remaining_time=$((avg_time_per_model * remaining_models))
        
        print_status "Completed $current/$total_models models"
        print_status "Time elapsed: ${elapsed}s, Estimated remaining: ${estimated_remaining_time}s"
        
        # Brief pause between models
        sleep 2
    done
    
    # Calculate total time
    local total_time=$(($(date +%s) - start_time))
    local hours=$((total_time / 3600))
    local minutes=$(((total_time % 3600) / 60))
    local seconds=$((total_time % 60))
    
    echo ""
    echo "======================================================"
    print_status "Evaluation Summary"
    echo "======================================================"
    echo "Total models evaluated: $total_models"
    echo "Successful runs: $successful_runs"
    echo "Failed runs: $failed_runs"
    printf "Total time: %02d:%02d:%02d\n" $hours $minutes $seconds
    echo "======================================================"
    
    # Create summary report
    create_summary_report
    
    # Show directory structure
    echo ""
    print_status "Generated directories:"
    for model in "${MODELS[@]}"; do
        local model_name=$(echo $model | sed 's/klue\///g' | sed 's/-/_/g')
        echo "  📁 logs_${model_name}/"
        echo "  📁 results_${model_name}/"
        echo "  📁 predictions_${model_name}/"
    done
    
    # Cleanup
    cleanup
    
    if [ $failed_runs -eq 0 ]; then
        print_success "All evaluations completed successfully!"
        exit 0
    else
        print_warning "Some evaluations failed. Check logs for details."
        exit 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -b, --batch-size    Set batch size (default: 16)"
    echo "  -l, --max-length    Set max sequence length (default: 512)"
    echo "  -m, --model         Evaluate single model only"
    echo "  --dry-run          Show what would be executed without running"
    echo "  --continue         Continue from failed models only"
    echo ""
    echo "Available models:"
    for model in "${MODELS[@]}"; do
        echo "  - $model"
    done
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run all models"
    echo "  $0 -b 8 -l 256                      # Custom batch size and length"
    echo "  $0 -m klue/roberta-base              # Single model"
    echo "  $0 --dry-run                         # Preview execution"
}

# Function for dry run
dry_run() {
    echo "DRY RUN - Commands that would be executed:"
    echo "=========================================="
    
    for model in "${MODELS[@]}"; do
        echo "python $PYTHON_SCRIPT --model $model --batch_size $BATCH_SIZE --max_length $MAX_LENGTH"
    done
    
    echo ""
    echo "Directories that would be created:"
    for model in "${MODELS[@]}"; do
        local model_name=$(echo $model | sed 's/klue\///g' | sed 's/-/_/g')
        echo "  logs_${model_name}/"
        echo "  results_${model_name}/"
        echo "  predictions_${model_name}/"
    done
}

# Function to check for existing results and continue
continue_evaluation() {
    local remaining_models=()
    
    print_status "Checking for existing results..."
    
    for model in "${MODELS[@]}"; do
        local model_name=$(echo $model | sed 's/klue\///g' | sed 's/-/_/g')
        local results_dir="results_${model_name}"
        
        if [ -d "$results_dir" ] && [ "$(ls -A $results_dir 2>/dev/null)" ]; then
            print_success "Found existing results for $model, skipping..."
        else
            remaining_models+=("$model")
        fi
    done
    
    if [ ${#remaining_models[@]} -eq 0 ]; then
        print_success "All models have existing results!"
        create_summary_report
        exit 0
    fi
    
    print_status "Will evaluate ${#remaining_models[@]} remaining models: ${remaining_models[*]}"
    MODELS=("${remaining_models[@]}")
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -l|--max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        -m|--model)
            # Validate model
            if [[ " ${MODELS[@]} " =~ " $2 " ]]; then
                MODELS=("$2")
            else
                print_error "Invalid model: $2"
                show_usage
                exit 1
            fi
            shift 2
            ;;
        --dry-run)
            dry_run
            exit 0
            ;;
        --continue)
            continue_evaluation
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate batch size and max length
if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$BATCH_SIZE" -lt 1 ]; then
    print_error "Invalid batch size: $BATCH_SIZE"
    exit 1
fi

if ! [[ "$MAX_LENGTH" =~ ^[0-9]+$ ]] || [ "$MAX_LENGTH" -lt 1 ]; then
    print_error "Invalid max length: $MAX_LENGTH"
    exit 1
fi

# Trap to handle interruption
trap 'echo ""; print_warning "Evaluation interrupted by user"; cleanup; exit 130' INT TERM

# Run main function
main