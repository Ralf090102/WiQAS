#!/bin/bash

# WiQAS GCP Setup Script - Continue from existing setup
# Assumes: CUDA 12.8, Python 3.12, ~/wiqas-venv, and Ollama already installed
# This script clones the repo and installs WiQAS dependencies

set -e  # Exit on error

echo "=========================================="
echo "   WiQAS Setup (Post-Infrastructure)"
echo "=========================================="
echo ""

# Configuration
VENV_PATH="${VENV_PATH:-$HOME/wiqas-venv}"
REPO_URL="${REPO_URL:-https://github.com/Ralf090102/WiQAS.git}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/WiQAS}"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Verify prerequisites
verify_prerequisites() {
    print_info "Verifying existing setup..."
    local failed=0
    
    # Check CUDA
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep release | sed -n 's/.*release \([0-9.]*\).*/\1/p')
        print_success "CUDA installed: Version $CUDA_VERSION"
    else
        print_error "CUDA not found. Please install CUDA first."
        failed=1
    fi
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        print_success "GPU detected: $GPU_NAME"
    else
        print_error "nvidia-smi not found. GPU drivers not installed."
        failed=1
    fi
    
    # Check Python 3.12
    if command -v python3.12 &> /dev/null; then
        print_success "Python 3.12 installed"
    else
        print_error "Python 3.12 not found"
        failed=1
    fi
    
    # Check virtual environment
    if [ -d "$VENV_PATH" ]; then
        print_success "Virtual environment found: $VENV_PATH"
    else
        print_warning "Virtual environment not found at $VENV_PATH"
        print_info "Creating virtual environment..."
        python3.12 -m venv "$VENV_PATH"
        print_success "Virtual environment created"
    fi
    
    # Check Ollama
    if command -v ollama &> /dev/null; then
        print_success "Ollama installed"
        
        # Check if Ollama is running
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            print_success "Ollama service is running"
        else
            print_warning "Ollama not running, starting it..."
            ollama serve &> /tmp/ollama.log &
            sleep 3
        fi
    else
        print_error "Ollama not found"
        failed=1
    fi
    
    if [ $failed -eq 1 ]; then
        print_error "Prerequisites check failed. Please complete setup first."
        exit 1
    fi
    
    print_success "All prerequisites verified!"
}

# Determine VM role
detect_vm_role() {
    print_info "Detecting VM role..."
    
    # Check hostname or instance name
    if command -v curl &> /dev/null; then
        INSTANCE_NAME=$(curl -s -H "Metadata-Flavor: Google" \
            http://metadata.google.internal/computeMetadata/v1/instance/name 2>/dev/null || hostname)
    else
        INSTANCE_NAME=$(hostname)
    fi
    
    if [[ "$INSTANCE_NAME" == *"embedding"* ]] || [[ "$INSTANCE_NAME" == *"reranking"* ]]; then
        VM_ROLE="embedding-reranking"
        print_success "VM Role: Embedding + Reranking (T4)"
    elif [[ "$INSTANCE_NAME" == *"generation"* ]] || [[ "$INSTANCE_NAME" == *"evaluation"* ]]; then
        VM_ROLE="generation-evaluation"
        print_success "VM Role: Generation + Evaluation (A100)"
    else
        print_warning "Could not auto-detect VM role from instance name"
        echo ""
        echo "Select VM role:"
        echo "  1) Embedding + Reranking (T4)"
        echo "  2) Generation + Evaluation (A100)"
        read -p "Enter choice [1-2]: " choice
        
        case $choice in
            1) VM_ROLE="embedding-reranking" ;;
            2) VM_ROLE="generation-evaluation" ;;
            *) print_error "Invalid choice"; exit 1 ;;
        esac
    fi
    
    print_info "Configuring for: $VM_ROLE"
}

# Clone WiQAS repository
clone_repository() {
    print_info "Cloning WiQAS repository..."
    
    if [ -d "$INSTALL_DIR" ]; then
        print_warning "Directory $INSTALL_DIR already exists"
        read -p "Delete and re-clone? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$INSTALL_DIR"
        else
            print_info "Using existing directory"
            cd "$INSTALL_DIR"
            return 0
        fi
    fi
    
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    print_success "Repository cloned to $INSTALL_DIR"
}

# Install PyTorch with CUDA 12.8
install_pytorch() {
    print_info "Installing PyTorch with CUDA 12.8 support..."
    
    source "$VENV_PATH/bin/activate"
    
    # Install PyTorch for CUDA 12.x
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Verify installation
    print_info "Verifying PyTorch CUDA installation..."
    python -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available: {torch.cuda.is_available()}"); print(f"CUDA version: {torch.version.cuda}"); print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}")'
    
    print_success "PyTorch installed with CUDA support"
}

# Install WiQAS dependencies
install_wiqas_deps() {
    print_info "Installing WiQAS dependencies..."
    
    source "$VENV_PATH/bin/activate"
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dependencies installed from requirements.txt"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
    
    # Verify key packages
    print_info "Verifying key dependencies..."
    python -c "import sentence_transformers, chromadb, langchain; print('✅ Core packages OK')" || print_warning "Some packages may need attention"
}

# Create data directories
create_directories() {
    print_info "Creating data directories..."
    
    mkdir -p data/knowledge_base
    mkdir -p data/chroma-data
    mkdir -p logs
    
    print_success "Directories created"
}

# Configure WiQAS based on VM role
configure_wiqas() {
    print_info "Configuring WiQAS for $VM_ROLE..."
    
    # Determine LLM model based on VM role
    if [ "$VM_ROLE" == "generation-evaluation" ]; then
        LLM_MODEL="aisingapore/Gemma-SEA-LION-v3-9B-IT"
        BACKEND="ollama"
        print_info "Using model: $LLM_MODEL (A100 GPU)"
    else
        LLM_MODEL="TeeZee/gemma-2-9b-it-abliterated"
        BACKEND="ollama"
        print_info "Using model: $LLM_MODEL (T4 GPU)"
    fi
    
    # Create .env file
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# WiQAS Configuration for $VM_ROLE
# Generated: $(date)

# LLM Configuration
WIQAS_LLM_MODEL=$LLM_MODEL
WIQAS_LLM_BACKEND=$BACKEND
WIQAS_LLM_TEMPERATURE=0.7
WIQAS_LLM_MAX_TOKENS=512

# Chunking Configuration
WIQAS_CHUNK_SIZE=512
WIQAS_CHUNK_OVERLAP=50
WIQAS_CHUNKING_STRATEGY=recursive

# Vector Store Configuration
WIQAS_VECTORSTORE_PERSIST_DIRECTORY=./data/chroma-data
WIQAS_EMBEDDING_MODEL=BAAI/bge-m3

# Retrieval Configuration
WIQAS_RETRIEVAL_DEFAULT_K=5
WIQAS_RETRIEVAL_TYPE=hybrid

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
EOF
        print_success "Configuration file created: .env"
    else
        print_warning ".env file already exists"
        read -p "Overwrite with new configuration? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            mv .env .env.backup
            print_info "Backed up existing .env to .env.backup"
            configure_wiqas  # Recursive call to create new config
        fi
    fi
}

# Check Ollama models
check_ollama_models() {
    print_info "Checking Ollama models..."
    
    if ollama list | grep -q "gemma"; then
        print_success "Ollama models found:"
        ollama list | grep "gemma"
    else
        print_warning "No Gemma models found in Ollama"
        print_info "Expected models:"
        echo "  - TeeZee/gemma-2-9b-it-abliterated"
        echo "  - aisingapore/Gemma-SEA-LION-v3-9B-IT"
    fi
}

# Create helper scripts
create_helper_scripts() {
    print_info "Creating helper scripts..."
    
    # Activation script
    cat > activate.sh << 'EOF'
#!/bin/bash
# Quick activation script for WiQAS
source ~/wiqas-venv/bin/activate
cd ~/WiQAS
echo "✅ WiQAS environment activated"
echo "💡 Run: python run.py --help"
EOF
    chmod +x activate.sh
    
    # GPU monitor script
    cat > monitor_gpu.sh << 'EOF'
#!/bin/bash
# Monitor GPU usage
watch -n 1 nvidia-smi
EOF
    chmod +x monitor_gpu.sh
    
    print_success "Helper scripts created: activate.sh, monitor_gpu.sh"
}

# Main setup function
main() {
    echo ""
    print_info "Starting WiQAS setup (continuing from existing infrastructure)..."
    echo ""
    
    # Verify prerequisites
    verify_prerequisites
    
    # Detect VM role
    detect_vm_role
    
    # Clone repository
    clone_repository
    
    # Run setup steps
    install_pytorch
    install_wiqas_deps
    create_directories
    configure_wiqas
    check_ollama_models
    create_helper_scripts
    
    echo ""
    print_success "=========================================="
    print_success "   WiQAS Setup Complete!"
    print_success "=========================================="
    echo ""
    print_success "VM Role: $VM_ROLE"
    print_success "Install Directory: $INSTALL_DIR"
    print_success "Virtual Environment: $VENV_PATH"
    echo ""
    print_info "Next steps:"
    echo ""
    echo "  1. Transfer knowledge base (5GB):"
    echo "     - Use scripts/transfer_kb.sh from your local machine"
    echo "     - Or manually upload to: $INSTALL_DIR/data/knowledge_base/"
    echo ""
    echo "  2. Activate environment:"
    echo "     source $VENV_PATH/bin/activate"
    echo "     cd $INSTALL_DIR"
    echo "     # Or use: ./activate.sh"
    echo ""
    echo "  3. Ingest documents:"
    echo "     python run.py ingest ./data/knowledge_base/ --workers 8"
    echo ""
    echo "  4. Test the system:"
    echo "     python run.py status"
    echo "     python run.py ask 'What is bayanihan?'"
    echo ""
    print_info "Monitoring commands:"
    echo "  - GPU usage: nvidia-smi  (or ./monitor_gpu.sh)"
    echo "  - Ollama models: ollama list"
    echo "  - Ollama status: curl http://localhost:11434/api/tags"
    echo ""
}

# Run main function
main "$@"
