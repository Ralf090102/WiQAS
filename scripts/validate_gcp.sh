#!/bin/bash
# WiQAS GCP Environment Validation Script
# Checks if all requirements are properly configured

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_header() { echo -e "\n${BLUE}=== $1 ===${NC}"; }

WIQAS_DIR="/home/ralf_hernandez/WiQAS"
VENV_PATH="/home/ralf_hernandez/wiqas-venv"
ERRORS=0
WARNINGS=0

echo "=========================================="
echo "   WiQAS GCP Environment Validation"
echo "=========================================="
echo ""

# Check if running on correct VM
print_header "VM Information"
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *"wiqas-generation-evaluation"* ]]; then
    print_success "Running on correct VM: $HOSTNAME"
else
    print_warning "VM name doesn't match expected: $HOSTNAME"
    ((WARNINGS++))
fi

# Check location
if command -v curl &> /dev/null; then
    ZONE=$(curl -s -H "Metadata-Flavor: Google" \
        http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null | cut -d'/' -f4)
    if [ "$ZONE" == "asia-southeast1-c" ]; then
        print_success "Correct zone: $ZONE"
    else
        print_warning "Unexpected zone: $ZONE (expected: asia-southeast1-c)"
        ((WARNINGS++))
    fi
fi

# Check GPU
print_header "GPU Configuration"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)
    
    if [[ "$GPU_NAME" == *"A100"* ]]; then
        print_success "GPU detected: $GPU_NAME ($GPU_MEMORY)"
    else
        print_error "Unexpected GPU: $GPU_NAME (expected: A100)"
        ((ERRORS++))
    fi
    
    # Check CUDA
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep release | sed -n 's/.*release \([0-9.]*\).*/\1/p')
        print_success "CUDA version: $CUDA_VERSION"
    else
        print_error "CUDA not found"
        ((ERRORS++))
    fi
else
    print_error "nvidia-smi not found - GPU drivers not installed"
    ((ERRORS++))
fi

# Check directories
print_header "Directory Structure"
if [ -d "$WIQAS_DIR" ]; then
    print_success "WiQAS directory exists: $WIQAS_DIR"
else
    print_error "WiQAS directory not found: $WIQAS_DIR"
    ((ERRORS++))
    exit 1
fi

cd "$WIQAS_DIR" || exit 1

REQUIRED_DIRS=("data" "data/chroma-data" "data/knowledge_base" "logs" "temp" "backend" "frontend" "src")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        print_success "Directory exists: $dir"
    else
        print_warning "Directory missing: $dir (will be created)"
        mkdir -p "$dir"
        ((WARNINGS++))
    fi
done

# Check virtual environment
print_header "Python Environment"
if [ -d "$VENV_PATH" ]; then
    print_success "Virtual environment exists: $VENV_PATH"
    
    source "$VENV_PATH/bin/activate"
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    print_success "Python version: $PYTHON_VERSION"
    
    # Check key packages
    print_info "Checking Python packages..."
    PACKAGES=("torch" "fastapi" "uvicorn" "chromadb" "sentence_transformers" "langchain" "ollama")
    for pkg in "${PACKAGES[@]}"; do
        if python -c "import $pkg" 2>/dev/null; then
            VERSION=$(python -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "unknown")
            print_success "Package installed: $pkg ($VERSION)"
        else
            print_error "Package missing: $pkg"
            ((ERRORS++))
        fi
    done
    
    # Check PyTorch CUDA
    print_info "Checking PyTorch CUDA support..."
    TORCH_CUDA=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [ "$TORCH_CUDA" == "True" ]; then
        TORCH_GPU=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        print_success "PyTorch CUDA available - GPU: $TORCH_GPU"
    else
        print_error "PyTorch CUDA not available"
        ((ERRORS++))
    fi
else
    print_error "Virtual environment not found: $VENV_PATH"
    ((ERRORS++))
fi

# Check configuration files
print_header "Configuration Files"
if [ -f ".env" ]; then
    print_success ".env file exists"
    
    # Check key variables
    print_info "Checking environment variables..."
    source .env 2>/dev/null || true
    
    VARS=("WIQAS_LLM_BASE_URL" "WIQAS_VECTORSTORE_PERSIST_DIRECTORY" "CUDA_VISIBLE_DEVICES")
    for var in "${VARS[@]}"; do
        value="${!var}"
        if [ -n "$value" ]; then
            print_success "$var is set: $value"
        else
            print_warning "$var not set in .env"
            ((WARNINGS++))
        fi
    done
    
    # Check if paths are absolute
    if [[ "$WIQAS_VECTORSTORE_PERSIST_DIRECTORY" == /* ]]; then
        print_success "Using absolute path for vector store"
    else
        print_warning "Vector store path is relative (should be absolute for GCP)"
        ((WARNINGS++))
    fi
else
    print_warning ".env file not found (copy from .env.example)"
    ((WARNINGS++))
fi

if [ -f "frontend/.env" ]; then
    print_success "Frontend .env file exists"
else
    print_warning "Frontend .env not found (copy from frontend/.env.example)"
    ((WARNINGS++))
fi

# Check Ollama
print_header "Ollama Service"
if command -v ollama &> /dev/null; then
    print_success "Ollama installed"
    
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_success "Ollama service is running"
        
        print_info "Available models:"
        ollama list | grep -E "(NAME|gemma|mistral|llama)" || print_warning "No models found"
    else
        print_warning "Ollama not running (use: ollama serve &)"
        ((WARNINGS++))
    fi
else
    print_error "Ollama not installed"
    ((ERRORS++))
fi

# Check network configuration
print_header "Network Configuration"
EXPECTED_IP="34.142.151.130"
EXTERNAL_IP=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip \
    2>/dev/null || echo "Unable to fetch")

if [ "$EXTERNAL_IP" != "Unable to fetch" ]; then
    if [ "$EXTERNAL_IP" == "$EXPECTED_IP" ]; then
        print_success "External IP matches expected: $EXTERNAL_IP"
    else
        print_warning "External IP changed: $EXTERNAL_IP (expected: $EXPECTED_IP)"
        print_warning "Update .env CORS_ORIGINS and frontend/.env PUBLIC_BACKEND_URL"
        ((WARNINGS++))
    fi
    echo ""
    print_info "Access URLs:"
    echo "  Backend API: http://$EXTERNAL_IP:8000"
    echo "  API Docs:    http://$EXTERNAL_IP:8000/docs"
    echo "  Frontend:    http://$EXTERNAL_IP:3000"
else
    print_warning "Could not fetch external IP"
    ((WARNINGS++))
    echo ""
    print_info "Expected Access URLs:"
    echo "  Backend API: http://$EXPECTED_IP:8000"
    echo "  API Docs:    http://$EXPECTED_IP:8000/docs"
    echo "  Frontend:    http://$EXPECTED_IP:3000"
fi

# Check if ports are available
print_info "Checking port availability..."
for port in 8000 3000 11434; do
    if lsof -i :$port > /dev/null 2>&1; then
        print_warning "Port $port is in use"
        lsof -i :$port | tail -1
    else
        print_success "Port $port is available"
    fi
done

# Check scripts
print_header "Helper Scripts"
SCRIPTS=("start_gcp.sh" "stop_gcp.sh" "wiqas.service" "ollama.service")
for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ] || [[ "$script" == *.service ]]; then
            print_success "Script found: $script"
        else
            print_warning "Script found but not executable: $script (use: chmod +x $script)"
            ((WARNINGS++))
        fi
    else
        print_warning "Script not found: $script"
        ((WARNINGS++))
    fi
done

# Check data
print_header "Data Status"
if [ -d "data/knowledge_base" ]; then
    KB_FILES=$(find data/knowledge_base -type f 2>/dev/null | wc -l)
    if [ "$KB_FILES" -gt 0 ]; then
        print_success "Knowledge base has $KB_FILES files"
    else
        print_warning "Knowledge base is empty (add documents to data/knowledge_base/)"
        ((WARNINGS++))
    fi
fi

if [ -d "data/chroma-data" ]; then
    if [ -f "data/chroma-data/chroma.sqlite3" ]; then
        CHROMA_SIZE=$(du -sh data/chroma-data 2>/dev/null | awk '{print $1}')
        print_success "ChromaDB exists (size: $CHROMA_SIZE)"
    else
        print_warning "ChromaDB not initialized (run: python run.py ingest ./data/knowledge_base/)"
        ((WARNINGS++))
    fi
fi

# Summary
print_header "Validation Summary"
echo ""
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    print_success "=========================================="
    print_success "   All checks passed! ✨"
    print_success "=========================================="
    echo ""
    print_info "Your WiQAS environment is ready!"
    echo ""
    print_info "Next steps:"
    echo "  1. Start services: ./start_gcp.sh"
    echo "  2. Test: curl http://localhost:8000/health"
    echo "  3. Access: http://$EXTERNAL_IP:8000/docs"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    print_warning "=========================================="
    print_warning "   Validation completed with $WARNINGS warnings"
    print_warning "=========================================="
    echo ""
    print_info "Review warnings above. System may still work."
    exit 0
else
    print_error "=========================================="
    print_error "   Validation failed!"
    print_error "   Errors: $ERRORS | Warnings: $WARNINGS"
    print_error "=========================================="
    echo ""
    print_error "Please fix the errors above before running WiQAS"
    exit 1
fi
