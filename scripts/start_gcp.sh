#!/bin/bash
# WiQAS GCP Startup Script
# VM: wiqas-generation-evaluation-20260120-061404
# Location: asia-southeast1-c
# GPU: 1x NVIDIA A100 40GB

set -e

# Configuration
WIQAS_DIR="/home/ralf_hernandez/WiQAS"
VENV_PATH="/home/ralf_hernandez/wiqas-venv"
LOG_DIR="$WIQAS_DIR/logs"
BACKEND_PORT=8000
FRONTEND_PORT=3000

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

echo "=========================================="
echo "   WiQAS GCP Startup"
echo "   VM: wiqas-generation-evaluation"
echo "   GPU: NVIDIA A100 40GB"
echo "=========================================="
echo ""

# Change to WiQAS directory
cd "$WIQAS_DIR" || exit 1
print_success "Changed to $WIQAS_DIR"

# Activate virtual environment
source "$VENV_PATH/bin/activate" || { print_error "Failed to activate venv"; exit 1; }
print_success "Activated virtual environment"

# Create log directory
mkdir -p "$LOG_DIR"

# Check GPU
print_info "Checking GPU status..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1)
    print_success "GPU detected: $GPU_INFO"
else
    print_warning "nvidia-smi not found"
fi

# Check Ollama
print_info "Checking Ollama status..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    print_success "Ollama is running"
else
    print_warning "Ollama not responding, attempting to start..."
    nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
    sleep 3
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_success "Ollama started successfully"
    else
        print_error "Failed to start Ollama. Check $LOG_DIR/ollama.log"
        exit 1
    fi
fi

# List available models
print_info "Available Ollama models:"
ollama list | grep -E "(NAME|gemma|mistral|llama)" || print_warning "No models found"

# Check .env file
if [ ! -f ".env" ]; then
    print_error ".env file not found!"
    print_info "The .env file should be included in the repository."
    print_info "If missing, copy from .env.example: cp .env.example .env"
    exit 1
else
    print_success ".env file found (pre-configured for this VM)"
fi

# Get external IP
print_info "Getting VM external IP..."
EXTERNAL_IP="34.142.151.130"  # Pre-configured
VERIFY_IP=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip \
    2>/dev/null || echo "")

if [ -n "$VERIFY_IP" ]; then
    if [ "$VERIFY_IP" != "$EXTERNAL_IP" ]; then
        print_warning "External IP changed from $EXTERNAL_IP to $VERIFY_IP"
        print_warning "Update .env CORS_ORIGINS and frontend/.env PUBLIC_BACKEND_URL"
        EXTERNAL_IP="$VERIFY_IP"
    fi
    print_success "External IP: $EXTERNAL_IP"
else
    print_warning "Could not verify external IP from metadata server (using pre-configured)"
    print_success "Using pre-configured IP: $EXTERNAL_IP"
fi

echo ""
print_info "Access URLs:"
echo "  Backend API: http://$EXTERNAL_IP:$BACKEND_PORT"
echo "  API Docs:    http://$EXTERNAL_IP:$BACKEND_PORT/docs"
echo "  Frontend:    http://$EXTERNAL_IP:$FRONTEND_PORT"
echo ""

# Start backend API
print_info "Starting WiQAS Backend API on port $BACKEND_PORT..."
nohup uvicorn backend.app:app \
    --host 0.0.0.0 \
    --port $BACKEND_PORT \
    --log-level info \
    > "$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > "$LOG_DIR/backend.pid"

# Disown the process so it doesn't get killed when script exits
disown $BACKEND_PID

sleep 3

# Check if backend started
if curl -s http://localhost:$BACKEND_PORT/health > /dev/null 2>&1; then
    print_success "Backend API started successfully (PID: $BACKEND_PID)"
else
    print_error "Backend API failed to start. Check $LOG_DIR/backend.log"
    exit 1
fi

echo ""
print_success "=========================================="
print_success "   WiQAS is now running!"
print_success "=========================================="
echo ""
print_info "Service Status:"
echo "  Backend API: http://localhost:$BACKEND_PORT (PID: $BACKEND_PID)"
echo "  Logs: $LOG_DIR/"
echo ""
print_info "Useful Commands:"
echo "  Check backend: curl http://localhost:$BACKEND_PORT/health"
echo "  View logs: tail -f $LOG_DIR/backend.log"
echo "  Stop backend: kill $BACKEND_PID"
echo "  Monitor GPU: nvidia-smi -l 1"
echo "  Test API: python run.py ask 'What is Filipino culture?'"
echo ""
print_info "To start frontend (in new terminal):"
echo "  cd $WIQAS_DIR/frontend"
echo "  npm run dev -- --host 0.0.0.0 --port $FRONTEND_PORT"
echo ""
print_warning "Remember to configure firewall rules for ports $BACKEND_PORT and $FRONTEND_PORT"

# Trap Ctrl+C to only stop log viewing, not the backend
trap 'echo ""; print_info "Stopped viewing logs. Backend is still running (PID: $BACKEND_PID)"; exit 0' INT

# Keep script running to show status
echo ""
print_info "Press Ctrl+C to stop showing logs (backend will continue running)"
echo ""
tail -f "$LOG_DIR/backend.log"
