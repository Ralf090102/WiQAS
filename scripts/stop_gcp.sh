#!/bin/bash
# WiQAS GCP Stop Script

WIQAS_DIR="/home/ralf_hernandez/WiQAS"
LOG_DIR="$WIQAS_DIR/logs"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

echo "=========================================="
echo "   Stopping WiQAS Services"
echo "=========================================="
echo ""

# Stop backend
if [ -f "$LOG_DIR/backend.pid" ]; then
    BACKEND_PID=$(cat "$LOG_DIR/backend.pid")
    if ps -p $BACKEND_PID > /dev/null 2>&1; then
        print_info "Stopping backend (PID: $BACKEND_PID)..."
        kill $BACKEND_PID
        sleep 2
        if ps -p $BACKEND_PID > /dev/null 2>&1; then
            print_warning "Backend still running, force killing..."
            kill -9 $BACKEND_PID
        fi
        rm "$LOG_DIR/backend.pid"
        print_success "Backend stopped"
    else
        print_warning "Backend not running (stale PID file)"
        rm "$LOG_DIR/backend.pid"
    fi
else
    print_warning "Backend PID file not found"
fi

# Stop any uvicorn processes
print_info "Checking for stray uvicorn processes..."
UVICORN_PIDS=$(pgrep -f "uvicorn backend.app:app")
if [ ! -z "$UVICORN_PIDS" ]; then
    print_warning "Found running uvicorn processes: $UVICORN_PIDS"
    echo "$UVICORN_PIDS" | xargs kill
    print_success "Stopped uvicorn processes"
else
    print_info "No stray uvicorn processes found"
fi

# Optionally stop Ollama
read -p "Stop Ollama service? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Stopping Ollama..."
    pkill ollama
    print_success "Ollama stopped"
fi

echo ""
print_success "WiQAS services stopped"
