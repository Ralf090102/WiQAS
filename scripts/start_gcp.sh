#!/bin/bash
# WiQAS GCP Startup Script
# VM: wiqas-generation-evaluation-20260120-061404
# Location: asia-southeast1-c
# GPU: 1x NVIDIA A100 40GB

set -e

# Configuration
WIQAS_DIR="/shared/WiQAS"
VENV_PATH="/shared/wiqas-venv"
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
EXTERNAL_IP="34.124.143.216"  # Pre-configured
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
# If another process is listening on the backend port, warn and try to stop it
is_port_in_use() {
    if command -v ss >/dev/null 2>&1; then
        ss -ltn "sport = :$1" | grep -q LISTEN
        return $?
    elif command -v lsof >/dev/null 2>&1; then
        lsof -iTCP:$1 -sTCP:LISTEN -t >/dev/null 2>&1
        return $?
    else
        return 1
    fi
}

kill_process_on_port() {
    if command -v lsof >/dev/null 2>&1; then
        PIDS=$(lsof -iTCP:$1 -sTCP:LISTEN -t || true)
    elif command -v ss >/dev/null 2>&1; then
        PIDS=$(ss -ltnp "sport = :$1" 2>/dev/null | awk -F"pid=|," '/users:/{print $2}' | tr '\n' ' ')
    else
        PIDS=""
    fi
    if [ -n "$PIDS" ]; then
        print_warning "Port $1 is in use by PIDs: $PIDS — attempting to kill"
        for p in $PIDS; do
            kill -TERM $p 2>/dev/null || kill -9 $p 2>/dev/null || true
        done
        sleep 1
    fi
}

# Ensure port is free before starting
if is_port_in_use $BACKEND_PORT; then
    print_warning "Port $BACKEND_PORT appears in use — attempting to free it"
    kill_process_on_port $BACKEND_PORT
    sleep 1
fi

# Truncate backend log so old shutdown messages don't appear
> "$LOG_DIR/backend.log"

nohup uvicorn backend.app:app \
    --host 0.0.0.0 \
    --port $BACKEND_PORT \
    --log-level info \
    > "$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > "$LOG_DIR/backend.pid"
disown $BACKEND_PID

# Wait for healthy /health endpoint with timeout
MAX_WAIT=15
WAITED=0
until curl -s http://localhost:$BACKEND_PORT/health > /dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        print_error "Backend API failed to start within ${MAX_WAIT}s. Check $LOG_DIR/backend.log"
        # show last lines to help debugging
        tail -n 200 "$LOG_DIR/backend.log"
        exit 1
    fi
    sleep 1
    WAITED=$((WAITED+1))
done
print_success "Backend API started successfully (PID: $BACKEND_PID)"

# Start frontend (build + preview) so the app is fully accessible
print_info "Starting Frontend (build + preview) on port $FRONTEND_PORT..."
if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    cd "$WIQAS_DIR/frontend" || { print_error "Failed to cd to frontend"; exit 1; }

    # Ensure port is free
    if is_port_in_use $FRONTEND_PORT; then
        print_warning "Port $FRONTEND_PORT appears in use — attempting to free it"
        kill_process_on_port $FRONTEND_PORT
        sleep 1
    fi

    # Install ALL dependencies (devDependencies needed for build: vite, svelte-kit, etc.)
    INSTALL_SUCCESS=false
    if [ -f package-lock.json ]; then
        print_info "Installing frontend dependencies (npm ci)..."
        if npm ci > "$LOG_DIR/frontend_install.log" 2>&1; then
            INSTALL_SUCCESS=true
        else
            print_warning "npm ci failed, cleaning and trying fresh install..."
            rm -rf node_modules package-lock.json
            if npm install > "$LOG_DIR/frontend_install.log" 2>&1; then
                INSTALL_SUCCESS=true
            fi
        fi
    else
        print_info "No lockfile found, running npm install..."
        if npm install > "$LOG_DIR/frontend_install.log" 2>&1; then
            INSTALL_SUCCESS=true
        fi
    fi

    if [ "$INSTALL_SUCCESS" = false ]; then
        print_error "Frontend install failed completely — check $LOG_DIR/frontend_install.log"
        cd "$WIQAS_DIR" || true
        # Continue anyway, backend is running
    else
        # Verify critical packages are installed
        print_info "Verifying critical packages..."
        MISSING_PACKAGES=""
        for pkg in "vite" "@sveltejs/kit" "unplugin-icons" "svelte"; do
            if [ ! -d "node_modules/$pkg" ]; then
                MISSING_PACKAGES="$MISSING_PACKAGES $pkg"
            fi
        done
        
        if [ -n "$MISSING_PACKAGES" ]; then
            print_warning "Missing packages:$MISSING_PACKAGES — installing explicitly..."
            npm install $MISSING_PACKAGES --save-dev >> "$LOG_DIR/frontend_install.log" 2>&1 || print_warning "Explicit install failed"
        fi

        # Build frontend
        print_info "Building frontend..."
        if ! npm run build > "$LOG_DIR/frontend_build.log" 2>&1; then
            print_warning "Build failed, trying clean install + rebuild..."
            rm -rf node_modules package-lock.json .svelte-kit
            npm install > "$LOG_DIR/frontend_install.log" 2>&1
            if npm run build > "$LOG_DIR/frontend_build.log" 2>&1; then
                print_success "Frontend built successfully after retry"
            else
                print_error "Frontend build failed after retry — check $LOG_DIR/frontend_build.log"
                cd "$WIQAS_DIR" || true
                # Continue, backend is still running
            fi
        fi
    fi

    # Only start preview if build directory exists
    if [ -d "build" ] || [ -d ".svelte-kit" ]; then
        # Start Vite preview in background
        nohup npm run preview -- --host 0.0.0.0 --port $FRONTEND_PORT > "$LOG_DIR/frontend.log" 2>&1 &
        FRONTEND_PID=$!
        echo $FRONTEND_PID > "$LOG_DIR/frontend.pid"
        disown $FRONTEND_PID

        # Return to repo root
        cd "$WIQAS_DIR" || true

        sleep 2
        if curl -sS http://localhost:$FRONTEND_PORT/ > /dev/null 2>&1; then
            print_success "Frontend started successfully (PID: $FRONTEND_PID)"
        else
            print_warning "Frontend may not be responding yet — check $LOG_DIR/frontend.log"
        fi
    else
        print_error "No build output found — frontend cannot start"
        cd "$WIQAS_DIR" || true
    fi
else
    print_warning "Node.js/npm not found. Skipping frontend startup. Install Node.js to run the frontend on this VM."
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
print_warning "Remember to configure firewall rules for ports $BACKEND_PORT and $FRONTEND_PORT"

# Trap Ctrl+C to only stop log viewing, not the services
trap 'echo ""; print_info "Stopped viewing logs. Services still running (Backend PID: $BACKEND_PID, Frontend PID: ${FRONTEND_PID:-N/A})"; exit 0' INT

# Keep script running to show status
echo ""
print_info "Press Ctrl+C to stop showing logs (services will continue running)"
echo ""

# Show both backend and frontend logs
if [ -f "$LOG_DIR/frontend.log" ]; then
    tail -f "$LOG_DIR/backend.log" "$LOG_DIR/frontend.log"
else
    tail -f "$LOG_DIR/backend.log"
fi
