#!/bin/bash
# UIDAI Hackathon - Quick Start Script
# Author: Shuvam Banerji Seal's Team

set -e

echo "🚀 UIDAI Aadhaar Data Analysis Platform"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check for required commands
check_requirements() {
    print_status "Checking requirements..."
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON=python3
        print_success "Python3 found: $($PYTHON --version)"
    elif command -v python &> /dev/null; then
        PYTHON=python
        print_success "Python found: $($PYTHON --version)"
    else
        print_error "Python not found. Please install Python 3.11+"
        exit 1
    fi
    
    # Check Node.js (optional for frontend)
    if command -v node &> /dev/null; then
        print_success "Node.js found: $(node --version)"
    else
        print_warning "Node.js not found. Frontend features will be limited."
    fi
    
    # Check Git LFS
    if command -v git-lfs &> /dev/null; then
        print_success "Git LFS found"
    else
        print_warning "Git LFS not found. Large files may not download properly."
    fi
    
    # Check Docker (optional)
    if command -v docker &> /dev/null; then
        print_success "Docker found: $(docker --version | head -1)"
    else
        print_warning "Docker not found. Docker features will be unavailable."
    fi
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        $PYTHON -m pip install -r requirements.txt --quiet
        print_success "Core dependencies installed"
    fi
    
    if [ -f "requirements-ml.txt" ]; then
        print_status "Installing ML dependencies (this may take a while)..."
        $PYTHON -m pip install -r requirements-ml.txt --quiet || {
            print_warning "Some ML dependencies failed to install. Basic analysis will still work."
        }
    fi
}

# Install frontend dependencies
install_frontend_deps() {
    if [ -d "web/frontend" ] && command -v npm &> /dev/null; then
        print_status "Installing frontend dependencies..."
        cd web/frontend
        npm install --silent
        cd ../..
        print_success "Frontend dependencies installed"
    fi
}

# Run analysis
run_analysis() {
    print_status "Running analysis pipeline..."
    
    cd analysis/codes
    
    $PYTHON run_all_analyses.py --output-dir ../../web/frontend/public/data 2>&1 | while read line; do
        echo "  $line"
    done
    
    cd ../..
    
    print_success "Analysis completed"
}

# Run frontend development server
run_frontend_dev() {
    if [ -d "web/frontend" ] && command -v npm &> /dev/null; then
        print_status "Starting frontend development server..."
        cd web/frontend
        npm run dev
    else
        print_error "Frontend not available or npm not installed"
    fi
}

# Build frontend
build_frontend() {
    if [ -d "web/frontend" ] && command -v npm &> /dev/null; then
        print_status "Building frontend for production..."
        cd web/frontend
        npm run build
        cd ../..
        print_success "Frontend built successfully in web/frontend/dist/"
    else
        print_error "Frontend not available or npm not installed"
    fi
}

# Run with Docker
run_docker() {
    if command -v docker-compose &> /dev/null; then
        print_status "Running with Docker Compose..."
        docker-compose up $1
    elif command -v docker &> /dev/null && command -v docker compose &> /dev/null; then
        print_status "Running with Docker Compose..."
        docker compose up $1
    else
        print_error "Docker Compose not found"
        exit 1
    fi
}

# Show help
show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  setup       - Install all dependencies"
    echo "  analysis    - Run the analysis pipeline"
    echo "  frontend    - Start frontend development server"
    echo "  build       - Build frontend for production"
    echo "  docker      - Run everything with Docker"
    echo "  help        - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup            # First time setup"
    echo "  $0 analysis         # Run analysis"
    echo "  $0 frontend         # Start dev server"
    echo "  $0 docker analysis  # Run analysis in Docker"
}

# Main
case "$1" in
    setup)
        check_requirements
        install_python_deps
        install_frontend_deps
        print_success "Setup complete!"
        ;;
    analysis)
        run_analysis
        ;;
    frontend)
        run_frontend_dev
        ;;
    build)
        build_frontend
        ;;
    docker)
        run_docker "${@:2}"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        check_requirements
        echo ""
        show_help
        ;;
esac
