#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Environment Setup for Pizza Detector Project${NC}"

# 1. Check for System Dependencies (Linux/Debian/Ubuntu)
if [ -f /etc/debian_version ]; then
    echo -e "${YELLOW}Checking system dependencies...${NC}"
    
    REQUIRED_PACKAGES="libosmesa6-dev libgl1-mesa-glx libglib2.0-0 python3-venv python3-dev build-essential"
    MISSING_PACKAGES=""
    
    for pkg in $REQUIRED_PACKAGES; do
        if ! dpkg -l | grep -q " $pkg "; then
            MISSING_PACKAGES="$MISSING_PACKAGES $pkg"
        fi
    done
    
    if [ ! -z "$MISSING_PACKAGES" ]; then
        echo -e "${YELLOW}Missing system packages detected: $MISSING_PACKAGES${NC}"
        echo "Installing missing packages (requires sudo)..."
        sudo apt-get update
        sudo apt-get install -y $MISSING_PACKAGES
    else
        echo -e "${GREEN}All system dependencies are installed.${NC}"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${YELLOW}MacOS detected. Please ensure you have installed dependencies via brew.${NC}"
    # Add brew commands if known, otherwise just warn
    echo "Recommended: brew install mesa-glu"
else
    echo -e "${YELLOW}Non-Debian Linux or other OS detected. Please manually ensure 'libosmesa6-dev' and OpenGL libraries are installed.${NC}"
fi

# 2. Setup Python Virtual Environment
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating Python virtual environment in $VENV_DIR...${NC}"
    python3 -m venv $VENV_DIR
else
    echo -e "${GREEN}Virtual environment already exists.${NC}"
fi

# 3. Activate and Install Python Requirements
echo -e "${YELLOW}Installing Python dependencies...${NC}"
source $VENV_DIR/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}Python dependencies installed.${NC}"
else
    echo -e "${RED}requirements.txt not found!${NC}"
    exit 1
fi

# 4. Install Spatial/3D Requirements if they exist
if [ -f "requirements_spatial.txt" ]; then
    echo -e "${YELLOW}Installing Spatial/3D dependencies...${NC}"
    pip install -r requirements_spatial.txt
fi

# 5. Setup Environment Variables (Optional)
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating default .env file...${NC}"
    echo "PYRENDER_OFFSCREEN=1" > .env
    echo "DATA_DIR=augmented_pizza" >> .env
    echo -e "${GREEN}.env file created.${NC}"
fi

echo -e "${GREEN}Setup Complete!${NC}"
echo "To activate the environment, run: source $VENV_DIR/bin/activate"
