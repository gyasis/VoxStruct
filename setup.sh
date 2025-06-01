#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting VoxStruct Setup...${NC}"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to setup system dependencies
install_sys_deps() { # Renamed from setup_system_dependencies for consistency
    echo -e "${YELLOW}Checking and installing system dependencies...${NC}"
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Detected Linux. Using apt-get..."
        # Check if sudo is available, otherwise notify user
        if ! command_exists sudo; then
            echo -e "${RED}sudo command not found. Please install the following packages manually:${NC}"
            echo "ffmpeg libsndfile1 python3-dev build-essential portaudio19-dev cmake"
            return 1 # Indicate failure
        fi
        # Update package list and install
        sudo apt-get update || { echo -e "${RED}apt-get update failed.${NC}"; return 1; }
        sudo apt-get install -y \
            ffmpeg \
            libsndfile1 \
            python3-dev \
            build-essential \
            portaudio19-dev \
            cmake || { echo -e "${RED}apt-get install failed for one or more packages.${NC}"; return 1; }
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Detected macOS. Using Homebrew..."
        if ! command_exists brew; then
            echo -e "${YELLOW}Homebrew not found. Attempting to install Homebrew...${NC}"
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || { echo -e "${RED}Homebrew installation failed.${NC}"; return 1; }
            # Need to add brew to PATH for the current script execution
            # This depends on the architecture (Apple Silicon vs Intel)
            if [[ -x /opt/homebrew/bin/brew ]]; then
                 eval "$(/opt/homebrew/bin/brew shellenv)"
            elif [[ -x /usr/local/bin/brew ]]; then
                 eval "$(/usr/local/bin/brew shellenv)"
            fi
        fi
        # Install dependencies using brew
        brew install \
            ffmpeg \
            libsndfile \
            portaudio \
            cmake || { echo -e "${RED}brew install failed for one or more packages.${NC}"; return 1; }
    else
        echo -e "${RED}Unsupported operating system: $OSTYPE ${NC}"
        echo "Please install the required system dependencies manually."
        echo "See system_requirements.txt for details."
        return 1 # Indicate failure
    fi
    echo -e "${GREEN}System dependencies check/installation complete.${NC}"
    return 0 # Indicate success
}

# === Main Setup Logic ===

# 1. Check System Dependencies
echo -e "\\nStep 1: Checking system dependencies..."
if ! install_sys_deps; then
    echo -e "${RED}System dependency installation failed. Please resolve the issues above and retry. Exiting.${NC}"
    exit 1
fi

# 2. Install Python packages using pip
echo -e "\\nStep 2: Installing Python packages using pip..."

# Determine which pip command to use
PIP_CMD=""
if command_exists pip3; then
    PIP_CMD="pip3"
elif command_exists pip; then
    PIP_CMD="pip"
else
    echo -e "${RED}Error: Neither pip nor pip3 found. Cannot install Python packages. Exiting.${NC}"
    exit 1
fi

echo "Using '$PIP_CMD' to install packages."

# Upgrade pip
echo "Upgrading pip..."
$PIP_CMD install --upgrade pip || { echo -e "${RED}Failed to upgrade pip. Continuing anyway...${NC}"; }

# Install core and dev dependencies from setup.py into the current environment
echo "Installing VoxStruct and its dependencies (editable mode)..."
echo "This will install packages into the Python environment currently managed by '$PIP_CMD'."
$PIP_CMD install -e ".[dev]" || { echo -e "${RED}Failed to install Python packages using $PIP_CMD. Exiting.${NC}"; exit 1; }

echo -e "${GREEN}Python packages installed successfully.${NC}"

# 3. Final Instructions
echo -e "\\n=== Setup Complete! ==="
echo "VoxStruct and its development dependencies have been installed in the current Python environment using '$PIP_CMD'."
echo "This was an 'editable' install (-e), meaning changes you make in the 'src/' directory will be reflected when you run the command."
echo ""
echo "You should now be able to run the 'voxstruct' command directly from your terminal, provided the install location is in your PATH."
echo "(If you installed globally without a venv, this is usually the case)."
echo ""
echo "Example: voxstruct --help"
echo "Example: voxstruct your_audio.mp3 --timestamp-granularity word"
echo ""
echo "If the command isn't found, ensure the directory where pip installed the script ('voxstruct') is included in your system's PATH environment variable."

exit 0 # Exit successfully

# --- Removed Conda/Venv/Pipenv specific functions and logic --- 