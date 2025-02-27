#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting setup...${NC}"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to setup system dependencies
setup_system_dependencies() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo -e "${YELLOW}Installing system dependencies for Linux...${NC}"
        sudo apt-get update
        sudo apt-get install -y \
            ffmpeg \
            libsndfile1 \
            python3-dev \
            build-essential \
            portaudio19-dev \
            cmake
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo -e "${YELLOW}Installing system dependencies for macOS...${NC}"
        if ! command_exists brew; then
            echo -e "${RED}Homebrew not found. Installing Homebrew...${NC}"
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew install \
            ffmpeg \
            libsndfile \
            portaudio \
            cmake
    else
        echo -e "${RED}Unsupported operating system. Please install dependencies manually.${NC}"
        echo "See system_requirements.txt for details."
        exit 1
    fi
}

# Function to initialize conda
initialize_conda() {
    # Initialize conda for shell script
    eval "$(conda shell.bash hook)"
    # Add conda-forge channel
    conda config --add channels conda-forge
    conda config --set channel_priority flexible
}

# Function to setup new conda environment
setup_new_conda() {
    echo -e "${YELLOW}Setting up new Conda environment...${NC}"
    
    # Create new conda environment
    conda create -n voxstruct python=3.10 -y
    
    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate voxstruct
    
    install_conda_packages
}

# Function to install a package with fallback to pip
install_package() {
    local package=$1
    local version=$2
    
    echo -e "${YELLOW}Trying to install $package with conda...${NC}"
    if conda install -y -c conda-forge "$package" &> /dev/null; then
        echo -e "${GREEN}Successfully installed $package with conda${NC}"
    else
        echo -e "${YELLOW}Conda install failed, trying pip...${NC}"
        if [ ! -z "$version" ]; then
            pip install --timeout 100 "$package==$version"
        else
            pip install --timeout 100 "$package"
        fi
    fi
}

# Function to install conda packages
install_conda_packages() {
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    
    # Try to install each package with conda first, fallback to pip
    declare -A packages=(
        ["python-dotenv"]="0.19.0"
        ["numpy"]="1.20.0"
        ["pandas"]="2.0.0"
        ["requests"]="2.25.0"
        ["pytest"]="7.0.0"
        ["pydub"]="0.25.0"
        ["librosa"]="0.10.0"
        ["soundfile"]="0.10.0"
        ["ffmpeg-python"]="0.2.0"
        ["scipy"]="1.7.0"
        ["matplotlib"]="3.5.0"
        ["plotly"]="5.0.0"
        ["tqdm"]="4.45.0"
        ["typing-extensions"]="4.0.0"
        ["numba"]="0.55.0"
        ["jinja2"]="3.0.0"
        ["wget"]="3.2"
    )
    
    for package in "${!packages[@]}"; do
        install_package "$package" "${packages[$package]}"
    done
    
    # Handle PyTorch separately due to CUDA requirements
    if command -v nvidia-smi &> /dev/null; then
        conda install -y pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    else
        conda install -y pytorch torchaudio cpuonly -c pytorch
    fi
    
    # Install pip-only packages
    echo -e "${YELLOW}Installing pip-only packages...${NC}"
    pip_packages=(
        "openai-whisper>=20231117"
        "vosk>=0.3.45"
        "deepspeech==0.9.3"
        "deepgram-sdk>=2.11.0"
        "wave>=0.0.2"
        "pathlib>=1.0.1"
    )
    
    for package in "${pip_packages[@]}"; do
        package_name=$(echo $package | cut -d'>' -f1 | cut -d'=' -f1)
        if [[ $package == *"=="* ]]; then
            version=$(echo $package | cut -d'=' -f3)
            pip install --timeout 100 "$package_name==$version"
        else
            version=$(echo $package | cut -d'>' -f2 | cut -d'=' -f2)
            pip install --timeout 100 "$package_name>=$version"
        fi
    done
}

# Function to check existing packages
check_existing_packages() {
    local env_name=$1
    echo -e "${YELLOW}Checking existing packages in $env_name...${NC}"
    
    # Define required packages with versions
    declare -A required_packages=(
        ["python-dotenv"]="0.19.0"
        ["numpy"]="1.20.0"
        ["pandas"]="2.0.0"
        ["requests"]="2.25.0"
        ["pytest"]="7.0.0"
        ["pydub"]="0.25.0"
        ["librosa"]="0.10.0"
        ["soundfile"]="0.10.0"
        ["ffmpeg-python"]="0.2.0"
        ["scipy"]="1.7.0"
        ["matplotlib"]="3.5.0"
        ["plotly"]="5.0.0"
        ["tqdm"]="4.45.0"
        ["typing-extensions"]="4.0.0"
        ["numba"]="0.55.0"
        ["jinja2"]="3.0.0"
        ["wget"]="3.2"
    )
    
    # Get list of installed packages with versions
    local installed_packages=$(conda list --name $env_name --json | python3 -c "
import json, sys
packages = json.load(sys.stdin)
for p in packages:
    print(f'{p[\"name\"]}=={p[\"version\"]}')
")
    
    # Create arrays for packages
    declare -a to_install=()
    declare -a existing=()
    
    # Check each required package
    for package in "${!required_packages[@]}"; do
        if echo "$installed_packages" | grep -q "^$package=="; then
            existing+=("$package")
        else
            to_install+=("$package")
        fi
    done
    
    # Show status of packages
    if [ ${#existing[@]} -gt 0 ]; then
        echo -e "${YELLOW}Found existing packages:${NC}"
        printf '  - %s\n' "${existing[@]}"
    fi
    
    echo -e "\n${YELLOW}How would you like to handle package installation?${NC}"
    echo "1) Skip existing packages (only install missing packages)"
    echo "2) Override all packages (install required versions)"
    echo "3) Interactive mode (choose for each package)"
    read -p "Enter your choice (1-3): " conflict_choice
    
    case $conflict_choice in
        1)
            if [ ${#to_install[@]} -gt 0 ]; then
                echo -e "${YELLOW}Installing missing packages:${NC}"
                printf '  - %s\n' "${to_install[@]}"
                for package in "${to_install[@]}"; do
                    install_package "$package" "${required_packages[$package]}"
                done
            else
                echo -e "${GREEN}No new packages to install${NC}"
            fi
            ;;
        2)
            echo -e "${YELLOW}Installing all required packages...${NC}"
            install_conda_packages
            ;;
        3)
            echo -e "${YELLOW}Interactive package installation...${NC}"
            for package in "${existing[@]}"; do
                read -p "Override $package? [y/N] " override
                if [[ $override =~ ^[Yy]$ ]]; then
                    install_package "$package" "${required_packages[$package]}"
                fi
            done
            # Install missing packages
            for package in "${to_install[@]}"; do
                install_package "$package" "${required_packages[$package]}"
            done
            ;;
        *)
            echo -e "${RED}Invalid choice. Exiting.${NC}"
            exit 1
            ;;
    esac
    
    # Install pip-only packages
    echo -e "${YELLOW}Installing pip-only packages...${NC}"
    pip_packages=(
        "openai-whisper>=20231117"
        "vosk>=0.3.45"
        "deepspeech==0.9.3"
        "deepgram-sdk>=2.11.0"
        "wave>=0.0.2"
        "pathlib>=1.0.1"
    )
    
    for package in "${pip_packages[@]}"; do
        package_name=$(echo $package | cut -d'>' -f1 | cut -d'=' -f1)
        if [[ $package == *"=="* ]]; then
            version=$(echo $package | cut -d'=' -f3)
            pip install --timeout 100 "$package_name==$version"
        else
            version=$(echo $package | cut -d'>' -f2 | cut -d'=' -f2)
            pip install --timeout 100 "$package_name>=$version"
        fi
    done
}

# Function to use existing conda environment
use_existing_conda() {
    if [[ ! -z "${CONDA_DEFAULT_ENV}" ]]; then
        echo -e "${YELLOW}Using current environment: ${CONDA_DEFAULT_ENV}${NC}"
        # Check and handle existing packages
        check_existing_packages "${CONDA_DEFAULT_ENV}"
    else
        echo -e "${RED}Not in a conda environment. Please activate your environment first.${NC}"
        exit 1
    fi
}

# Function to setup venv environment
setup_venv() {
    echo -e "${YELLOW}Setting up Python virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip install -r requirements.txt
}

# Ask user which environment they prefer
echo -e "${YELLOW}Which Python environment would you like to use?${NC}"
if [[ ! -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo -e "${GREEN}Currently in conda environment: ${CONDA_DEFAULT_ENV}${NC}"
    echo "1) Create new Conda environment"
    echo "2) Use current Conda environment (${CONDA_DEFAULT_ENV})"
    echo "3) Use Venv (standard Python virtual environment)"
else
    echo "1) New Conda environment (recommended for GPU support and better dependency management)"
    echo "2) Existing Conda environment (install packages in your current environment)"
    echo "3) Venv (standard Python virtual environment)"
fi
read -p "Enter your choice (1, 2, or 3): " env_choice

# Install system dependencies
setup_system_dependencies

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p models output converted segments comparison_results

# Setup Python environment based on user choice
case $env_choice in
    1)
        if ! command_exists conda; then
            echo -e "${RED}Conda not found. Please install Conda first.${NC}"
            echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
            exit 1
        fi
        setup_new_conda
        ;;
    2)
        if ! command_exists conda; then
            echo -e "${RED}Conda not found. Please install Conda first.${NC}"
            echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
            exit 1
        fi
        use_existing_conda
        ;;
    3)
        setup_venv
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

# Download default models
echo -e "${YELLOW}Downloading default models...${NC}"
python3 - << EOF
import os
import wget
from vosk import Model
import whisper

# Download Vosk model
vosk_model = "vosk-model-small-en-us"
if not os.path.exists(f"models/{vosk_model}"):
    print("Downloading Vosk model...")
    wget.download(f"https://alphacephei.com/vosk/models/{vosk_model}.zip")
    os.system(f"unzip {vosk_model}.zip -d models/")
    os.remove(f"{vosk_model}.zip")

# Download DeepSpeech model
deepspeech_model = "deepspeech-0.9.3-models.pbmm"
if not os.path.exists(f"models/{deepspeech_model}"):
    print("\nDownloading DeepSpeech model...")
    wget.download(f"https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/{deepspeech_model}", "models/")

# Download Whisper model (this will happen automatically when first used)
print("\nPreparing Whisper model...")
whisper.load_model("base")
EOF

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${YELLOW}To get started:${NC}"

if [ "$env_choice" == "1" ]; then
    echo "1. Activate the conda environment: conda activate voxstruct"
elif [ "$env_choice" == "2" ]; then
    echo "1. Your existing conda environment is already activated"
else
    echo "1. Activate the virtual environment: source venv/bin/activate"
fi

echo "2. Place your audio file in the project directory"
echo "3. Run: python scripts/compare_engines.py your_audio_file.mp3"

# Create a conda environment file if using new conda environment
if [ "$env_choice" == "1" ]; then
    conda env export > environment.yml
    echo -e "${YELLOW}Conda environment exported to environment.yml${NC}"
fi 