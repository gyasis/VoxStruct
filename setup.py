from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="voxstruct",
    version="0.1.2",
    author="gyasis",
    author_email="your.email@example.com",
    description="A tool for structured audio transcription with LLM supervision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gyasis/VoxStruct",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "litellm>=1.0.0",
        "python-dotenv>=0.19.0",
        "pydub>=0.25.1",
        "openai-whisper",
        "whisper-timestamped",
        "vosk>=0.3.42",
        "anthropic>=0.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=3.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "voxstruct=voxstruct.main:main",
        ],
    },
) 