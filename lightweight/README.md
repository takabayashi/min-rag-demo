# Lightweight RAG System

A lightweight RAG (Retrieval-Augmented Generation) system with multiple backend options for optimal performance on different hardware.

## 🚀 Performance Optimizations

### Multiple Backend Support
- **PyTorch (LocalLLM)**: Traditional PyTorch implementation with MPS/GPU acceleration
- **MLX (MLXLLM)**: Apple Silicon optimized backend for faster inference
- **Ollama**: Local LLM server for easy model management

### Performance Features
- **FP16 Precision**: Automatic half-precision for faster inference on supported devices
- **MPS Acceleration**: Native Apple Silicon GPU support
- **Input Length Limiting**: Configurable max input length for faster processing
- **Device Auto-Detection**: Automatic detection of best available device (MPS > CUDA > CPU)

## Quick Start

```bash
# Install dependencies
uv sync

# Add your markdown files to the faqs/ directory

# Run the system
uv run python main.py
```

## 🚀 Performance Configuration

### 1. **For Apple Silicon (M1/M2/M3) - Recommended**
```bash
# Install MLX for optimal performance
make install-mlx

# Edit config.py
USE_MLX = True
USE_FP16 = True
```

### 2. **For NVIDIA GPU**
```bash
# Edit config.py
USE_OLLAMA = False
USE_MLX = False
USE_FP16 = True
AVAILABLE_DEVICES = "cuda"
```

### 3. **For CPU-only**
```bash
# Edit config.py
USE_OLLAMA = False
USE_MLX = False
USE_FP16 = False
AVAILABLE_DEVICES = "cpu"
```

### 4. **Using Ollama (Easiest)**
```bash
# Install and start Ollama
make install-ollama-models
make start-ollama

# Edit config.py
USE_OLLAMA = True
```

### 5. **Performance Benchmarking**
```bash
# Test performance of different backends
make benchmark
```

## Jupyter Lab Integration

```bash
# Install Jupyter Lab and kernel
uv add jupyterlab ipykernel ipywidgets

# Install kernel for this environment
uv run python -m ipykernel install --user --name=lightweight-rag --display-name="Lightweight RAG"

# Start Jupyter Lab
uv run jupyter lab
```

## Features

- ✅ **Minimal Dependencies** - Only PyTorch, transformers, scikit-learn, numpy
- ✅ **Local Embeddings** - Uses local transformer models (no API calls)
- ✅ **Local LLM Integration** - DeepSeek with PyTorch optimization
- ✅ **Simple Vector Store** - Dictionary-based storage with numpy arrays
- ✅ **Cosine Similarity** - Fast similarity search using scikit-learn
- ✅ **Document Chunking** - Smart chunking based on markdown headers
- ✅ **Interactive CLI** - Easy-to-use query interface
- ✅ **Performance Benchmarking** - Built-in speed testing
- ✅ **Persistent Storage** - Save/load vector store to disk

## Files

- `main.py` - Main RAG system implementation
- `pyproject.toml` - Project configuration and dependencies
- `README_DETAILED.md` - Detailed documentation

## Requirements

- Python 3.8+
- 8GB+ RAM (for DeepSeek model)
- 13GB+ disk space (for model download)
- GPU recommended (CUDA compatible)

## Usage

1. **Install dependencies**: `uv sync`
2. **Add documents**: Place markdown files in `../faqs/` directory
3. **Run system**: `uv run python main.py`
4. **Interactive mode**: Ask questions and get AI-powered answers

### Jupyter Lab Usage

1. **Install Jupyter**: `uv add jupyterlab ipykernel ipywidgets`
2. **Install kernel**: `uv run python -m ipykernel install --user --name=lightweight-rag --display-name="Lightweight RAG"`
3. **Start Jupyter**: `uv run jupyter lab`
4. **Select kernel**: Choose "Lightweight RAG" kernel in your notebooks

## 🤖 LLM Model Alternatives

### **Code-Focused Models (Recommended for RAG)**
```python
# DeepSeek Models (Code Specialists)
"deepseek-ai/deepseek-coder-1.3b-instruct"      # 1.3B params, fast, lightweight
"deepseek-ai/deepseek-coder-6.7b-instruct"      # 6.7B params, balanced (current default)
"deepseek-ai/deepseek-coder-33b-instruct"       # 33B params, high quality, needs more RAM

# CodeLlama Models (Meta's Code LLM)
"codellama/CodeLlama-7b-Instruct-hf"            # 7B params, good code understanding
"codellama/CodeLlama-13b-Instruct-hf"           # 13B params, better quality
"codellama/CodeLlama-34b-Instruct-hf"           # 34B params, excellent quality

# WizardCoder Models
"WizardLM/WizardCoder-15B-V1.0"                 # 15B params, strong coding abilities
"WizardLM/WizardCoder-Python-34B-V1.0"          # 34B params, Python specialist
```

### **General Purpose Models**
```python
# Llama Models
"meta-llama/Llama-2-7b-chat-hf"                 # 7B params, general purpose
"meta-llama/Llama-2-13b-chat-hf"                # 13B params, better reasoning
"meta-llama/Llama-2-70b-chat-hf"                # 70B params, best quality, needs 40GB+ RAM

# Mistral Models
"mistralai/Mistral-7B-Instruct-v0.2"            # 7B params, excellent performance
"mistralai/Mixtral-8x7B-Instruct-v0.1"          # 8x7B params, MoE architecture

# Phi Models (Microsoft)
"microsoft/phi-2"                                # 2.7B params, very fast, good quality
"microsoft/phi-3-mini-4k-instruct"              # 3.8B params, optimized for instruction
```

### **Lightweight Models (Fast Inference)**
```python
# Tiny Models
"microsoft/DialoGPT-small"                       # 117M params, very fast
"microsoft/DialoGPT-medium"                      # 345M params, fast
"gpt2"                                           # 124M params, classic, fast

# Distilled Models
"distilgpt2"                                     # 82M params, ultra-fast
"microsoft/DialoGPT-small"                       # 117M params, fast
```

### **How to Change Models**
```python
# In main.py, change the LLM_MODEL constant:
LLM_MODEL = "codellama/CodeLlama-7b-Instruct-hf"  # Example

# Or when creating the LLM:
llm = LocalLLM(model_name="mistralai/Mistral-7B-Instruct-v0.2")
```

For detailed documentation, see `README_DETAILED.md`.
