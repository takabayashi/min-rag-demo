"""
Configuration settings for the Min RAG Demo
"""

import os
from pathlib import Path

# Document settings
DOCS_DIR = Path('./docs/')
SUPPORTED_EXTENSIONS = ['.md']

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 250
HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]

# Vector store settings
COLLECTION_NAME = "faq_context_collections"
EMBEDDING_MODEL = "nomic-embed-text"

# Retrieval settings
DEFAULT_K = 5
DEFAULT_THRESHOLD = 1.0

# LLM settings
DEFAULT_TEMPERATURE = 0
OLLAMA_MODEL = "deepseek-r1" #deepseek-r1, mistral, llama3.1, gpt-oss:latest
OPENAI_MODEL = "gpt-3.5-turbo"

# Environment variables
USE_OPENAI_EMBEDDINGS = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "false"
USE_OPENAI_LLM = os.getenv("USE_OPENAI_LLM", "false").lower() == "false"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Display settings
MAX_CONTENT_LENGTH = 300
SEPARATOR_LENGTH = 50