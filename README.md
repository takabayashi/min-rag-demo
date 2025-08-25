# Min RAG Demo

A RAG (Retrieval-Augmented Generation) demonstration project using LangChain.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for Python dependency management.

### Prerequisites

- Python 3.11+
- uv installed

### Installation

1. Clone the repository:
```bash
git clone <your-repository>
cd min-rag-demo
```

2. Install dependencies:
```bash
uv sync
```

3. Activate the virtual environment:
```bash
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

## Main Dependencies

- **langchain**: Main framework for RAG
- **langchain-openai**: OpenAI integration
- **langchain-community**: Community integrations
- **langchain-ollama**: Ollama integration for local LLMs
- **chromadb**: Vector database
- **python-dotenv**: Environment variable management

## Usage

### Included FAQ Documents

This project comes with comprehensive FAQ documents for Python and Java programming:

**Python FAQ (5 files):**
- `python_basics_faq.md` - Basic concepts, syntax, data types
- `python_oop_faq.md` - Object-oriented programming concepts
- `python_advanced_faq.md` - Advanced topics like decorators, generators
- `python_libraries_faq.md` - Popular libraries and frameworks
- `python_best_practices_faq.md` - Best practices and debugging

**Java FAQ (5 files):**
- `java_basics_faq.md` - Basic concepts, syntax, data types
- `java_oop_faq.md` - Object-oriented programming concepts
- `java_collections_faq.md` - Collections framework and data structures
- `java_exceptions_faq.md` - Exception handling and multithreading
- `java_frameworks_faq.md` - Popular frameworks and libraries

### Adding Your Own Documents

1. Create documents in the `docs/` folder:
   - Supported formats: `.md` (Markdown files only)
   - The system will automatically load all .md files recursively
   - Files can be organized in subdirectories

2. Example document structure:
```
docs/
├── company_docs/
│   ├── handbook.md
│   └── policies.md
├── technical_guides/
│   ├── api_guide.md
│   └── setup_instructions.md
└── knowledge_base.md
```

### Running the Application

Run the main file:
```bash
python main.py
```

The system will:
- Automatically load all .md documents from the `docs/` folder (including the included FAQ documents)
- Create embeddings and store them in ChromaDB
- Start an interactive CLI where you can ask questions about Python, Java, or any other topics in your documents
- Show answers with references to source documents and chunks

### Example Usage

Once the system is running, you can ask questions like:

```
Enter your question: what is Python?
Enter your question: how do I create a class in Java?
Enter your question: what are decorators in Python?
Enter your question: how do I handle exceptions in Java?
Enter your question: quit
```

The system will provide detailed answers with references to the relevant FAQ documents.

## Project Structure

```
min-rag-demo/
├── docs/                # Document storage directory
│   ├── python_basics_faq.md
│   ├── python_oop_faq.md
│   ├── python_advanced_faq.md
│   ├── python_libraries_faq.md
│   ├── python_best_practices_faq.md
│   ├── java_basics_faq.md
│   ├── java_oop_faq.md
│   ├── java_collections_faq.md
│   ├── java_exceptions_faq.md
│   └── java_frameworks_faq.md
├── main.py              # Main application file
├── config.py            # Configuration settings
├── pyproject.toml       # Project configuration
├── uv.lock             # Dependency lock file
├── README.md           # This file
├── .gitignore          # Git ignore file
├── chroma_db/          # Vector database (auto-generated)
└── .venv/              # Virtual environment
```

## Environment Variables Setup

Create a `.env` file in the project root with your API keys:

```env
# OpenAI Configuration (Optional)
# Set to "true" to use OpenAI embeddings and LLM instead of Ollama
OPENAI_API_KEY=your_openai_api_key_here
USE_OPENAI_EMBEDDINGS=false
USE_OPENAI_LLM=false

# Note: If you don't have OpenAI API key or prefer to use Ollama (default):
# - Keep USE_OPENAI_EMBEDDINGS=false
# - Keep USE_OPENAI_LLM=false
# - Make sure you have Ollama installed and running locally
```

## Development

To add new dependencies:
```bash
uv add package-name
```

To remove dependencies:
```bash
uv remove package-name
```

## Customization

### Supported File Types
Currently supported: `.md` (Markdown files only)

To add support for more file types, modify the `load_documents_from_directory()` function in `main.py`.

### Chunking Strategy
Document chunking can be adjusted in the `setup_rag_system()` function:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Adjust chunk size
    chunk_overlap=200,      # Adjust overlap
    separators=["\n\n", "\n", " ", ""]
)
```

### Retrieval Settings
Modify retrieval parameters:
```python
retriever=vectorstore.as_retriever(search_kwargs={"k": 3})  # Number of documents to retrieve
```
