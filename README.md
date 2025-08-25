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
- **chromadb**: Vector database
- **sentence-transformers**: Embedding models
- **python-dotenv**: Environment variable management

## Usage

### Adding Your Documents

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
- Automatically load all .md documents from the `docs/` folder
- Create embeddings and store them in ChromaDB
- Start an interactive CLI where you can ask questions
- Show answers with references to source documents and chunks

## Project Structure

```
min-rag-demo/
├── docs/                # Document storage directory
│   ├── python_basics.md
│   ├── machine_learning.md
│   └── rag_systems.md
├── main.py              # Main file
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
OPENAI_API_KEY=your_openai_key_here
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
