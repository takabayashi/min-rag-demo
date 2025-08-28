# Min RAG Demo

A stateless RAG (Retrieval-Augmented Generation) demonstration project using LangChain. 
In this demo a FAQ assitent is simulate. the context of the assisten is define by the .md files available on the ./docs folder.
This repo is just an example of basic steps needed for build a RAG system to work properly and is not recomended for production enviroments.

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
- **pytest**: Testing framework

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
├── tests/               # Test files
│   ├── __init__.py
│   ├── test_main.py
│   ├── test_main_pytest.py
│   ├── test_rag_system.py
│   └── test_local_model_validation.py
├── main.py              # Main application file
├── config.py            # Configuration settings
├── pyproject.toml       # Project configuration
├── pytest.ini          # Pytest configuration
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

## Testing

This project includes comprehensive unit and integration tests to ensure code quality and reliability.

### Running Tests

Run all tests:
```bash
python -m pytest
```

Run tests with verbose output:
```bash
python -m pytest -v
```

Run only unit tests (no integration tests):
```bash
python -m pytest -m "not integration"
```

Run only integration tests (requires Ollama running):
```bash
python -m pytest -m integration
```

Run a specific test file:
```bash
python -m pytest tests/test_main_pytest.py
```

Run RAG system tests (requires Ollama running):
```bash
python -m pytest tests/test_rag_system.py -v
```

Run local model validation tests (requires Ollama running):
```bash
python -m pytest tests/test_local_model_validation.py -v
```

Run a specific test:
```bash
python -m pytest tests/test_main_pytest.py::TestSplitDocuments::test_split_documents_basic
```

### Test Coverage

Currently, the tests cover:
- Document splitting functionality
- Edge cases (empty text, no headers)
- Large text handling
- Metadata preservation
- Prompt template creation and validation
- Security features in prompts
- Mock examples for external dependencies
- **Integration tests with real local models**
- **RAG system validation using actual Ollama models**
- **Prompt quality and response validation**
- **Context relevance testing**
- **Complex query handling**

### Test Files

- `tests/test_main.py` - Basic unittest tests
- `tests/test_main_pytest.py` - Comprehensive pytest tests
- `tests/test_rag_system.py` - **Integration tests for RAG system using real local models**
- `tests/test_local_model_validation.py` - **Advanced validation tests using local models**

### Integration Tests

The integration tests require:
- Ollama running locally
- The `deepseek-r1` model available in Ollama
- Documents in the `docs/` folder

These tests validate:
- Real RAG system functionality
- Local model response quality
- Context relevance and accuracy
- Prompt injection protection
- Response consistency
- Complex query handling

## Customization

### Supported File Types
Currently supported: `.md` (Markdown files only)

To add support for more file types, modify the `load_documents_from_directory()` function in `main.py` or just add more files to the ./docs folder.

### Chunking Strategy
Document chunking can be adjusted in the `setup_rag_system()` function:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Adjust chunk size
    chunk_overlap=200,      # Adjust overlap
    separators=["Q:", "\n", " ", ""]
)
```

### Retrieval Settings
Modify retrieval parameters:
```python
retriever=vectorstore.as_retriever(search_kwargs={"k": 3})  # Number of documents to retrieve
```

## Production Deployment & Scaling

This section provides guidance on scaling and deploying the RAG system to production environments.

### 🚀 **Deployment Options & Tradeoffs**

#### **1. Containerized Deployment (Docker)**
**Pros:**
- ✅ Consistent environment across deployments
- ✅ Easy scaling and orchestration
- ✅ Version control for dependencies

**Cons:**
- ❌ Additional complexity for simple deployments
- ❌ Container overhead

#### **2. Cloud Deployment**
**AWS ECS/Fargate:**
- ✅ Managed container orchestration
- ✅ Auto-scaling capabilities
- ❌ Vendor lock-in

**Google Cloud Run:**
- ✅ Serverless containers
- ✅ Pay-per-use pricing
- ❌ Cold start latency

**Azure Container Instances:**
- ✅ Simple deployment
- ✅ Integration with Azure services
- ❌ Limited scaling options

#### **3. Serverless Deployment**
**Pros:**
- ✅ No infrastructure management
- ✅ Automatic scaling
- ✅ Pay-per-use

**Cons:**
- ❌ Cold start latency
- ❌ Limited execution time
- ❌ Memory constraints

### 📈 **Scaling Strategies & Tradeoffs**

#### **1. Vector Database Scaling**

**ChromaDB (Current):**
```
┌─────────────┐
│   ChromaDB  │ ← Local file-based storage
│   (Local)   │
└─────────────┘
```
**Tradeoffs:**
- ✅ Simple setup, no external dependencies
- ❌ No horizontal scaling, single point of failure
- ❌ Limited concurrent access

**Production Alternatives:**

**Pinecone (Managed):**
```
┌─────────────┐    ┌─────────────┐
│   Client    │───▶│  Pinecone   │ ← Cloud-managed vector DB
│             │    │   (Cloud)   │
└─────────────┘    └─────────────┘
```
**Tradeoffs:**
- ✅ Fully managed, automatic scaling
- ✅ High availability
- ❌ Vendor lock-in, ongoing costs

**Weaviate (Self-hosted):**
```
┌─────────────┐    ┌─────────────┐
│   Client    │───▶│  Weaviate   │ ← Self-hosted vector DB
│             │    │ (Cluster)   │
└─────────────┘    └─────────────┘
```
**Tradeoffs:**
- ✅ Full control, no vendor lock-in
- ✅ Horizontal scaling
- ❌ Operational overhead

#### **2. LLM Scaling**

**Current (Ollama Local):**
```
┌─────────────┐
│   Ollama    │ ← Local LLM inference
│   (Local)   │
└─────────────┘
```
**Tradeoffs:**
- ✅ No API costs, privacy
- ❌ Limited model selection
- ❌ Resource intensive

**Production Alternatives:**

**OpenAI API:**
```
┌─────────────┐    ┌─────────────┐
│   Client    │───▶│   OpenAI    │ ← Cloud LLM API
│             │    │    API      │
└─────────────┘    └─────────────┘
```
**Tradeoffs:**
- ✅ Best model quality, reliability
- ❌ High costs, data privacy concerns

**Anthropic Claude:**
```
┌─────────────┐    ┌─────────────┐
│   Client    │───▶│  Anthropic  │ ← Alternative cloud LLM
│             │    │    API      │
└─────────────┘    └─────────────┘
```
**Tradeoffs:**
- ✅ Good quality, competitive pricing
- ❌ Still cloud-dependent

### 🏗️ **Architecture Patterns**

#### **1. Monolithic Architecture (Current)**
```
┌─────────────────────────────────────┐
│           RAG Application           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│  │  API    │ │  RAG    │ │ Vector  │ │
│  │ Layer   │ │ Engine  │ │ Store   │ │
│  └─────────┘ └─────────┘ └─────────┘ │
└─────────────────────────────────────┘
```
**Tradeoffs:**
- ✅ Simple deployment and debugging
- ❌ Difficult to scale individual components
- ❌ Single point of failure

#### **2. Microservices Architecture**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   API       │───▶│   RAG       │───▶│  Vector     │
│ Gateway     │    │ Service     │    │ Store       │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Load      │    │   LLM       │    │   Document  │
│ Balancer    │    │ Service     │    │   DB        │
└─────────────┘    └─────────────┘    └─────────────┘
```
**Tradeoffs:**
- ✅ Independent scaling, fault isolation
- ❌ Increased complexity, network overhead

#### **3. Event-Driven Architecture**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Document    │───▶│   Kafka     │───▶│  Flink      │
│ Watcher     │    │  (Events)   │    │ Processing  │
└─────────────┘    └─────────────┘    └─────────────┘
                           │                   │
                           ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐
                   │   Vector    │    │   RAG       │
                   │   Store     │    │   Service   │
                   └─────────────┘    └─────────────┘
```
**Tradeoffs:**
- ✅ Loose coupling, real-time processing
- ❌ Complex event management, debugging

### 📊 **Performance Optimization Tradeoffs**

#### **1. Caching Strategies**

**Redis Caching:**
```
┌─────────────┐    ┌─────────────┐
│   Client    │───▶│   Redis     │ ← Cache layer
│             │    │   Cache     │
└─────────────┘    └─────────────┘
       │                   │
       ▼                   ▼
┌─────────────┐    ┌─────────────┐
│   RAG       │    │   Vector    │
│   Service   │    │   Store     │
└─────────────┘    └─────────────┘
```
**Tradeoffs:**
- ✅ Faster response times, reduced load
- ❌ Cache invalidation complexity, memory usage

#### **2. Batch Processing vs Real-time**

**Batch Processing:**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Documents   │───▶│   Batch     │───▶│   Vector    │
│             │    │ Processor   │    │   Store     │
└─────────────┘    └─────────────┘    └─────────────┘
```
**Tradeoffs:**
- ✅ Efficient resource usage, cost-effective
- ❌ Delayed updates, not real-time

**Real-time Processing:**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Documents   │───▶│  Stream     │───▶│   Vector    │
│             │    │ Processor   │    │   Store     │
└─────────────┘    └─────────────┘    └─────────────┘
```
**Tradeoffs:**
- ✅ Immediate updates, real-time
- ❌ Higher resource usage, complexity

### 🔒 **Security & Monitoring Tradeoffs**

#### **1. Authentication Strategies**

**JWT Tokens:**
- ✅ Stateless, scalable
- ❌ Token management complexity

**OAuth2:**
- ✅ Industry standard, secure
- ❌ Complex implementation

#### **2. Monitoring Approaches**

**OpenTelemetry (Recommended):**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   RAG       │───▶│ OpenTelemetry│───▶│   Jaeger    │
│   Service   │    │   Traces    │    │   (Tracing) │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Metrics   │    │ Prometheus  │    │ Grafana     │
│   (OTel)    │    │   (Metrics) │    │ (Dashboard) │
└─────────────┘    └─────────────┘    └─────────────┘
```
**Tradeoffs:**
- ✅ Comprehensive observability, standards-based
- ❌ Setup complexity, learning curve

**Simple Logging:**
- ✅ Easy to implement, familiar
- ❌ Limited insights, no correlation

## Model Context Protocol (MCP) Integration

The Model Context Protocol (MCP) enables AI assistants to access external data sources and tools. This section shows how to convert the RAG system into an MCP server.

### 🔌 **MCP Architecture Overview**

#### **1. MCP Server Architecture**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   AI        │───▶│   MCP       │───▶│   RAG       │
│ Assistant   │    │   Server    │    │   System    │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Tools     │    │   Resources │    │   Vector    │
│ Discovery   │    │ Management  │    │   Store     │
└─────────────┘    └─────────────┘    └─────────────┘
```

#### **2. MCP vs Direct Integration**

**Direct Integration:**
```
┌─────────────┐    ┌─────────────┐
│   AI        │───▶│   RAG       │
│ Assistant   │    │   System    │
└─────────────┘    └─────────────┘
```
**Tradeoffs:**
- ✅ Direct communication, lower latency
- ❌ Tight coupling, vendor lock-in
- ❌ Limited tool discovery

**MCP Integration:**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   AI        │───▶│   MCP       │───▶│   RAG       │
│ Assistant   │    │   Protocol  │    │   System    │
└─────────────┘    └─────────────┘    └─────────────┘
```
**Tradeoffs:**
- ✅ Standardized interface, tool discovery
- ✅ Loose coupling, extensibility
- ❌ Additional protocol overhead

### 🔄 **MCP Features & Tradeoffs**

#### **1. Tool Discovery**
**Benefits:**
- ✅ Automatic tool discovery by AI assistants
- ✅ Self-documenting APIs
- ✅ Dynamic capability exposure

**Tradeoffs:**
- ❌ Additional implementation complexity
- ❌ Protocol overhead for simple use cases

#### **2. Resource Management**
**Benefits:**
- ✅ Access to external data sources
- ✅ Document browsing capabilities
- ✅ Real-time resource updates

**Tradeoffs:**
- ❌ Resource synchronization complexity
- ❌ Access control management

#### **3. Streaming Responses**
**Benefits:**
- ✅ Real-time response streaming
- ✅ Better user experience
- ✅ Progressive disclosure

**Tradeoffs:**
- ❌ Implementation complexity
- ❌ Error handling challenges

### 🚀 **MCP Deployment Options**

#### **1. Local Development**
```
┌─────────────┐    ┌─────────────┐
│   Local     │───▶│   MCP       │
│   AI Tool   │    │   Server    │
└─────────────┘    └─────────────┘
```
**Tradeoffs:**
- ✅ Simple setup, fast development
- ❌ Limited scalability, single user

#### **2. Cloud Deployment**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Cloud     │───▶│   Load      │───▶│   MCP       │
│   AI Tool   │    │ Balancer    │    │   Servers   │
└─────────────┘    └─────────────┘    └─────────────┘
```
**Tradeoffs:**
- ✅ Scalability, high availability
- ❌ Infrastructure complexity, costs

### 📊 **MCP Benefits Summary**

| Feature | Benefit | Tradeoff |
|---------|---------|----------|
| **Standardized Interface** | Consistent API across AI assistants | Protocol overhead |
| **Tool Discovery** | Automatic capability detection | Implementation complexity |
| **Resource Management** | Access to external data sources | Access control complexity |
| **Streaming Support** | Real-time responses | Error handling complexity |
| **Extensibility** | Easy to add new tools | Learning curve |
| **Vendor Independence** | No lock-in to specific AI tools | Additional abstraction layer |
