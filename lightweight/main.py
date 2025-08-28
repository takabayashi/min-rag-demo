"""
Lightweight RAG Implementation using PyTorch and scikit-learn
"""

import os
import json
import sys
import torch
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import re
from datetime import datetime

# Import configuration
from config import *

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# MLX imports (optional)
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class DocumentLoader:
    def __init__(self, docs_dir: str = DOCS_DIR):
        self.docs_dir = docs_dir
    
    def load_documents(self) -> List[Dict]:
        docs_path = Path(self.docs_dir)
        if not docs_path.exists():
            print(f"Directory {self.docs_dir} not found!")
            return []
        
        documents = []
        for file_path in docs_path.glob("**/*.md"):
            try:
                content = file_path.read_text(encoding='utf-8')
                documents.append({
                    'filename': file_path.name,
                    'filepath': str(file_path),
                    'content': content,
                    'size': len(content)
                })
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def chunk_document(self, content: str, filename: str) -> List[Dict]:
        """Chunk document semantically, keeping Q&A pairs together"""
        chunks = []
        
        # Extract headers
        h1_match = re.search(HEADER_PATTERNS['h1'], content, re.MULTILINE)
        h2_match = re.search(HEADER_PATTERNS['h2'], content, re.MULTILINE)
        h3_match = re.search(HEADER_PATTERNS['h3'], content, re.MULTILINE)
        
        h1 = h1_match.group(1) if h1_match else ""
        h2 = h2_match.group(1) if h2_match else ""
        h3 = h3_match.group(1) if h3_match else ""
        headers = [h1, h2, h3]
        
        # Split content into Q&A pairs
        qa_matches = re.findall(QA_PATTERN, content, re.DOTALL)

        # Handle content without Q&A format
        if not qa_matches:
            # Split content into words
            words = content.split()
            
            # Create overlapping chunks
            for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk_words = words[i:i + CHUNK_SIZE]
                chunk_text = ' '.join(chunk_words)
                
                if len(chunk_text.strip()) > MIN_CHUNK_SIZE:
                    chunks.append({
                        'content': chunk_text,
                        'filename': filename,
                        'headers': headers,
                        'chunk_id': len(chunks),
                        'qa_pair_id': None,
                        'word_count': len(chunk_words)
                    })
            
            return chunks
        
        print(f"Found {len(qa_matches)} Q&A pairs")
        
        for i, (question, answer) in enumerate(qa_matches):
            qa_text = f"Q: {question.strip()}\nA: {answer.strip()}"
            
            # Split Q&A into chunks if too long
            words = qa_text.split()
            for j in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk_words = words[j:j + CHUNK_SIZE]
                chunk_text = ' '.join(chunk_words)
                
                if len(chunk_text.strip()) > MIN_CHUNK_SIZE:
                    chunks.append({
                        'content': chunk_text,
                        'filename': filename,
                        'headers': headers,
                        'qa_pair_id': i,
                        'chunk_id': len(chunks),
                        'word_count': len(chunk_words)
                    })
        
        return chunks

class LocalEmbeddings:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.device = torch.device(AVAILABLE_DEVICES)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True, 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings.cpu().numpy().flatten()
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return np.array(embeddings)

class OllamaEmbeddings:
    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        
        print(f"Loading Ollama Embeddings: {model_name}")
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                print(f"Ollama connection failed: {response.status_code}")
        except Exception as e:
            print(f"Ollama not running: {e}")
            print("Please start Ollama with: ollama serve")
    
    def get_embedding(self, text: str) -> np.ndarray:
        try:
            payload = {
                "model": self.model_name,
                "prompt": text
            }
            
            response = requests.post(f"{self.base_url}/api/embeddings", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return np.array(result.get('embedding', []))
            elif response.status_code == 404:
                print(f"Embedding model '{self.model_name}' not found. Please install it with:")
                print(f"  ollama pull {self.model_name}")
                return np.zeros(384)
            else:
                print(f"Ollama embeddings error: {response.status_code}")
                return np.zeros(384)
                
        except Exception as e:
            print(f"Ollama embeddings failed: {e}")
            return np.zeros(384)
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return np.array(embeddings)

class VectorStore:
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.metadata = {}
    
    def add_documents(self, chunks: List[Dict], embeddings: np.ndarray):
        self.chunks = chunks
        self.embeddings = embeddings
        
        for i, chunk in enumerate(chunks):
            self.metadata[i] = {
                'filename': chunk['filename'],
                'headers': chunk['headers'],
                'qa_pair_id': chunk['qa_pair_id'],
                'chunk_id': chunk['chunk_id'],
                'word_count': chunk['word_count']
            }
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = DEFAULT_K) -> List[Tuple[Dict, float]]:
        if self.embeddings is None:
            return []
        
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            similarity = similarities[idx]
            results.append((chunk, similarity))
        
        return results
    
    def save(self, filepath: str):
        """Save vector store using JSON for visibility"""
        data = {
            'chunks': self.chunks,
            'embeddings': self.embeddings.tolist() if self.embeddings is not None else None,
            'metadata': self.metadata,
            'created_at': datetime.now().isoformat(),
            'version': '1.0',
            'stats': {
                'total_chunks': len(self.chunks),
                'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else None,
                'files_processed': len(set([chunk['filename'] for chunk in self.chunks]))
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Vector store saved to {filepath}")
    
    def load(self, filepath: str):
        """Load vector store using JSON"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.chunks = data['chunks']
            self.embeddings = np.array(data['embeddings']) if data['embeddings'] else None
            self.metadata = data['metadata']
            
            # Print stats if available
            if 'stats' in data:
                stats = data['stats']
                print(f"Loaded vector store: {stats.get('total_chunks', 'N/A')} chunks, "
                      f"{stats.get('files_processed', 'N/A')} files")
            
        except FileNotFoundError:
            print(f"{MESSAGES['vector_store_not_found'].format(filepath=filepath)}")
            raise
        except Exception as e:
            print(f"{MESSAGES['vector_store_error'].format(error=e)}")
            raise

class LocalLLM:
    def __init__(self, model_name: str = LLM_MODEL):
        # Optimize device detection
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")
        
        # Optimize model loading
        torch_dtype = torch.float16 if USE_FP16 and self.device.type in ["cuda", "mps"] else torch.float32
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Move model to device if not using device_map
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        print(f"Local LLM loaded: {model_name}")
        
    def generate(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = DEFAULT_TEMPERATURE) -> str:
        try:
            # Limit input length
            if len(prompt) > MAX_INPUT_LENGTH:
                prompt = prompt[:MAX_INPUT_LENGTH]
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=MAX_INPUT_LENGTH,
                truncation=True,
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=LLM_TOP_P,
                    top_k=LLM_TOP_K,
                    repetition_penalty=LLM_REPETITION_PENALTY,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode the full response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response
            response = extract_response(full_response, prompt)
            
            return response if response else "I don't have enough information to answer that question."
            
        except Exception as e:
            print(f"LLM generation failed: {e}")
            if "temperature" in str(e).lower():
                print("Hint: Temperature must be > 0.0. Try setting DEFAULT_TEMPERATURE to 0.1 or higher.")
            return MESSAGES['llm_fallback']

class MLXLLM:
    """MLX-based LLM for Apple Silicon optimization"""
    
    def __init__(self, model_name: str = MLX_MODEL):
        if not MLX_AVAILABLE:
            raise ImportError("MLX not available. Install with: pip install mlx mlx-lm")
        
        print(f"Loading MLX LLM: {model_name}")
        self.model, self.tokenizer = load(model_name)
        print(f"MLX LLM loaded: {model_name}")
    
    def generate(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = DEFAULT_TEMPERATURE) -> str:
        try:
            # Limit input length
            if len(prompt) > MAX_INPUT_LENGTH:
                prompt = prompt[:MAX_INPUT_LENGTH]
            
            # Generate response using MLX
            response = generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=LLM_TOP_P,
                top_k=LLM_TOP_K,
                repetition_penalty=LLM_REPETITION_PENALTY
            )
            
            # Extract response
            response = extract_response(response, prompt)
            
            return response if response else "I don't have enough information to answer that question."
            
        except Exception as e:
            print(f"MLX LLM generation failed: {e}")
            return MESSAGES['llm_fallback']

class OllamaLLM:
    def __init__(self, model_name: str = "qwen2.5:7b"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        
        print(f"Loading Ollama LLM: {model_name}")
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                print(f"Ollama connection failed: {response.status_code}")
        except Exception as e:
            print(f"Ollama not running: {e}")
            print("Please start Ollama with: ollama serve")
    
    def generate(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = DEFAULT_TEMPERATURE) -> str:
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": LLM_TOP_P,
                    "top_k": LLM_TOP_K,
                    "repeat_penalty": LLM_REPETITION_PENALTY
                }
            }
            
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            elif response.status_code == 404:
                print(f"Model '{self.model_name}' not found. Please install it with:")
                print(f"  ollama pull {self.model_name}")
                return MESSAGES['llm_fallback']
            else:
                print(f"Ollama API error: {response.status_code}")
                return MESSAGES['llm_fallback']
                
        except Exception as e:
            print(f"Ollama generation failed: {e}")
            return MESSAGES['llm_fallback']

class LightweightRAG:
    def __init__(self, vector_store: VectorStore, embeddings_model: LocalEmbeddings, llm: LocalLLM = None):
        self.vector_store = vector_store
        self.embeddings_model = embeddings_model
        self.llm = llm or LocalLLM()
    
    def query(self, question: str, k: int = DEFAULT_K) -> Dict:
        # Get relevant chunks
        query_embedding = self.embeddings_model.get_embedding(question)
        relevant_chunks = self.vector_store.similarity_search(query_embedding, k=k)
        
        if not relevant_chunks:
            return {
                'question': question,
                'answer': MESSAGES['no_context'],
                'context': '',
                'sources': [],
                'similarities': []
            }
        
        # Create context and generate response
        context = self._create_context(relevant_chunks)
        
        # Choose prompt template based on configuration
        if PROMPT_TEMPLATE_TYPE == "strict":
            prompt = RAG_PROMPT_TEMPLATE_STRICT.format(context=context, question=question)
        else:
            prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        
        try:
            response = self.llm.generate(prompt)
        except Exception as e:
            print(f"{MESSAGES['llm_generation_failed'].format(error=e)}")
            response = self._fallback_response(relevant_chunks)
        
        return {
            'question': question,
            'answer': response,
            'context': context,
            'sources': [chunk['filename'] for chunk, _ in relevant_chunks],
            'similarities': [sim for _, sim in relevant_chunks]
        }
    
    def _create_context(self, relevant_chunks: List[Tuple[Dict, float]]) -> str:
        context_parts = []
        
        for chunk, similarity in relevant_chunks:
            chunk_text = f"[Source: {chunk['filename']} | Similarity: {similarity:.3f}]\n{chunk['content']}\n"
            context_parts.append(chunk_text)
            
            if len('\n'.join(context_parts)) > MAX_CONTEXT_LENGTH:
                break
        
        return "\n".join(context_parts)
    
    def _fallback_response(self, relevant_chunks: List[Tuple[Dict, float]]) -> str:
        if not relevant_chunks:
            return MESSAGES['no_relevant_info']
        
        # Use the best chunk
        best_chunk = relevant_chunks[0][0]
        return f"Based on the available information: {best_chunk['content'][:200]}..."
    
    def interactive_query(self):
        print("\nRAG System Ready! Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                question = input("\nEnter your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                result = self.query(question)
                
                print(f"\nAnswer: {result['answer']}")
                print(f"Sources:")
                for i, (source, similarity) in enumerate(zip(result['sources'], result['similarities']), 1):
                    print(f"{i}. {source} (similarity: {similarity:.3f})")
                
                print("\n" + "="*50)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def batch_query(self, questions: List[str]) -> List[Dict]:
        """Process a list of questions and return results"""
        results = []
        
        print(f"Processing {len(questions)} questions...")
        print("=" * 50)
        
        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i}/{len(questions)}: {question}")
            print("-" * 40)
            
            try:
                result = self.query(question)
                results.append(result)
                
                print(f"Answer: {result['answer']}")
                if result['sources']:
                    print(f"Sources: {', '.join(result['sources'])}")
                else:
                    print("No sources found")
                
            except Exception as e:
                print(f"Error processing question: {e}")
                results.append({
                    'question': question,
                    'answer': f"Error: {e}",
                    'context': '',
                    'sources': [],
                    'similarities': []
                })
        
        return results

def load_or_create_vector_store():
    """Try to load existing vector store, create new one if not found"""
    if Path(VECTOR_STORE_PATH).exists():
        print(f"Loading existing vector store...")
        try:
            vector_store = VectorStore()
            vector_store.load(VECTOR_STORE_PATH)
            print(f"Loaded {len(vector_store.chunks)} chunks")
            return vector_store
        except Exception as e:
            print(f"Error loading vector store: {e}")
    
    return create_new_vector_store()

def create_new_vector_store():
    """Create new vector store from documents"""
    # Load and chunk documents
    document_loader = DocumentLoader()
    documents = document_loader.load_documents()
    
    if not documents:
        print(MESSAGES['no_documents'])
        return None
    
    # Create chunks
    print("Processing documents...")
    all_chunks = []
    for doc in documents:
        chunks = document_loader.chunk_document(doc['content'], doc['filename'])
        all_chunks.extend(chunks)
    
    print(f"Created {len(all_chunks)} chunks")
    
    # Generate embeddings
    if USE_OLLAMA:
        embeddings_model = OllamaEmbeddings(OLLAMA_EMBEDDING_MODEL)
    else:
        embeddings_model = LocalEmbeddings()
    
    print("Generating embeddings...")
    chunk_texts = [chunk['content'] for chunk in all_chunks]
    chunk_embeddings = embeddings_model.get_embeddings_batch(chunk_texts)
    
    print(f"Generated {len(chunk_embeddings)} embeddings")
    
    # Create and save vector store
    print("Creating vector store...")
    vector_store = VectorStore()
    vector_store.add_documents(all_chunks, chunk_embeddings)
    
    print(f"Vector store created with {len(vector_store.chunks)} documents")
    
    print("Saving vector store...")
    vector_store.save(VECTOR_STORE_PATH)
    
    return vector_store

def initialize_rag_system(vector_store):
    """Initialize RAG system with vector store"""
    if USE_OLLAMA:
        llm = OllamaLLM(OLLAMA_LLM_MODEL)
        embeddings_model = OllamaEmbeddings(OLLAMA_EMBEDDING_MODEL)
    elif USE_MLX and MLX_AVAILABLE:
        llm = MLXLLM(MLX_MODEL)
        embeddings_model = LocalEmbeddings()
    else:
        llm = LocalLLM(model_name=LLM_MODEL)
        embeddings_model = LocalEmbeddings()
    
    rag_system = LightweightRAG(vector_store, embeddings_model, llm)
    
    return rag_system

def extract_response(full_response: str, prompt: str) -> str:
    """Extract response from model output"""
    # Look for assistant part and extract after it
    if "assistant" in full_response.lower():
        parts = full_response.split("assistant")
        if len(parts) > 1:
            response = parts[-1].strip()
            # Remove end markers
            for marker in ["<|im_end|>", "human:", "user:"]:
                if marker in response.lower():
                    response = response.split(marker)[0].strip()
            return response
    
    # Fallback: extract after prompt
    if full_response.startswith(prompt):
        return full_response[len(prompt):].strip()
    
    return full_response.strip()

def main():
    print("Starting Lightweight RAG System...")
    print("=" * 50)
    
    # Load or create vector store
    vector_store = load_or_create_vector_store()
    if not vector_store:
        return
    
    # Initialize RAG system
    rag_system = initialize_rag_system(vector_store)
    
    # Check if sample questions should be run
    
    if len(sys.argv) > 1 and sys.argv[1] == '--sample':
        if SAMPLE_QUESTIONS:
            rag_system.batch_query(SAMPLE_QUESTIONS)
        else:
            print("No sample questions found in config")
    else:
        # Start interactive mode
        rag_system.interactive_query()


if __name__ == "__main__":
    main()
