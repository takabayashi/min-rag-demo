# Run the RAG system with sample questions
test:
	python main.py --sample

chunks:
	cat lightweight_vector_store.json | jq '.stats'
	cat lightweight_vector_store.json | jq '.chunks[0]'
	cat lightweight_vector_store.json | jq '.chunks[1]'

# Run the RAG system in interactive mode
run:
	python main.py

# Install Ollama models
install-ollama-models:
	ollama pull qwen2.5:7b
	ollama pull nomic-embed-text

# Start Ollama service
start-ollama:
	ollama serve

# Clean generated files
clean:
	rm -f lightweight_vector_store.json
	rm -f *.pkl

# Show help
help:
	@echo "Available commands:"
	@echo "  make test              - Run RAG system with sample questions"
	@echo "  make run               - Run RAG system in interactive mode"
	@echo "  make chunks            - Show chunks in vector store"
	@echo "  make install-ollama-models - Install required Ollama models"
	@echo "  make start-ollama      - Start Ollama service"
	@echo "  make clean             - Remove generated files"
	@echo "  make help              - Show this help"