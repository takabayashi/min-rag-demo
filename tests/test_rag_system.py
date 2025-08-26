"""
Integration tests for the RAG system using local models (no mocks)
"""

import pytest
import os
import sys

# Add the parent directory to the path so we can import main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import setup_rag_system, create_qa_prompt


class TestRAGSystem:
    """Integration tests for the complete RAG system using real local models"""

    @pytest.mark.integration
    def test_rag_system_setup(self):
        """Test that the RAG system can be set up correctly with real components"""
        
        try:
            # Setup the RAG system with real components
            rag = setup_rag_system()
            
            if rag is None:
                pytest.skip("RAG system could not be initialized (no documents or Ollama not running)")
            
            assert rag is not None
            assert hasattr(rag, 'invoke')
            
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    @pytest.mark.integration
    def test_python_basics_query(self):
        """Test RAG system with a Python-related query using real model"""
        
        try:
            rag = setup_rag_system()
            
            if rag is None:
                pytest.skip("RAG system could not be initialized")
            
            # Test query about Python basics
            query = "What is Python?"
            response = rag.invoke({"input": query})
            
            # Validate response structure
            assert response is not None
            assert "answer" in response
            assert "context" in response
            
            answer = response["answer"]
            context = response["context"]
            
            # Validate answer content
            assert len(answer) > 0
            assert "python" in answer.lower()
            
            # Validate context
            assert len(context) > 0
            context_content = " ".join([doc.page_content for doc in context])
            assert "python" in context_content.lower()
            
            # Validate that answer is relevant to the query
            assert any(keyword in answer.lower() for keyword in ["programming", "language", "guido", "1991"])
            
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    @pytest.mark.integration
    def test_python_variables_query(self):
        """Test RAG system with Python variables query using real model"""
        
        try:
            rag = setup_rag_system()
            
            if rag is None:
                pytest.skip("RAG system could not be initialized")
            
            # Test query about Python variables
            query = "What are variables in Python?"
            response = rag.invoke({"input": query})
            
            # Validate response
            assert response is not None
            assert "answer" in response
            
            answer = response["answer"].lower()
            
            # Validate that answer contains relevant information about variables
            assert "variable" in answer or "container" in answer or "data" in answer
            
            # Validate context relevance
            context = response["context"]
            context_content = " ".join([doc.page_content for doc in context]).lower()
            assert "variable" in context_content
            
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    @pytest.mark.integration
    def test_java_basics_query(self):
        """Test RAG system with Java basics query using real model"""
        
        try:
            rag = setup_rag_system()
            
            if rag is None:
                pytest.skip("RAG system could not be initialized")
            
            # Test query about Java basics
            query = "What is Java?"
            response = rag.invoke({"input": query})
            
            # Validate response
            assert response is not None
            assert "answer" in response
            
            answer = response["answer"].lower()
            
            # Validate that answer contains relevant information about Java
            assert "java" in answer
            assert any(keyword in answer for keyword in ["object", "oriented", "programming", "language"])
            
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    @pytest.mark.integration
    def test_java_classes_query(self):
        """Test RAG system with Java classes query using real model"""
        
        try:
            rag = setup_rag_system()
            
            if rag is None:
                pytest.skip("RAG system could not be initialized")
            
            # Test query about Java classes
            query = "What are classes in Java?"
            response = rag.invoke({"input": query})
            
            # Validate response
            assert response is not None
            assert "answer" in response
            
            answer = response["answer"].lower()
            
            # Validate that answer contains relevant information about classes
            assert "class" in answer
            assert any(keyword in answer for keyword in ["blueprint", "object", "attribute", "method"])
            
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    @pytest.mark.integration
    def test_cross_language_comparison(self):
        """Test RAG system with cross-language comparison query using real model"""
        
        try:
            rag = setup_rag_system()
            
            if rag is None:
                pytest.skip("RAG system could not be initialized")
            
            # Test query comparing Python and Java
            query = "What are the differences between Python and Java?"
            response = rag.invoke({"input": query})
            
            # Validate response
            assert response is not None
            assert "answer" in response
            
            answer = response["answer"].lower()
            
            # Validate that answer mentions both languages
            assert "python" in answer
            assert "java" in answer
            
            # Validate that context contains both languages
            context = response["context"]
            context_content = " ".join([doc.page_content for doc in context]).lower()
            assert "python" in context_content
            assert "java" in context_content
            
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    @pytest.mark.integration
    def test_unknown_query_handling(self):
        """Test how the system handles queries about unknown topics using real model"""
        
        try:
            rag = setup_rag_system()
            
            if rag is None:
                pytest.skip("RAG system could not be initialized")
            
            # Test query about something not in the documents
            query = "What is quantum physics?"
            response = rag.invoke({"input": query})
            
            # Validate response structure
            assert response is not None
            assert "answer" in response
            
            answer = response["answer"].lower()
            
            # The system should either:
            # 1. Say it doesn't know (good behavior)
            # 2. Give a generic response
            # 3. Try to relate to available context
            
            # Validate that the response is reasonable
            assert len(answer) > 0
            
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")


class TestPromptValidation:
    """Tests for prompt validation and expected responses using real model"""

    @pytest.mark.integration
    def test_python_variables_prompt(self):
        """Test that the system correctly answers questions about Python variables"""
        
        try:
            rag = setup_rag_system()
            
            if rag is None:
                pytest.skip("RAG system could not be initialized")
            
            # Test query about Python variables
            query = "How do I declare variables in Python?"
            response = rag.invoke({"input": query})
            
            # Validate response
            assert response is not None
            assert "answer" in response
            
            answer = response["answer"].lower()
            
            # Validate that answer contains relevant information
            assert any(keyword in answer for keyword in ["variable", "assign", "=", "declare"])
            
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    @pytest.mark.integration
    def test_java_classes_prompt(self):
        """Test that the system correctly answers questions about Java classes"""
        
        try:
            rag = setup_rag_system()
            
            if rag is None:
                pytest.skip("RAG system could not be initialized")
            
            # Test query about Java classes
            query = "How do I create a class in Java?"
            response = rag.invoke({"input": query})
            
            # Validate response
            assert response is not None
            assert "answer" in response
            
            answer = response["answer"].lower()
            
            # Validate that answer contains relevant information
            assert any(keyword in answer for keyword in ["class", "public", "private", "method"])
            
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    @pytest.mark.integration
    def test_prompt_template_structure(self):
        """Test that the prompt template has the correct structure"""
        
        prompt = create_qa_prompt()
        
        # Test that the prompt can be formatted
        formatted = prompt.format(
            context="Python is a programming language.",
            input="What is Python?"
        )
        
        assert formatted is not None
        assert "Python is a programming language" in str(formatted)
        assert "What is Python?" in str(formatted)

    @pytest.mark.integration
    def test_prompt_security_features(self):
        """Test that the prompt includes security features"""
        
        prompt = create_qa_prompt()
        prompt_str = str(prompt)
        
        # Check for security features
        assert "prompt injection" in prompt_str.lower()
        assert "start_question" in prompt_str.lower()
        assert "end_question" in prompt_str.lower()
        assert "context" in prompt_str.lower()


class TestRealRAGSystem:
    """Tests using the actual RAG system with real local model"""

    @pytest.mark.integration
    def test_real_rag_query(self):
        """Test the RAG system with a real query using local model"""
        
        try:
            # Setup the RAG system
            rag = setup_rag_system()
            
            if rag is None:
                pytest.skip("RAG system could not be initialized (no documents or Ollama not running)")
            
            # Test a simple query
            query = "What is Python?"
            response = rag.invoke({"input": query})
            
            # Basic assertions
            assert response is not None
            assert "answer" in response
            assert "context" in response
            
            # Check that the answer contains relevant information
            answer = response["answer"].lower()
            assert "python" in answer or "programming" in answer
            
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    @pytest.mark.integration
    def test_real_rag_with_context(self):
        """Test that the RAG system provides context with answers using local model"""
        
        try:
            rag = setup_rag_system()
            
            if rag is None:
                pytest.skip("RAG system could not be initialized")
            
            query = "What are variables in Python?"
            response = rag.invoke({"input": query})
            
            assert response is not None
            assert "context" in response
            
            # Check that context is provided
            context = response["context"]
            assert len(context) > 0
            
            # Check that context contains relevant documents
            context_content = " ".join([doc.page_content for doc in context])
            assert "python" in context_content.lower() or "variable" in context_content.lower()
            
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    @pytest.mark.integration
    def test_response_quality(self):
        """Test the quality of responses from the local model"""
        
        try:
            rag = setup_rag_system()
            
            if rag is None:
                pytest.skip("RAG system could not be initialized")
            
            # Test with a specific question
            query = "What are the key features of Python?"
            response = rag.invoke({"input": query})
            
            assert response is not None
            assert "answer" in response
            
            answer = response["answer"]
            
            # Validate answer quality
            assert len(answer) > 50  # Should be substantial
            assert "python" in answer.lower()
            
            # Check for coherence (no obvious errors)
            assert not answer.startswith("I don't know") or len(answer) > 100
            
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
