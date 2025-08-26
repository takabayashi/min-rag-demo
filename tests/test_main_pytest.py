"""
Unit tests for the main RAG system using pytest
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path so we can import main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import split_documents, create_qa_prompt
from langchain.schema import Document


class TestSplitDocuments:
    """Test cases for split_documents function"""

    def test_split_documents_basic(self):
        """Test that split_documents correctly splits a simple markdown text"""
        
        # Test input
        test_text = """# Title 1
This is some content for title 1.

## Subtitle 1
This is content for subtitle 1.

## Subtitle 2
This is content for subtitle 2.

# Title 2
This is content for title 2.
"""
        
        custom_metadata = {
            "source": "test.md",
            "filename": "test.md",
            "type": "md"
        }
        
        # Call the function
        result = split_documents(test_text, custom_metadata)
        
        # Assertions
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Check that all items are Document objects
        for doc in result:
            assert isinstance(doc, Document)
            assert "source" in doc.metadata
            assert "filename" in doc.metadata
            assert "type" in doc.metadata
            assert doc.metadata["source"] == "test.md"
            assert doc.metadata["filename"] == "test.md"
            assert doc.metadata["type"] == "md"
        
        # Check that content is preserved
        all_content = " ".join([doc.page_content for doc in result])
        assert "Title 1" in all_content
        assert "Title 2" in all_content
        assert "Subtitle 1" in all_content
        assert "Subtitle 2" in all_content

    def test_split_documents_empty_text(self):
        """Test that split_documents handles empty text gracefully"""
        
        test_text = ""
        custom_metadata = {
            "source": "empty.md",
            "filename": "empty.md",
            "type": "md"
        }
        
        result = split_documents(test_text, custom_metadata)
        
        # Should return empty list for empty text
        assert result == []

    def test_split_documents_no_headers(self):
        """Test that split_documents handles text without headers"""
        
        test_text = "This is just plain text without any headers."
        custom_metadata = {
            "source": "plain.md",
            "filename": "plain.md",
            "type": "md"
        }
        
        result = split_documents(test_text, custom_metadata)
        
        # Should still return at least one document
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Check that content is preserved
        all_content = " ".join([doc.page_content for doc in result])
        assert "plain text" in all_content

    def test_split_documents_large_text(self):
        """Test that split_documents handles large text appropriately"""
        
        # Create a large text that should be split into multiple chunks
        large_text = "# Large Document\n\n" + "This is a very long paragraph. " * 100
        
        custom_metadata = {
            "source": "large.md",
            "filename": "large.md",
            "type": "md"
        }
        
        result = split_documents(large_text, custom_metadata)
        
        # Should return multiple documents for large text
        assert isinstance(result, list)
        assert len(result) > 1  # Should be split into multiple chunks
        
        # Check that all content is preserved
        all_content = " ".join([doc.page_content for doc in result])
        assert "Large Document" in all_content
        assert "very long paragraph" in all_content


class TestQAPrompt:
    """Test cases for create_qa_prompt function"""

    def test_create_qa_prompt_returns_template(self):
        """Test that create_qa_prompt returns a valid prompt template"""
        
        result = create_qa_prompt()
        
        # Should return a ChatPromptTemplate
        assert result is not None
        assert hasattr(result, 'messages')
        
        # Check that it has the expected structure
        messages = result.messages
        assert len(messages) >= 2  # Should have at least system and human messages
        
        # Check that the template can be formatted
        try:
            formatted = result.format(context="test context", input="test input")
            assert formatted is not None
        except Exception as e:
            pytest.fail(f"Template formatting failed: {e}")

    def test_create_qa_prompt_security_features(self):
        """Test that the prompt includes security features"""
        
        result = create_qa_prompt()
        
        # Convert the template to string to check content
        template_str = str(result)
        
        # Check for security features
        assert "prompt injection" in template_str.lower()
        assert "start_question" in template_str.lower()
        assert "end_question" in template_str.lower()
        assert "context" in template_str.lower()


class TestMockExample:
    """Example of how to test functions that depend on external services"""

    @patch('main.OllamaEmbeddings')
    @patch('main.Chroma.from_documents')
    def test_create_vectorstore_with_mock(self, mock_chroma, mock_embeddings):
        """Example test showing how to mock external dependencies"""
        
        # This is just an example - you would need to import the function
        # from main import create_vectorstore
        
        # Mock the embeddings
        mock_emb_instance = MagicMock()
        mock_embeddings.return_value = mock_emb_instance
        
        # Mock the Chroma.from_documents
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
        
        # This test demonstrates the pattern for testing functions
        # that depend on external services like Ollama or ChromaDB
        assert mock_embeddings.called is False
        assert mock_chroma.called is False
        
        # In a real test, you would call the function here:
        # result = create_vectorstore(mock_documents)
        # assert result == mock_vectorstore
