"""
Unit tests for the main RAG system
"""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import os

# Add the parent directory to the path so we can import main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import split_documents, load_documents
from langchain.schema import Document


class TestMainFunctions(unittest.TestCase):
    """Test cases for main functions"""

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
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        
        # Check that all items are Document objects
        for doc in result:
            self.assertIsInstance(doc, Document)
            self.assertIn("source", doc.metadata)
            self.assertIn("filename", doc.metadata)
            self.assertIn("type", doc.metadata)
            self.assertEqual(doc.metadata["source"], "test.md")
            self.assertEqual(doc.metadata["filename"], "test.md")
            self.assertEqual(doc.metadata["type"], "md")
        
        # Check that content is preserved
        all_content = " ".join([doc.page_content for doc in result])
        self.assertIn("Title 1", all_content)
        self.assertIn("Title 2", all_content)
        self.assertIn("Subtitle 1", all_content)
        self.assertIn("Subtitle 2", all_content)

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
        self.assertEqual(result, [])

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
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        
        # Check that content is preserved
        all_content = " ".join([doc.page_content for doc in result])
        self.assertIn("plain text", all_content)


if __name__ == '__main__':
    unittest.main()
