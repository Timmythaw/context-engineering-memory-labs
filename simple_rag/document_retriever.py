"""
Simple RAG - Core Logic
Document retrieval using TF-IDF and Cosine Similarity.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from token_estimator.token_estimator import read_file

class DocumentRetriever:
    """
    Simple document retrieval system using TF-IDF and cosine similarity.
    """
    
    def __init__(self):
        """Initialize the retriever with a TF-IDF vectorizer."""
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=1000  # Limit vocabulary size for efficiency
        )
        self.document_vectors = None
        self.documents = []
        self.document_names = []
    
    def add_documents(self, documents: List[str], document_names: List[str]) -> None:
        """
        Add documents to the retriever and vectorize them.
        
        Args:
            documents: List of document text contents
            document_names: List of document names/filenames
        """
        if len(documents) != len(document_names):
            raise ValueError("Number of documents must match number of document names")
        
        self.documents = documents
        self.document_names = document_names
        
        # Fit TF-IDF vectorizer on all documents
        self.document_vectors = self.vectorizer.fit_transform(documents)
    
    def query(self, query_text: str, top_k: int = 1) -> List[Tuple[str, str, float]]:
        """
        Find most relevant documents for a query.
        
        Args:
            query_text: User's search query
            top_k: Number of top results to return (default: 1)
            
        Returns:
            List of tuples: (document_name, document_text, similarity_score)
        """
        if self.document_vectors is None:
            raise ValueError("No documents have been added. Call add_documents first.")
        
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([query_text])
        
        # Calculate cosine similarity between query and all documents
        similarities = cosine_similarity(query_vector, self.document_vectors)[0]
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((
                self.document_names[idx],
                self.documents[idx],
                float(similarities[idx])
            ))
        
        return results
    
    def get_document_count(self) -> int:
        """Return the number of documents loaded."""
        return len(self.documents)
    
    def clear(self) -> None:
        """Clear all documents and reset the retriever."""
        self.documents = []
        self.document_names = []
        self.document_vectors = None


def read_document(file, filename: str) -> Tuple[str, str]:
    """
    Read document content from uploaded file (supports text and PDF).
    Reuses the file reading logic from token_estimator module.
    
    Args:
        file: File-like object from Streamlit uploader
        filename: Name of the file
        
    Returns:
        Tuple of (filename, content)
    """
    # Get file extension
    file_extension = filename.split('.')[-1].lower()
    
    try:
        # Use the read_file function from token_estimator
        content = read_file(file, file_extension)
        return filename, content
    except ValueError as e:
        raise ValueError(f"Error reading {filename}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Could not read {filename}. Error: {str(e)}")
