"""
Simple RAG Module
Document retrieval using TF-IDF and Cosine Similarity.
"""

from .document_retriever import DocumentRetriever, read_document

__all__ = ['DocumentRetriever', 'read_document']
