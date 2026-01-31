"""
Token Estimator - Core Logic
Provides functions for text analysis without UI dependencies.
"""

import PyPDF2
from io import BytesIO

def read_text_file(file) -> str:
    """
    Read plain text file.
    
    Args:
        file: File-like object from Streamlit uploader
        
    Returns:
        Text content as string
    """
    return file.read().decode('utf-8')

def read_pdf_file(file) -> str:
    """
    Extract text from PDF file.
    
    Args:
        file: File-like object from Streamlit uploader
        
    Returns:
        Extracted text as string
    """
    pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_file(file, file_extension: str) -> str:
    """
    Read file based on extension.
    
    Args:
        file: File-like object from Streamlit uploader
        file_extension: File extension (e.g., 'txt', 'pdf')
        
    Returns:
        File content as string
        
    Raises:
        ValueError: If file type is not supported
    """
    if file_extension == 'pdf':
        return read_pdf_file(file)
    elif file_extension in ['txt', 'md', 'py', 'json', 'csv']:
        return read_text_file(file)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in text.
    Rule: 1 token ≈ 4 characters
    
    Args:
        text: Input text string
        
    Returns:
        Estimated token count
    """
    return len(text) // 4

def count_words(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text: Input text string
        
    Returns:
        Word count
    """
    return len(text.split())

def count_characters(text: str) -> int:
    """
    Count total characters in text.
    
    Args:
        text: Input text string
        
    Returns:
        Character count
    """
    return len(text)

def check_limit(char_count: int, limit: int = 4000) -> tuple[bool, str]:
    """
    Check if character count exceeds limit.
    
    Args:
        char_count: Number of characters
        limit: Maximum allowed characters (default: 4000)
        
    Returns:
        Tuple of (is_within_limit, message)
    """
    if char_count > limit:
        return False, f"WARNING: File contains {char_count:,} characters, which exceeds the {limit:,} character limit!"
    return True, f"File is within limits ({char_count:,}/{limit:,} characters)"

def analyze_text(text: str, limit: int = 4000) -> dict:
    """
    Perform complete text analysis.
    
    Args:
        text: Input text string
        limit: Character limit for warning (default: 4000)
        
    Returns:
        Dictionary with analysis results
    """
    char_count = count_characters(text)
    word_count = count_words(text)
    token_count = estimate_tokens(text)
    within_limit, message = check_limit(char_count, limit)
    
    return {
        "characters": char_count,
        "words": word_count,
        "tokens": token_count,
        "within_limit": within_limit,
        "message": message
    }
