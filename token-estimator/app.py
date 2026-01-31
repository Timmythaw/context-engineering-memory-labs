"""
Token Estimator - Streamlit App
UI for uploading files and displaying text analysis.
"""

import streamlit as st
from token_estimator import read_file, analyze_text

def main():
    st.title("Token Estimator")
    st.markdown("Upload a text file or PDF to estimate tokens and word count")
    
    # File uploader - now includes PDF
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['txt', 'md', 'py', 'json', 'csv', 'pdf']
    )
    
    if uploaded_file is not None:
        # Get file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Read file content based on type
        try:
            text = read_file(uploaded_file, file_extension)
        except UnicodeDecodeError:
            st.error("Error: Could not decode file. Please upload a valid text file.")
            return
        except ValueError as e:
            st.error(f"Error: {str(e)}")
            return
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return
        
        # Analyze text using core logic
        results = analyze_text(text, limit=4000)
        
        # Display results
        st.success(f"File processed successfully! ({file_extension.upper()})")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Characters", f"{results['characters']:,}")
        
        with col2:
            st.metric("Words", f"{results['words']:,}")
        
        with col3:
            st.metric("Estimated Tokens", f"{results['tokens']:,}")
        
        # Display limit check message
        if results['within_limit']:
            st.info(results['message'])
        else:
            st.warning(results['message'])
        
        # Show preview
        with st.expander("File Preview (First 500 characters)"):
            st.code(text[:500], language='text')
        
        # Additional info
        with st.expander("About Token Estimation"):
            st.markdown("""
            **Token Estimation Rule**: 1 token ≈ 4 characters
            
            This is a simplified approximation. Real tokenizers (like GPT's) 
            use more sophisticated methods:
            - Common words = 1 token
            - Rare words may be split into multiple tokens
            - Numbers and special characters vary
            
            **Supported File Types:**
            - Text files: `.txt`, `.md`, `.py`, `.json`, `.csv`
            - Documents: `.pdf`
            
            For production use, consider libraries like `tiktoken` for accurate counting.
            """)

if __name__ == "__main__":
    main()
