"""
Simple RAG - Streamlit App
Upload documents and query them using TF-IDF and cosine similarity.
"""

import streamlit as st
from document_retriever import DocumentRetriever, read_document

def main():
    st.title("Simple RAG System")
    st.markdown("Upload documents (text or PDF) and search using TF-IDF and Cosine Similarity")
    
    # Initialize session state for retriever
    if 'retriever' not in st.session_state:
        st.session_state.retriever = DocumentRetriever()
        st.session_state.documents_loaded = False
    
    # Step 1: Document Upload
    st.header("Step 1: Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload multiple documents (text or PDF)",
        type=['txt', 'md', 'py', 'json', 'csv', 'pdf'],  # Added PDF
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            try:
                # Read all documents
                documents = []
                document_names = []
                
                with st.spinner("Processing documents..."):
                    for file in uploaded_files:
                        filename, content = read_document(file, file.name)
                        documents.append(content)
                        document_names.append(filename)
                
                # Add to retriever
                st.session_state.retriever.add_documents(documents, document_names)
                st.session_state.documents_loaded = True
                
                st.success(f"Successfully processed {len(documents)} documents!")
                
                # Show document summary
                with st.expander("Loaded Documents"):
                    for i, (name, doc) in enumerate(zip(document_names, documents)):
                        file_type = name.split('.')[-1].upper()
                        st.markdown(f"**{i+1}. {name}** `({file_type})`")
                        st.text(f"Length: {len(doc)} characters, {len(doc.split())} words")
                        st.code(doc[:200] + "..." if len(doc) > 200 else doc, language='text')
                        st.divider()
                        
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
                return
    
    # Step 2: Query Documents
    if st.session_state.documents_loaded:
        st.header("Step 2: Search Documents")
        
        query = st.text_input(
            "Enter your search query",
            placeholder="Example: I want to find information on Apples"
        )
        
        # Number of results to show
        top_k = st.slider("Number of results to show", min_value=1, max_value=5, value=1)
        
        if st.button("Search", type="primary"):
            if not query:
                st.warning("Please enter a search query")
                return
            
            try:
                with st.spinner("Searching documents..."):
                    results = st.session_state.retriever.query(query, top_k=top_k)
                
                st.success(f"Found {len(results)} relevant document(s)")
                
                # Display results
                for i, (doc_name, doc_content, similarity) in enumerate(results):
                    with st.container():
                        file_type = doc_name.split('.')[-1].upper()
                        st.markdown(f"### Result #{i+1}: {doc_name} `({file_type})`")
                        
                        # Similarity score with color coding
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.progress(similarity)
                        with col2:
                            st.metric("Similarity", f"{similarity:.2%}")
                        
                        # Show document preview
                        with st.expander("View Document Content"):
                            st.text_area(
                                "Content",
                                value=doc_content,
                                height=200,
                                disabled=True,
                                label_visibility="collapsed"
                            )
                        
                        st.divider()
                        
            except Exception as e:
                st.error(f"Error during search: {str(e)}")
    
    else:
        st.info("Upload and process documents to start searching")
    
    # Sidebar with info
    with st.sidebar:
        st.header("How It Works")
        st.markdown("""
        **TF-IDF (Term Frequency-Inverse Document Frequency)**
        - Converts documents into numerical vectors
        - Important words get higher weights
        - Common words (like 'the', 'a') get lower weights
        
        **Cosine Similarity**
        - Measures angle between query and document vectors
        - Score ranges from 0 (not similar) to 1 (identical)
        - Higher score = more relevant document
        
        **Supported Files:**
        - Text: `.txt`, `.md`, `.py`, `.json`, `.csv`
        - Documents: `.pdf`
        
        **Process:**
        1. Upload multiple documents
        2. System vectorizes each document
        3. Enter a search query
        4. Query is vectorized the same way
        5. Calculate similarity between query and each document
        6. Return most relevant documents
        """)
        
        # Show current state
        st.divider()
        st.metric("Documents Loaded", st.session_state.retriever.get_document_count())
        
        if st.button("Clear All Documents"):
            st.session_state.retriever.clear()
            st.session_state.documents_loaded = False
            st.rerun()

if __name__ == "__main__":
    main()
