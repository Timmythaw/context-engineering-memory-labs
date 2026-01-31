"""
Lab 2B: AI Model with Gemini 2.5 Flash
Streamlit app showing real agent process: thinking, tool calling, and responses
"""

import streamlit as st
import sys
import os
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import from token_estimator for file reading
sys.path.append(str(Path(__file__).parent.parent))
from token_estimator.token_estimator import read_file

from agent import AIAgent

def show_agent_process(agent, user_message):
    """
    Stream agent process and show thinking, tool calls, and final response.
    This shows the REAL agent process, not hardcoded steps.
    """
    config = {"configurable": {"thread_id": agent.thread_id}}
    
    # Container for streaming updates
    process_container = st.container()
    
    with process_container:
        st.markdown("###Agent Process")
        
        try:
            # Stream the agent's execution
            events = agent.agent.stream(
                {"messages": [HumanMessage(content=user_message)]},
                config=config,
                stream_mode="values"
            )
            
            final_response = None
            step_count = 0
            
            # Process each event from the stream
            for event in events:
                if "messages" in event:
                    messages = event["messages"]
                    
                    # Get the last message
                    if messages:
                        last_msg = messages[-1]
                        
                        # Show different types of messages
                        if isinstance(last_msg, AIMessage):
                            step_count += 1
                            
                            # Check if it's a tool call
                            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                                with st.expander(f"Step {step_count}: Tool Selection", expanded=True):
                                    for tool_call in last_msg.tool_calls:
                                        st.markdown(f"**Agent decided to use:** `{tool_call.get('name', 'unknown')}`")
                                        st.code(f"Query: {tool_call.get('args', {}).get('query', 'N/A')}", language="text")
                            
                            # Check if it has content (final response)
                            elif last_msg.content:
                                final_response = last_msg.content
                        
                        elif isinstance(last_msg, ToolMessage):
                            step_count += 1
                            with st.expander(f"Step {step_count}: Tool Execution Result", expanded=True):
                                # Show truncated result - ensure content is string
                                content_str = str(last_msg.content) if not isinstance(last_msg.content, str) else last_msg.content
                                result_preview = content_str[:300] + "..." if len(content_str) > 300 else content_str
                                st.markdown(result_preview)
            
            # Show final response
            if final_response:
                st.divider()
                st.markdown("### Final Response")
                return final_response
            else:
                return "Sorry, I couldn't generate a response."
                
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            # Fallback to simple invoke
            response = agent.chat(user_message)
            return response

def main():
    st.set_page_config(page_title="AI Agent with Gemini", layout="wide")
    
    st.title("AI Agent with Gemini 2.5 Flash")
    st.markdown("*Real-time agent process visualization: See thinking, tool calls, and responses*")
    
    # Sidebar: Configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Show current configuration
        st.info("**Model:** gemini-2.5-flash\n\n")
        
        # Initialize agent
        if st.button("Initialize Agent", type="primary", use_container_width=True):
            # Get API key from environment
            api_key = os.getenv("GOOGLE_API_KEY")
            
            if not api_key:
                st.error("GOOGLE_API_KEY not found in .env file")
                st.info("Add `GOOGLE_API_KEY=your_key_here` to your .env file")
                return
            
            with st.spinner("Initializing AI Agent..."):
                try:
                    agent = AIAgent(api_key)
                    st.session_state.agent = agent
                    st.session_state.agent_initialized = True
                    st.session_state.messages = []
                    
                    st.success("Agent ready!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Stats (if agent initialized)
        if 'agent_initialized' in st.session_state and st.session_state.agent_initialized:
            st.divider()
            st.header("Stats")
            
            doc_count = st.session_state.agent.get_document_count()
            conv_length = st.session_state.agent.get_conversation_length()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Docs", doc_count)
            with col2:
                st.metric("Messages", conv_length)
            
            if st.button("Clear Conversation", use_container_width=True):
                st.session_state.agent.clear_conversation()
                st.session_state.messages = []
                st.success("Conversation cleared!")
                st.rerun()
        
        st.divider()
        
        # Info
        with st.expander("About This Lab"):
            st.markdown("""
            ### What You'll See
            
            - **Agent's reasoning** (real-time)
            - **Document searches** (tool calls)
            - **Tool results** (search findings)
            - **Final synthesized answer**
            
            ### Technologies
            
            - **LangChain**: Agent framework
            - **LangGraph**: Memory & checkpointer
            - **Gemini 2.5 Flash**: Google's AI
            - **TF-IDF**: Document retrieval
            """)
    
    # Main area
    if 'agent_initialized' not in st.session_state or not st.session_state.agent_initialized:
        st.info("Click 'Initialize Agent' to get started")
        
        st.markdown("""
        ### Quick Start
        
        1. **Click "Initialize Agent"** in sidebar
        2. **Upload documents** (optional)
        3. **Start chatting!**
        
        ---
        
        ### Powered by Gemini 2.5 Flash
        - Ultra-fast responses
        - Large context window
        - Advanced tool calling
        - Persistent memory via LangGraph
        """)
        return
    
    # Document Upload
    st.header("Step 1: Upload Documents (Optional)")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload documents for the agent to reference",
            type=['txt', 'md', 'py', 'json', 'csv'],
            accept_multiple_files=True,
            key="file_uploader"
        )
    
    with col2:
        if uploaded_files:
            if st.button("Process", type="primary"):
                try:
                    documents = []
                    document_names = []
                    
                    with st.spinner("Processing..."):
                        for file in uploaded_files:
                            file_extension = file.name.split('.')[-1].lower()
                            content = read_file(file, file_extension)
                            documents.append(content)
                            document_names.append(file.name)
                    
                    st.session_state.agent.load_documents(documents, document_names)
                    st.success(f"Loaded {len(documents)} documents!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Show loaded documents
    doc_count = st.session_state.agent.get_document_count()
    if doc_count > 0:
        st.success(f"{doc_count} document(s) loaded and ready for search")
    
    st.divider()
    
    # Chat Interface
    st.header("Step 2: Chat with AI Agent")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Show agent process and get response
        with st.chat_message("assistant"):
            # Show the REAL agent process (thinking, tool calls, etc.)
            response = show_agent_process(st.session_state.agent, prompt)
            
            # Display final response prominently
            st.markdown(response)
        
        # Store AI response
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Example prompts (if no conversation yet)
    if len(st.session_state.messages) == 0:
        st.markdown("### Try these examples:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**General Questions:**")
            st.code("What can you help me with?", language="text")
            st.code("Explain quantum computing", language="text")
        
        with col2:
            if doc_count > 0:
                st.markdown("**Document Questions:**")
                st.code("Summarize the documents", language="text")
                st.code("What are the key points?", language="text")
            else:
                st.markdown("**Tip:**")
                st.info("Upload documents to see the agent search through them!")

if __name__ == "__main__":
    main()
