"""
AI Agent with Document Retrieval
Uses LangChain agent API built on LangGraph with proper memory checkpointer
"""

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from typing import List
import sys
from pathlib import Path
import uuid

# Import document retriever from Lab 2A
sys.path.append(str(Path(__file__).parent.parent))
from simple_rag.document_retriever import DocumentRetriever

class AIAgent:
    """
    AI Agent with conversation memory and document retrieval.
    Built on modern LangChain agent API with LangGraph checkpointer.
    """
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", temperature: float = 0.7):
        """
        Initialize AI agent.
        
        Args:
            api_key: Google API key
            model: Gemini model name
            temperature: Temperature for response generation
        """
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature
        )
        
        # Initialize document retriever
        self.document_retriever = DocumentRetriever()
        
        # Initialize LangGraph memory checkpointer
        self.memory = MemorySaver()
        
        # Create unique thread ID for this conversation
        self.thread_id = str(uuid.uuid4())
        
        # Create agent with memory
        self.agent = self._create_agent()
    
    def _document_search_tool(self, query: str) -> str:
        """Search uploaded documents with visual feedback."""
        if self.document_retriever.get_document_count() == 0:
            return "No documents uploaded."
        
        try:
            results = self.document_retriever.query(query, top_k=2)
            
            response = "**Document Search Results**\n\n"
            for i, (doc_name, content, similarity) in enumerate(results, 1):
                response += f"**{doc_name}**\n"
                response += f"Relevance: {similarity:.1%}\n\n"
                response += f"{content}\n\n"
                response += "---\n\n"
            
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _create_agent(self):
        """
        Create agent with document search tool and LangGraph memory.
        """
        # Define tools
        tools = [self._document_search_tool]
        
        # Create agent with LangGraph memory checkpointer
        agent = create_agent(
            model=self.llm,
            tools=tools,
            checkpointer=self.memory,
            system_prompt="""You are a helpful AI assistant with access to document search.

            When users ask questions:
            1. If the question relates to uploaded documents, use the document_search_tool
            2. Synthesize information from multiple sources if needed
            3. Always cite which documents you're referencing
            4. Be conversational and maintain context across the conversation
            5. If documents don't contain the answer, use your general knowledge but clarify the source

            You have persistent memory of our entire conversation via LangGraph's checkpointer.
            """
        )
        
        return agent
    
    def load_documents(self, documents: List[str], document_names: List[str]):
        """
        Load documents into the retriever.
        
        Args:
            documents: List of document contents
            document_names: List of document names
        """
        self.document_retriever.add_documents(documents, document_names)
    
    def chat(self, user_message: str) -> str:
        """
        Send a message to the agent and get response.
        Memory is persisted via LangGraph checkpointer.
        
        Args:
            user_message: User's message
            
        Returns:
            Agent's response
        """
        try:
            # User Message
            response = self.agent.invoke(
                {"messages": [HumanMessage(content=user_message)]},
                config={"configurable": {"thread_id": self.thread_id}}
            )
            
            # Extract AI response (AIMessage)
            if response and "messages" in response:
                for msg in reversed(response["messages"]):
                    # Check if message is from AI
                    if isinstance(msg, AIMessage):
                        # msg.content can be str or list, ensure we return str
                        if isinstance(msg.content, str):
                            return msg.content
                        else:
                            # Convert list/dict content to string representation
                            return str(msg.content)
            
            return "Sorry, I couldn't generate a response."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_document_count(self) -> int:
        """Get number of loaded documents."""
        return self.document_retriever.get_document_count()
    
    def get_conversation_history(self) -> List:
        """
        Get conversation history from LangGraph checkpointer.
        
        Returns:
            List of messages from memory
        """
        try:
            checkpoint = self.memory.get({"configurable": {"thread_id": self.thread_id}})
            
            if checkpoint and "channel_values" in checkpoint:
                messages = checkpoint["channel_values"].get("messages", [])
                return messages
            return []
        except Exception as e:
            # Return empty list on error to maintain return type
            return []
    
    def get_conversation_length(self) -> int:
        """Get number of messages in conversation."""
        return len(self.get_conversation_history())
    
    def clear_conversation(self):
        """Clear conversation history by creating new thread."""
        self.thread_id = str(uuid.uuid4())
