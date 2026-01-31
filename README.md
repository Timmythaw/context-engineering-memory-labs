# Context Engineering & Memory Labs

A comprehensive implementation of context engineering and memory management techniques for AI agents, developed as part of the Frontier Tech Leaders Agentic AI Training Programme.

**Live Demo:** [https://context-engineering-memory-labs.streamlit.app/](https://context-engineering-memory-labs.streamlit.app/)

## Overview

This repository demonstrates three progressive labs that build upon each other to create a production-ready AI agent with document retrieval capabilities:

1. **Token Estimator** - Text analysis and file processing
2. **Simple RAG** - Document retrieval using TF-IDF
3. **LLM Integration** - Complete AI agent with Gemini 2.5 Flash

## Project Structure

```
context-engineering-memory-labs/
├── token_estimator/          # Lab 1: Token counting and file reading
├── simple_rag/               # Lab 2A: Document retrieval system
├── llm-integration/          # Lab 2B: AI agent with memory (DEPLOYED)
├── pyproject.toml           # Project dependencies
└── README.md
```

## Labs

### Lab 1: Token Estimator

A utility for analyzing text files and estimating token counts for LLM context management.

**Features:**
- Multi-format file support (TXT, PDF, MD, PY, JSON, CSV)
- Character, word, and token counting
- Context limit validation
- PDF text extraction using PyPDF2

**Location:** `token_estimator/`

**Run locally:**
```bash
cd token_estimator
streamlit run app.py
```

### Lab 2A: Simple RAG (Retrieval-Augmented Generation)

A document retrieval system using TF-IDF vectorization for semantic search without requiring external dependencies.

**Features:**
- TF-IDF based document vectorization
- Cosine similarity search
- Multi-document support
- Lightweight and fast

**Location:** `simple_rag/`

**Run locally:**
```bash
cd simple_rag
python document_retriever.py  # Run tests
```

### Lab 2B: LLM Integration (DEPLOYED)

A complete AI agent with conversation memory, document retrieval, and real-time process visualization.

**Features:**
- Gemini 2.5 Flash integration via LangChain
- Persistent conversation memory using LangGraph checkpointer
- Document search tool with relevance scoring
- Real-time agent process visualization
- Multi-format document upload
- Chat history management

**Location:** `llm-integration/`

**Live Demo:** [https://context-engineering-memory-labs.streamlit.app/](https://context-engineering-memory-labs.streamlit.app/)

**Run locally:**
```bash
cd llm-integration
streamlit run app.py
```

## Installation

### Prerequisites
- Python 3.13+
- uv (recommended) or pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Timmythaw/context-engineering-memory-labs.git
cd context-engineering-memory-labs
```

2. Install dependencies using uv:
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API key:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

4. Run any lab:
```bash
# Token Estimator
cd token_estimator && streamlit run app.py

# Simple RAG (tests)
cd simple_rag && python document_retriever.py

# LLM Integration
cd llm-integration && streamlit run app.py
```

## Technology Stack

- **LangChain** - Agent framework and tool orchestration
- **LangGraph** - Memory management and checkpointing
- **Gemini 2.5 Flash** - Google's generative AI model
- **Streamlit** - Web application framework
- **scikit-learn** - TF-IDF vectorization
- **PyPDF2** - PDF text extraction
- **python-dotenv** - Environment variable management

## Architecture

The LLM Integration (Lab 2B) follows a modular architecture:

```
User Input
    ↓
Streamlit UI (app.py)
    ↓
AI Agent (agent.py)
    ↓
┌─────────────────┬──────────────────┐
│  Gemini 2.5     │  Document        │
│  Flash LLM      │  Retriever       │
└─────────────────┴──────────────────┘
    ↓
LangGraph Memory Checkpointer
    ↓
Response with Context
```

## Key Concepts Demonstrated

### Context Engineering
- Document chunking and retrieval
- Relevance scoring and ranking
- Context window management

### Memory Management
- Persistent conversation history
- Thread-based memory isolation
- Efficient state checkpointing

### Agent Design Patterns
- Tool use pattern (document search)
- Reasoning and tool selection
- Result synthesis from multiple sources

## Usage Examples

### Upload Documents and Ask Questions

1. Visit the deployed app: [https://context-engineering-memory-labs.streamlit.app/](https://context-engineering-memory-labs.streamlit.app/)
2. Click "Initialize Agent" in the sidebar
3. Upload documents (PDF, TXT, MD, etc.)
4. Ask questions about the documents
5. View the agent's reasoning process in real-time

### Sample Queries

**General Questions:**
```
What can you help me with?
Explain machine learning in simple terms.
```

**Document-Based Questions:**
```
Summarize the uploaded documents.
What are the key points in the research paper?
Compare the findings across all documents.
```

## Development

### Project Dependencies

Core dependencies are managed in `pyproject.toml`:
- langchain
- langchain-google-genai
- langgraph
- streamlit
- scikit-learn
- pypdf2
- python-dotenv

### Running Tests

```bash
# Test document retriever
cd simple_rag
python document_retriever.py

# Test token estimator
cd token_estimator
python -m pytest  # if tests are added
```

## Deployment

The LLM Integration lab is deployed on Streamlit Cloud. To deploy your own instance:

1. Fork this repository
2. Sign up for Streamlit Cloud
3. Connect your GitHub repository
4. Add `GOOGLE_API_KEY` to Streamlit secrets
5. Deploy from the `llm-integration` directory

## Contributing

This is an educational project developed as part of the FTL Agentic AI Training Programme. Contributions, suggestions, and feedback are welcome.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Frontier Tech Leaders Programme
- LangChain and LangGraph teams
- Google Generative AI team

## Author

**Thaw Zin**
- GitHub: [@Timmythaw](https://github.com/Timmythaw)
- Mae Fah Luang University

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note:** Only the LLM Integration (Lab 2B) is currently deployed. Labs 1 and 2A require local setup to run.
