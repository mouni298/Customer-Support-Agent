# Customer Support Agent

An intelligent customer support system powered by LangChain and Google's Gemini LLM that combines RAG (Retrieval Augmented Generation) with web search capabilities to provide accurate responses to customer queries.

## Features

- **Hybrid Search System**: Integrates both local document search (Vector DB) and web search capabilities
- **Intelligent Query Routing**: Automatically routes queries between local FAQs and web search
- **Document Relevance Grading**: Ensures high-quality responses through automated relevance assessment
- **RAG Implementation**: Uses Retrieval Augmented Generation for accurate and contextual responses
- **Database Integration**: Supports SQL database queries for structured data retrieval

## Tech Stack

- **LLM**: Google Vertex AI (Gemini-2.0-flash)
- **Vector Database**: Chroma
- **Embeddings**: Google Vertex AI (text-embedding-004)
- **Framework**: LangChain
- **Web Search**: Tavily API
- **Graph Processing**: LangGraph

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Customer-Support-Agent.git
cd Customer-Support-Agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```env
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
TAVILY_API_KEY=your_tavily_api_key
```

4. Prepare your FAQ document:
- Place your FAQ document (FAQs.docx) in the project root directory
- Format it according to the template provided

## Usage

Run the application:
```bash
python app.py
```

Example query:
```python
for event in graph.stream({"question": "How do I track my order?"}):
    for value in event.values():
        print(value["generation"])
```

