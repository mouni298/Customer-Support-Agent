from IPython.display import Image, display
from typing import Annotated, Any, Dict, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.tools import Tool
from langchain_google_vertexai import VertexAIEmbeddings
import time
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from typing import List
import os
from database_tools import DatabaseTool

load_dotenv()
web_search_tool = TavilySearchResults(max_results=2)

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str]
    db_results: Optional[List[Dict[Any, Any]]] 


model = init_chat_model("gemini-2.0-flash", model_provider="google_vertexai")

# After initializing other tools like web_search_tool
db_uri = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
db_tool = DatabaseTool(db_uri)

# Create LangChain tools for database operations
db_query_tool = Tool(
    name="database_query",
    description="Execute SQL queries to get information about orders and products",
    func=db_tool.query_database
)

db_schema_tool = Tool(
    name="database_schema",
    description="Get database schema information for a specific table",
    func=db_tool.get_table_schema
)


 # Create an instance of the StateGraph, passing in the State class
graph_builder = StateGraph(GraphState)


loader = UnstructuredWordDocumentLoader("FAQs.docx")

data = loader.load()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(data)
print(f"length of document chunks generated :{len(doc_splits)}")

# Initialize the a specific Embeddings Model version
embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

vectorstore = Chroma.from_documents(documents=doc_splits,
                                    embedding=embeddings,
                                    collection_name="local-rag")

retriever = vectorstore.as_retriever(search_kwargs={"k":2})

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context"]
)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain

rag_chain = prompt | model | StrOutputParser()




prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
    user question to a vectorstore or web search. Use the vectorstore for questions on Customer Frequently Asked Questions (FAQs) on Orders & Payments,
    Shippings & Delivery, Offers & Membership,Seller & Product Queries,Account & Customer Support, Returns & Refunds. You do not need to be stringent with the keywords 
    in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
    or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
    no premable or explaination. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)
start = time.time()
question_router = prompt | model | JsonOutputParser()



prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)
retrieval_grader = prompt | model | JsonOutputParser()




def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
#
def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    db_results = state.get("db_results", None)
    print(f"Documents: {documents}")
    print(f"DB Results: {db_results}")
    print(f"Question: {question}")
    context = ""
    if documents:
        context += f"\nDocument context: {documents}"
    if db_results:
        context += f"\nDatabase results: {db_results}"
    
    # RAG generation
    generation = rag_chain.invoke({"context": context, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
#
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score['score']
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def route_question(state):
    """
    Route question to web search or RAG or Orders or Products.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    
    # Update prompt to include database routing
    router_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert at routing user questions. Route the question to the appropriate source:
        - Use database for questions about specific orders or product details
        - Use vectorstore for FAQ questions about Orders & Payments, Shipping & Delivery, 
          Offers & Membership, Seller & Product Queries, Account & Customer Support, Returns & Refunds
        - Use web_search for general questions
        
        Return only: {{"datasource": "database"|"vectorstore"|"web_search"}}
        
        Question: {question}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"]
    )
    
    router_chain = router_prompt | model | JsonOutputParser()
    source = router_chain.invoke({"question": question})
    
    if source['datasource'] == 'database':
        return "database"
    elif source['datasource'] == 'web_search':
        return "websearch"
    else:
        return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    

def query_database(state):
    """
    Query database for order or product information

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updated state with database query results
    """
    print("---QUERYING DATABASE---")
    question = state["question"]
    
    # First get schema information to help with query generation
    schema_info = db_schema_tool.run("orders, products")
    print(f"Schema information: {schema_info}")
    
    # Add prompt to convert question to SQL using schema information
    sql_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        Convert the following question into a  VALID SQL query with right syntax like 'SELECT * FROM Orders;'.
        Available tables and their schemas:
        {schema}
        
        IMPORTANT: Return ONLY the raw SQL query.
        DO NOT include any markdown, separators, or decorators.
        BAD: ```sql SELECT * FROM Orders; ```
        GOOD: SELECT * FROM Orders;
        Question: {question}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "schema"]
    )
    
    sql_chain = sql_prompt | model | StrOutputParser()
    query = sql_chain.invoke({
        "question": question,
        "schema": schema_info
    })
    query = query.replace("```sql", "").replace("```", "").replace("<|file_separator|>","").strip()
    print(f"Generated SQL query: {query}")
    # Execute query using the db_query_tool
    results = db_query_tool.run(query)
    print(f"Query results: {results}")
    
    return {
        "question": question,
        "documents": state.get("documents", []),
        "db_results": results
    }


def web_search_method(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}



graph_builder.add_node("websearch", web_search_method) # web search
graph_builder.add_node("retrieve", retrieve) # retrieve
graph_builder.add_node("grade_documents", grade_documents)
graph_builder.add_node("generate", generate) # generatae
graph_builder.add_node("database", query_database)


graph_builder.set_conditional_entry_point(
    route_question,
    {
        "database": "database",
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)

graph_builder.add_edge("retrieve", "generate")
graph_builder.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
graph_builder.add_edge("websearch", "generate")
graph_builder.add_edge("database", "generate")

graph = graph_builder.compile()

user_input = input("ask a question: ")

if user_input:
    # Add user message to chat histor
    response = graph.invoke({
            "question": user_input
        })
    print(response["generation"])


        # Add assistant response to chat history

        