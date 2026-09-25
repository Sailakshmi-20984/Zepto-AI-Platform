import os
from typing import List, TypedDict
from pydantic import BaseModel, Field
import chromadb
from chromadb.utils import embedding_functions
from langgraph.graph import StateGraph, END

# Pydantic Schema
class QueryResponse(BaseModel):
    answer: str = Field(description="Generated answer")
    sources: List[str] = Field(default=[], description="Source document IDs")
    confidence: float = Field(default=1.0, description="Confidence score between 0 and 1")

# State definition
class AgentState(TypedDict):
    query: str
    intent: str
    sources: List[str]
    response: QueryResponse

# MOCK_LLM Toggle (Default 1 / unset = Mock Mode)
MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"

MODULE_DIR = os.path.dirname(__file__)
DB_DIR = os.path.join(MODULE_DIR, "chroma_db")

# Structured Prompt Template Text
PROMPT_TEMPLATE = """
Role: You are an official Zepto Customer Support AI assistant.
Context: {context}
Task: Answer the customer's query using only the provided context.
Negative Constraint: Do not answer using information not present in the provided context. If unsure, state that you do not know.
Format: Return JSON with keys "answer", "sources", and "confidence".
Length: Under 100 words.

Example:
Query: What is the delivery fee?
Context: Standard delivery is free over INR 149, else INR 25.
Answer: {{ "answer": "Standard delivery is free on orders over INR 149. Below INR 149, a flat fee of INR 25 applies.", "sources": ["doc_01.txt"], "confidence": 1.0 }}

Customer Query: {query}
"""

def get_chroma_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return client.get_or_create_collection(name="zepto_policies", embedding_function=embed_fn)

# Node 1: Classify Intent
def classify_intent(state: AgentState) -> AgentState:
    query = state["query"].lower()
    keywords = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]
    
    if MOCK_LLM:
        intent = "policy_question" if any(k in query for k in keywords) else "general_question"
    else:
        # LLM classification path
        intent = "policy_question" if any(k in query for k in keywords) else "general_question"
        
    state["intent"] = intent
    return state

# Node 2: Retrieve & Answer
def retrieve_and_answer(state: AgentState) -> AgentState:
    collection = get_chroma_collection()
    results = collection.query(query_texts=[state["query"]], n_results=3)
    
    docs = results["documents"][0] if results["documents"] else []
    doc_ids = results["ids"][0] if results["ids"] else []
    
    if MOCK_LLM:
        top_snippet = docs[0][:200] if docs else "No relevant policy found."
        answer_str = f"Based on the retrieved context: {top_snippet}"
        state["response"] = QueryResponse(answer=answer_str, sources=doc_ids, confidence=1.0)
    else:
        # MOCK_LLM=0 real LLM call placeholder / retry logic
        answer_str = f"Based on the retrieved context: {docs[0][:200]}"
        state["response"] = QueryResponse(answer=answer_str, sources=doc_ids, confidence=0.95)
        
    return state

# Node 3: Direct Answer
def direct_answer(state: AgentState) -> AgentState:
    answer_str = "I can only answer questions about Zepto policies right now."
    state["response"] = QueryResponse(answer=answer_str, sources=[], confidence=1.0)
    return state

# Router
def route_intent(state: AgentState) -> str:
    return state["intent"]

# LangGraph Build
workflow = StateGraph(AgentState)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.set_entry_point("classify_intent")
workflow.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer"
    }
)
workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

app_graph = workflow.compile()