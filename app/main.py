import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Response
import logfire
from pydantic import BaseModel

from app.agents.graph import rag_agent
from app.graudrails import guard

load_dotenv()

logfire.configure(send_to_logfire=os.environ.get("LOGFIRE_SEND_TO"), service_name="rag-enterprise")

app = FastAPI(title="StackSage API")


class QueryRequest(BaseModel):
    query: str
    thread_id: Optional[str] = "default user"


@app.get("/")
def home():
    return {"message": "StackSage API is running"}


@app.get("/graph")
def get_graph():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        logfire.error("Error getting graph", error=e)
        return {"error": f"Could not generate graph image: {e}"}


@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    query_text = request.query
    thread_id = request.thread_id
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "message": [{"role": "user", "content": query_text}],
        "current_query": query_text,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
    }
    try:

        logfire.info("🛡️ Checking query with guardrails")
        rail_fired, rail_response = guard(query_text)

        if rail_fired:
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            return {
                "question": query_text,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": [],
            }

        logfire.info("✅ Query passed guardrails, invoking LangGraph RAG pipeline")
        final_output = rag_agent.invoke(initial_state, config=config)
        return {
            "question": query_text,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", []),
        }
    except Exception as e:
        logfire.error(f"❌ Backend Execution Failed: {e}")
        return {
            "question": query_text,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": [],
        }
