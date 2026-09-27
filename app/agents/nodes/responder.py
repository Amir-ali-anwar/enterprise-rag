import logfire
from langchain_groq import ChatGroq

from app.agents.state import AgentState
from app.config import settings

llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model=settings.GROQ_MODEL,
    temperature=0.1,
    max_retries=2,
)


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    """
    query = state["current_query"]
    logfire.info("Generating Response", query=query)

    history_str = ""

    for msg in state['message'][:-1]:
        role = "User" if msg['role'] == "user" else "Assistant"
        history_str += f"{role} : {msg['content']}\n"

    user_msg = state['message'][-1]['content'] if state['message'] else ''

    sources_used = []

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")

        prompt = f"""
        You are StackSage, a professional web development assistant specialising in
        HTML, CSS, JavaScript, React.js, Node.js, MongoDB, Docker, and Kubernetes.

        RULES — follow these strictly:
        1. You may respond to greetings, farewells, and questions about your
           capabilities.
        2. You may answer follow-up questions that reference the CONVERSATION
           HISTORY below (e.g. "what did I just ask?").
        3. You MUST NOT answer off-topic requests such as jokes, poems, trivia,
           recipes, math homework, movie recommendations, weather, sports, or
           any subject outside HTML, CSS, JavaScript, React.js, Node.js,
           MongoDB, Docker, and Kubernetes.
        4. If the user's message is off-topic, respond ONLY with:
           "I'm StackSage, focused on web development: HTML, CSS, JavaScript,
            React, Node.js, MongoDB, Docker, and Kubernetes. I can't help with
            that — but ask me anything technical!"

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        logfire.info("Generating technical RAG response.")

        max_context_chars = 25000

        # Using documents (plural) as defined in AgentState
        documents = state.get("documents") or []

        context_blocks = []
        seen_sources = set()
        total_len = 0

        for i, doc in enumerate(documents, start=1):
            content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
            source = doc.get("source", "unknown") if isinstance(doc, dict) else "unknown"

            block = f"[{i}] SOURCE: {source}\n{content}"
            if total_len + len(block) >= max_context_chars:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

            context_blocks.append(block)
            total_len += len(block)

            if source not in seen_sources:
                seen_sources.add(source)
                sources_used.append((len(sources_used) + 1, source))

        full_context = "\n\n".join(context_blocks)

        prompt = f"""
        You are StackSage, a Senior Full-Stack Web Development Architect.
        Answer the question using the TECHNICAL CONTEXT provided.
        The context is split into numbered blocks like [1], [2] — each tagged with its SOURCE.
        Cite the block numbers inline (e.g. "... autoscaling [1] ...") for claims drawn from the context.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    try:
        with logfire.span("Generating response with Groq LLM"):
            response = llm.invoke(prompt)

        answer = response.content
        if sources_used:
            citations = "\n\n**Sources:**\n" + "\n".join(
                f"[{n}] {source}" for n, source in sources_used
            )
            answer += citations

        return {
            "message": [{"role": "assistant", "content": answer}],
            "status": "Response Generated",
            "final_answer": answer,
        }

    except Exception as e:
        logfire.error("Error generating response with Groq LLM: {error}", error=e)
        return {"message": [{"role": "assistant", "content": "I apologize, but I encountered an error while generating the response. Please try again."}], 
                "status": "Error"
        }
          

        
        