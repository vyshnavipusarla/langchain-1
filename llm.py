from langchain_groq import ChatGroq

MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
]


# Step 3: LLM call (Groq)
def get_llm(api_key: str, model: str, temperature: float, max_tokens: int) -> ChatGroq:
    return ChatGroq(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
    )