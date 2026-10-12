from langchain_core.prompts import ChatPromptTemplate

# Step 2: Prompt build
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer clearly and concisely. "
            "Format: a short direct answer first, then up to 3 bullet points "
            "of supporting detail.",
        ),
        ("human", "{question}"),
    ]
)