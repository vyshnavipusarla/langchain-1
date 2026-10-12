from langchain_core.output_parsers import StrOutputParser

from llm import get_llm
from prompt import prompt


# LCEL pipe: PromptTemplate | LLM | OutputParser
def build_chain(api_key: str, model: str, temperature: float, max_tokens: int):
    llm = get_llm(api_key, model, temperature, max_tokens)
    return prompt | llm | StrOutputParser()