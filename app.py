import os

import streamlit as st
from dotenv import load_dotenv

from chain import build_chain
from llm import MODELS

load_dotenv()
api_key = os.getenv("GROQ_API_KEY", "")

st.set_page_config(page_title="My First LLM Chain", page_icon="🔗")
st.title("🔗 Your First LLM Chain")
st.caption("PromptTemplate → ChatGroq → StrOutputParser, composed with LCEL")

# Sidebar: model configuration
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", MODELS)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Max tokens", 64, 4096, 512, 64)

# Step 1: User input
with st.form("qa_form"):
    question = st.text_area("Ask a question", placeholder="e.g. What is LCEL in LangChain?")
    submitted = st.form_submit_button("Submit")

# Step 4: Display (streamed)
if submitted:
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it to your .env file and restart the app.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            chain = build_chain(api_key, model_name, temperature, max_tokens)
            st.write_stream(chain.stream({"question": question}))
        except Exception as e:
            st.error(f"Something went wrong: {e}")