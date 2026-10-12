# 🔗 Your First LLM Chain (LangChain + Groq + Streamlit)

A question-answering assistant with a live streaming response, configurable temperature, max tokens, and model selection. It is built from the three primitives that sit underneath almost every LangChain application: **PromptTemplate → LLM → OutputParser**, joined together with the **LCEL pipe operator (`|`)**.

 

---

## 1. What this project does

1. The user types a question in a Streamlit text box and presses **Submit**.
2. The question is inserted into a structured prompt (system message plus user message).
3. The prompt is sent to a Groq-hosted LLM.
4. The answer streams back token by token and is displayed live in the browser.

---

## 2. Project structure

```
project/
├── app.py             # Streamlit UI (input, settings, streamed output)
├── chain.py           # Builds the LCEL chain: prompt | llm | parser
├── llm.py             # Creates the Groq chat model + list of models
├── prompt.py          # The prompt template
├── requirements.txt   # Python dependencies
├── .env               # Your secret API key (NOT committed)
└── .gitignore         # Keeps .env and venv out of Git
```

**Why split into files?** Each file has one job (separation of concerns). You can change the prompt without touching the UI, swap the model provider by editing only `llm.py`, and reuse `chain.py` in a different front end (CLI, FastAPI, a notebook).

Data flow between files:

```
app.py ──calls──▶ chain.build_chain()
                      │
                      ├── imports prompt  from prompt.py
                      └── calls   get_llm() from llm.py
                      
          returns:  prompt | llm | StrOutputParser()
app.py ──streams──▶ chain.stream({"question": ...}) ──▶ st.write_stream
```

---

## 3. Architecture: the 4 steps

| Step | Name | What happens | Where in code |
|------|------|--------------|---------------|
| 1 | User Input | User types a question and clicks Submit | `app.py` (`st.form`) |
| 2 | Prompt Build | Question is filled into a structured prompt | `prompt.py` |
| 3 | LLM Call | Prompt goes to Groq; response is streamed back through the chain | `llm.py`, `chain.py` |
| 4 | Display | Tokens are rendered live as they arrive | `app.py` (`st.write_stream`) |

---

## 4. Core concepts explained

### 4.1 LangChain

LangChain is a Python framework for building applications around LLMs. Instead of manually writing HTTP requests, formatting strings and parsing JSON for every provider, LangChain gives you standard building blocks that all share the same interface. That means swapping OpenAI for Groq (or Ollama, Anthropic, etc.) usually changes one or two lines.

Packages used here:
- `langchain-core`: the base abstractions (prompts, parsers, runnables, LCEL).
- `langchain-groq`: the integration that connects LangChain to the Groq API (`ChatGroq`).

### 4.2 The three primitives

Every LangChain app, simple or complex, is built from these three pieces.

**1) PromptTemplate (`prompt.py`)**
A reusable prompt with placeholders. Instead of building strings by hand with f-strings, you define the structure once and fill in variables at runtime.

```python
ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant..."),
    ("human", "{question}"),
])
```

- `{question}` is a **template variable**. It is replaced when you call `chain.stream({"question": "..."})`.
- Using a template keeps the prompt in one place, makes it testable, and prevents messy string concatenation.

**2) LLM / Chat Model (`llm.py`)**
The model that generates the answer. Here it is `ChatGroq`, which talks to Groq's API. Groq runs open models (like Llama) on custom hardware, which makes responses very fast.

**3) OutputParser (`chain.py`)**
A chat model returns an `AIMessage` object (content plus metadata). `StrOutputParser` extracts just the text string so the rest of your code (and Streamlit) gets a plain string instead of a message object.

### 4.3 Chat messages and roles

Chat models don't take one big string. They take a list of messages, each with a **role**:

| Role | Meaning | Purpose |
|------|---------|---------|
| `system` | Instructions for the model | Sets behavior, tone, and output format |
| `human` (user) | What the user said | The actual question |
| `ai` (assistant) | The model's reply | Used for conversation history |

Our system message defines the behavior ("short direct answer, then up to 3 bullets"), and the human message carries the user's question.

### 4.4 LCEL: LangChain Expression Language and the pipe operator (`|`)

LCEL is LangChain's way of composing components into a pipeline:

```python
chain = prompt | llm | StrOutputParser()
```

Read it like a Unix pipe: *the output of the left side becomes the input of the right side.*

```
{"question": "What is RAG?"}
        │
        ▼
   prompt           → turns the dict into a list of messages
        │
        ▼
    llm (Groq)      → generates an AIMessage
        │
        ▼
 StrOutputParser    → extracts the text string
        │
        ▼
  "RAG is ..."
```

**Why it works:** every component is a **Runnable**, an object with a standard set of methods. Because they all share the same interface, the `|` operator can chain any of them. You get these for free on the finished chain:

| Method | Behavior |
|--------|----------|
| `.invoke(input)` | Run once, return the full result |
| `.stream(input)` | Yield the result piece by piece as it's generated |
| `.batch([inputs])` | Run many inputs in parallel |
| `.ainvoke / .astream` | Async versions |

This project uses `.stream()`.

### 4.4.1 Why LCEL instead of plain function calls?

- Streaming, batching and async support come built in.
- Components are swappable (change the model or parser without rewriting the chain).
- It's the foundation for bigger things: RAG, agents, and LangGraph all build on Runnables.

### 4.5 Model parameters

Configured in the sidebar and passed to `ChatGroq` in `llm.py`.

**`model` (model name)**
Which LLM to use. Bigger models are generally smarter but slower; smaller ones are faster and cheaper.
- `llama-3.3-70b-versatile`: large, high quality.
- `llama-3.1-8b-instant`: small, very fast.
- `openai/gpt-oss-20b`: an open-weight model hosted on Groq.

Groq updates its model lineup over time. If a model name errors, check [console.groq.com](https://console.groq.com) for current names.

**`temperature` (0.0 to 1.5 in this app)**
Controls randomness when the model picks the next token.
- **Low (0.0–0.3):** focused and consistent. Good for facts, extraction, code.
- **Medium (0.5–0.8):** balanced.
- **High (1.0+):** more creative and varied, but more likely to wander or be wrong.

**`max_tokens`**
The maximum number of tokens the model may generate in its answer. A *token* is a chunk of text (roughly ¾ of an English word). It caps response length and cost. If the limit is too low, answers get cut off mid-sentence.

### 4.6 Streaming

Without streaming, the user stares at a spinner until the entire answer is ready. With streaming, the model sends tokens as it generates them, and the UI shows them immediately, so it feels instant.

- `ChatGroq(..., streaming=True)` enables streaming on the model.
- `chain.stream({...})` returns a **generator**, a Python object that produces values one at a time. Each value is a small chunk of text.
- Because `StrOutputParser` is the last step, each chunk is a plain string.

```python
for chunk in chain.stream({"question": "Hi"}):
    print(chunk, end="")
```

### 4.7 Streamlit

Streamlit turns a Python script into a web app with no HTML/JS.

Key ideas used here:

- **Rerun model:** every time the user interacts with a widget, Streamlit re-executes the entire script from top to bottom. That's why the chain is rebuilt on each submit, and why widgets keep their values (Streamlit remembers widget state between reruns).
- **`st.sidebar`:** a side panel, used for settings.
- **`st.form` + `st.form_submit_button`:** groups inputs so nothing runs until Submit is pressed. Without a form, every keystroke would trigger a rerun.
- **`st.selectbox` / `st.slider`:** dropdown and slider widgets for model, temperature and max_tokens.
- **`st.write_stream(generator)`:** takes a generator of text chunks and renders them live, like a typing effect. It also returns the final complete text.
- **`st.error` / `st.warning`:** show red and yellow message boxes for validation and failures.

### 4.8 Environment variables and API key security

API keys are secrets. Anyone with your key can use your quota.

- The key lives in a **`.env`** file: `GROQ_API_KEY=...`
- `python-dotenv`'s `load_dotenv()` reads that file and puts the values into environment variables.
- `os.getenv("GROQ_API_KEY")` reads it in code.
- The key is **never** shown in the UI and **never** hard-coded in source files.
- `.gitignore` lists `.env` so it isn't pushed to GitHub.
- `.env.example` is a safe template with a placeholder value that you *can* commit.

If you ever accidentally commit a real key, treat it as leaked: revoke it in the Groq console and create a new one.

### 4.9 Error handling

`app.py` validates before calling the model:
1. API key missing → clear error message.
2. Empty question → warning.
3. Anything else (network, invalid model, rate limit) is caught by `try/except` and shown with `st.error` instead of crashing the app with a traceback.

---

## 5. Setup and run (Windows)

```powershell
# 1. Go to your project folder
cd C:\Users\user\Desktop\first-llm-chain

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your .env file from the template
copy .env.example .env
# then open .env and replace your_key_here with your real Groq key

# 5. Run the app
streamlit run app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com). Restart Streamlit after editing `.env`, because it's only read at startup.

---

## 6. Try it yourself (experiments)

- Set temperature to `0` and ask the same question 3 times. The answers should be nearly identical. Then set it to `1.2` and repeat.
- Set max_tokens to `50` and watch the answer get cut off.
- Compare `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` on a tricky reasoning question.
- Edit the system message in `prompt.py` (for example "Answer like a pirate" or "Answer in Telugu") and see how behavior changes without touching any other file.

---

## 7. Troubleshooting

| Problem | Likely cause / fix |
|---------|--------------------|
| `GROQ_API_KEY not found` | `.env` is missing, in the wrong folder, or Streamlit wasn't restarted |
| `ModuleNotFoundError` | venv not activated or `pip install -r requirements.txt` not run |
| `model_not_found` / 404 error | Model name was retired. Update the list in `llm.py` |
| `401 Unauthorized` | Invalid or revoked API key |
| `429 Too Many Requests` | Free-tier rate limit hit. Wait a bit and retry |
| Answer cut off mid-sentence | `max_tokens` too low |

---


## 8. Glossary

| Term | Meaning |
|------|---------|
| **LLM** | Large Language Model, a neural network that predicts the next token |
| **Token** | A chunk of text (word or word piece) that the model reads and writes |
| **Prompt** | The input text or messages sent to the model |
| **Runnable** | LangChain's standard component interface (`invoke`, `stream`, `batch`) |
| **LCEL** | LangChain Expression Language, composing Runnables with `\|` |
| **Chain** | A pipeline of Runnables, here `prompt \| llm \| parser` |
| **Generator** | A Python object that yields values one at a time (used for streaming) |
| **Temperature** | A randomness setting for token selection |
| **Environment variable** | A value stored outside code, used for secrets and configuration |
