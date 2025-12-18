import os
import streamlit as st
from dotenv import load_dotenv

# -------------------- LOAD ENV --------------------
load_dotenv()

# -------------------- LANGCHAIN IMPORTS --------------------
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="💬",
    layout="wide"
)

# -------------------- HEADER --------------------
st.title("💬 PDF Chat Assistant")
st.markdown(
    "Chat continuously with your PDF.\n\n"
    "✔ Ask multiple questions\n"
    "✔ Context-aware answers\n"
    "✔ Stable (No Torch / No GPU)"
)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    st.markdown("---")
    if os.getenv("GROQ_API_KEY"):
        st.success("Groq API Key Loaded")
    else:
        st.error("Groq API Key Missing")

    st.markdown("---")
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []

# -------------------- SAFETY PROMPT --------------------
prompt_template = """
You are a helpful assistant.
Answer ONLY using the provided document context.
If the answer is not in the document, say:
"I cannot find the answer in this document."

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# -------------------- LLM --------------------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.1
)

# -------------------- EMBEDDINGS (STABLE) --------------------
embeddings = FakeEmbeddings(size=384)

# -------------------- SESSION STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# -------------------- PDF PROCESSING --------------------
if uploaded_file and st.session_state.qa_chain is None:
    with st.spinner("Indexing PDF..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=False,
            chain_type_kwargs={"prompt": PROMPT}
        )

    st.success("PDF ready for chat!")

# -------------------- DISPLAY CHAT HISTORY --------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------- CHAT INPUT --------------------
if st.session_state.qa_chain:
    user_input = st.chat_input("Ask something about the PDF...")

    if user_input:
        # Show user message
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain.invoke(
                {"query": user_input}
            )
            answer = result["result"]

        # Show assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        with st.chat_message("assistant"):
            st.markdown(answer)

else:
    st.info("👈 Upload a PDF to start chatting.")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("PDF Chat Assistant | Streamlit + LangChain + Groq")
