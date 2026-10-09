import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Assistant")
st.write(
    "Upload a PDF and ask questions about its contents."
)


@st.cache_resource
def load_embedding_model():
    return OpenAIEmbeddings()


@st.cache_resource
def load_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b"
    )


embedding_model = load_embedding_model()
llm = load_llm()


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI assistant.

            Use ONLY the provided context to answer the question.

            If the answer is not present in the context,
            say:

            "I could not find the answer in the document."

            Do not use outside knowledge.
            """
        ),
        (
            "human",
            """
            Context:
            {context}

            Question:
            {question}
            """
        )
    ]
)

st.sidebar.header("📄 Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


if uploaded_file is not None:

    if (
        "file_name" not in st.session_state
        or st.session_state.file_name != uploaded_file.name
    ):

        with st.spinner("Processing PDF..."):

            temp_dir = tempfile.mkdtemp()

            pdf_path = os.path.join(
                temp_dir,
                uploaded_file.name
            )

            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model
            )

            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 4,
                    "fetch_k": 10,
                    "lambda_mult": 0.5
                }
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = retriever
            st.session_state.file_name = uploaded_file.name

            st.session_state.messages = []

            shutil.rmtree(temp_dir, ignore_errors=True)

        st.success(
            f"PDF processed successfully: {uploaded_file.name}"
        )


if "retriever" not in st.session_state:

    st.info(
        "👈 Upload a PDF from the sidebar to start asking questions."
    )

else:

    st.subheader(
        f"💬 Ask questions about: {st.session_state.file_name}"
    )

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # User input
    query = st.chat_input(
        "Ask something about the PDF..."
    )


    if query:

        # Display user message
        with st.chat_message("user"):
            st.markdown(query)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )


        # Retrieve documents
        with st.spinner("Searching the document..."):

            docs = st.session_state.retriever.invoke(query)

            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in docs
                ]
            )


        # Create prompt
        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": query
            }
        )


        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("Generating answer..."):

                response = llm.invoke(
                    final_prompt
                )

                answer = response.content

                st.markdown(answer)


        # Save answer
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )