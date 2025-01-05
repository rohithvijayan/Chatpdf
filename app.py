import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
# Extract text from uploaded PDFs
def get_pdf_text(docfiles):
    text = ""
    for pdf in docfiles:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Split text into manageable chunks
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

# Create a vector store from text chunks
def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store

# Build a RetrievalQA chain
def build_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Top 5 relevant chunks
    llm = ChatOpenAI(temperature=0)  # Use OpenAI's ChatGPT model
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain

# Function to render styled chat messages
def render_message_box(message, message_type):
    if message_type == "user":
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; margin: 5px;">
                <div style="
                    max-width: 70%;
                    background-color: #DCF8C6;
                    color: black;
                    padding: 10px;
                    border-radius: 20px 20px 0 20px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                ">
                    {message}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-start; margin: 5px;">
                <div style="
                    max-width: 70%;
                    background-color: #EAEAEA;
                    color: black;
                    padding: 10px;
                    border-radius: 20px 20px 20px 0;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                ">
                    {message}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
# Function to handle input changes
def handle_input_change():
    st.session_state.pending_input = st.session_state.user_query
# Main Streamlit app
def main():
    OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")  # Load environment variables (for OpenAI API key)

    st.set_page_config(page_title="ChatPDF", page_icon=":books:", layout="centered")
    st.title("RAG ChatBot Demo")
    st.text("Chat with your uploaded documents")

    # Initialize session state
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_input" not in st.session_state:
        st.session_state.pending_input = ""  # Holds the current user query

    with st.sidebar:
        st.subheader("RAG CHAT BOT DEMO Settings")
        doc_files = st.file_uploader(
            label="Upload Your Documents", accept_multiple_files=True, type=["pdf"]
        )
        if st.button(label="Process"):
            with st.spinner("Processing..."):
                # Extract text from PDFs
                raw_text = get_pdf_text(doc_files)
                if not raw_text.strip():
                    st.error("No text could be extracted from the uploaded PDFs.")
                    return

                # Split text into chunks
                chunks = get_text_chunks(raw_text)

                # Create vector store
                vector_store = get_vectorstore(chunks)
                st.success("Documents processed successfully!")

                # Build QA chain
                st.session_state.qa_chain = build_qa_chain(vector_store)

    # Chat interface
    if st.session_state.qa_chain:
        st.subheader("Chat Interface")

        # Display chat history with styled message boxes
        for message in st.session_state.chat_history:
            render_message_box(message["content"], message["type"])

        # Input box for user query (always at the bottom)
        user_query = st.text_input(
            "Ask your question:",
            placeholder="Type your query here...",
            key="user_query",
            value=st.session_state.pending_input,
            on_change=handle_input_change,  # Call the handler when the input changes
        )

        if st.button("Send"):
            # Check if the input is not empty
            if st.session_state.pending_input.strip():
                # Add user message to chat history
                user_query = st.session_state.pending_input
                st.session_state.chat_history.append({"type": "user", "content": user_query})

                # Get response from QA chain
                with st.spinner("Fetching answer..."):
                    response = st.session_state.qa_chain({"query": user_query})
                    answer = response["result"]

                    # Add system reply to chat history
                    st.session_state.chat_history.append({"type": "system", "content": answer})

                # Clear the pending input and trigger rerender
                st.session_state.pending_input = ""


if __name__ == "__main__":
    main()
