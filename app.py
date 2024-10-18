import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores  import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings

#extracting the  text from pdf and return entire doc text as a string
def get_pdf_text(docfiles):
    text=""
    for pdf in docfiles:
        reader=PdfReader(pdf)
        for page in reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(raw_text):
        text_splitter=CharacterTextSplitter(separator='\n',chunk_size=1000,chunk_overlap=200,length_function=len)
        chunks=text_splitter.split_text(raw_text)
        return chunks
   
def get_vectorstore(chunks):
    embeddings=OpenAIEmbeddings()
    vector_store=FAISS.from_texts(texts=chunks,embedding=embeddings)
    return vector_store
        
def main():
    load_dotenv()    
    st.set_page_config(page_title="ChatPDF",page_icon=":books:",layout="centered")
    st.title("ChatPDF")
    st.text_input(label="Chat With Your Uploaded Documen !",value="Ask your Question:")
    with st.sidebar:
        st.subheader("Chat PDF Settings")
        doc_files=st.file_uploader(label="Upload Your Documents",accept_multiple_files=True)
        if(st.button(label="Process")):
            with st.spinner("Processing"):    
                #get pdf text
                raw_text=get_pdf_text(doc_files)
                #st.write(raw_text)
                #get text chunks
                chunks=get_text_chunks(raw_text)
                #st.write(chunks)
                #vector store
                vector_store=get_vectorstore(chunks)

if __name__=="__main__":
    main()