import streamlit as st
from dotenv import load_dotenv

def main():
    load_dotenv()    
    st.set_page_config(page_title="ChatPDF",page_icon=":books:",layout="centered")
    st.title("ChatPDF")
    st.text_input(label="Chat With Your Uploaded Documen !",value="Ask your Question:")
    with st.sidebar:
        st.subheader("Chat PDF Settings")
        doc_files=st.file_uploader(label="Upload Your Documents",accept_multiple_files=True)
        st.write("Press 'Download' to download the chat as a PDF.")

if __name__=="__main__":
    main()