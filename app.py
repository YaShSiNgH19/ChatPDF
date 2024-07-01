import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pickle

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Initialize session state for chat history and vector store
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    st.session_state.vector_store = vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    if st.session_state.vector_store is None:
        st.error("Please process a PDF file first.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}, 
        return_only_outputs=True
    )

    st.session_state.chat_history.append({"question": user_question, "answer": response["output_text"]})

def save_chat_history():
    with open('chat_history.pkl', 'wb') as f:
        pickle.dump(st.session_state.chat_history, f)
    st.success("Chat history saved successfully!")

def load_chat_history():
    if os.path.exists('chat_history.pkl'):
        with open('chat_history.pkl', 'rb') as f:
            st.session_state.chat_history = pickle.load(f)
        st.success("Chat history loaded successfully!")

def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Chat with PDF using Gemini💁</h1>", unsafe_allow_html=True)

    st.sidebar.title("Menu")
    pdf_docs = st.sidebar.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    if st.sidebar.button("Submit & Process"):
        if pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.sidebar.success("Done")
        else:
            st.sidebar.error("Please upload at least one PDF file.")

    if st.sidebar.button("Load Previous Chat History"):
        load_chat_history()

    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []

    if st.sidebar.button("Save Chat History"):
        save_chat_history()

    st.sidebar.markdown("---")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question_input")
    
    if user_question:
        user_input(user_question)

    if st.session_state.chat_history:
        st.markdown("<h2 style='color: #4CAF50;'>Chat History</h2>", unsafe_allow_html=True)
        for i, entry in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {entry['question']}")
            st.markdown(f"**A{i+1}:** {entry['answer']}")
            st.markdown("---")

if __name__ == "__main__":
    main()
