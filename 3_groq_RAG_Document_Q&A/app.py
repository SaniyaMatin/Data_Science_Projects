import streamlit as st
import os
from langchain_groq import ChatGroq
#from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatMessagePromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
#from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
load_dotenv()
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# set up Streamlit
st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload PDFs and chat with their document content")

#Input the Groq API Key
api_key=st.text_input("Enter your Groq API Key:",type="password")

#Check if Groq API Key is provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")

    #Chat interface
    session_id=st.text_input("Session ID", value="default_session")
    # successfully manage chat history

    if 'store' not in st.session_state:
        st.session_state.store={}

    uploaded_files=st.file_uploader("Choose a PDF File",type="pdf",accept_multiple_files=True)
    # Process uploaded files
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name
            
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)
    # Split and create embeddings for document
        text_splitter= RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        splits=text_splitter.split_documents(documents)
        vectorestore=Chroma.from_documents(documents=splits, embedding= HuggingFaceEmbeddings())
        retreiver=vectorestore.as_retriever()

        contextualize_q_system_prompt=(
            "Given a chat history and latest user question"
            "Which might reference context in the chat history,"
            "formulate a standalone question which can be understood"
            "without the chat history, do not answer the question,"
            "just formulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )

        history_aware_retriever=create_history_aware_retriever(llm,retreiver,contextualize_q_prompt)
        #Answer question

        # Answer Question
        system_promt=(
            "You are an assistant for question-answering tasks."
            "Use the following pieces of retreived context to answer"
            "the question. If you dont know the answer say that you"
            "don't know. Use three sentences maximum and keep the"
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_promt=ChatPromptTemplate.from_messages(
            [
                ("system",system_promt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )

        question_answer_chain=create_stuff_documents_chain(llm,qa_promt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input= st.text_input("Your Question:")
        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                }, #constructs a key "abc123" in 'store'
            )
            st.write(st.session_state.store)
            st.write("Assistant:",response['answer'])
            st.write("Chat History:",session_history.messages)

else:
    st.warning(" Please Enter the Groq API Key")
