import streamlit as st 
import os 
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings,ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS



from dotenv import load_dotenv
load_dotenv()


#Loading the NVIDIA API KEY
Nvidia_api_key=os.getenv("NVIDIA_API_KEY")


#Initilaizing NVIDIA LLM
llm=ChatNVIDIA(model="meta/llama3-70b-instruct")


#Fucntion for creating vector embeddings
def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=NVIDIAEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("./us_census")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50)
        # For fast processing i have added only first 20 documents,you can add all
        st.session_state.final_docs=st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors=FAISS.from_documents(documents=st.session_state.final_docs,embedding=st.session_state.embeddings)


st.title("PDF Q&A with NVIDIA NIM and LANGCHAIN")

prompt=ChatPromptTemplate.from_template(
""" 
Answer the questions based on the provided context only.
please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)

prompt1=st.write("To proceed click on Document Embedding")

if st.button("Document Embedding"):
    vector_embeddings()
    st.write("FAISS DB is ready with NIVIA embeddingss")

prompt1=st.text_input("Enter your Questions From Documents")

if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    with st.spinner("konchem agu ra osthadi answer..😂"):
                response= retrieval_chain.invoke({"input":prompt1})

    st.success("Ochindi chaduko inka:")
    st.write(response["answer"])