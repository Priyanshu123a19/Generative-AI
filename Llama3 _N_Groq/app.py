# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# import time


# from dotenv import load_dotenv
# load_dotenv()

# ##will be importing the groq api key from the .env file
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# st.title("Llama 3 - N - Groq")

# llm=ChatGroq(GROQ_API_KEY=GROQ_API_KEY, model="Llama3-8b-8192", temperature=0)


# prompt=ChatPromptTemplate.from_template(
# """
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question
# <context>
# {context}
# <context>
# Questions:{input}
# """
# )



# def Vector_Embedding():
#     ##all things will be made in the session state
#     if "vectordb" not in st.session_state:  
#         st.session_state.embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#         st.session_state.loader=PyPDFDirectoryLoader("pdfs/")
#         st.session_state.documents=st.session_state.loader.load()
#         st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         st.session_state.docs=st.session_state.text_splitter.split_documents(st.session_state.documents[:50])
#         st.session_state.vectordb=Chroma.from_documents(st.session_state.docs, st.session_state.embeddings, collection_name="my_collection")



# prompt1=st.text_area("Prompt", "What is the capital of France?")

# if st.button("GenerateEmbeddings"):
#     Vector_Embedding()
#     st.success("Embeddings Generated")


# document_chain=create_stuff_documents_chain(llm=llm, prompt=prompt)
# retriever=st.session_state.vectordb.as_retriever()
# retriever_chain=create_retrieval_chain(retriever,document_chain)

# if prompt1:
#     start=time.process_time()
#     response=retriever_chain.invoke({"input":prompt1})
#     end=time.process_time()
#     st.write(response['answer'])
#     st.write(f"Time taken to answer the question is {end-start} seconds")
    
#     with st.expander("Document Similarity Search"):
#         # Find the relevant chunks
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------------")

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time

from dotenv import load_dotenv
load_dotenv()

# Will be importing the groq api key from the .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.title("Llama 3 - N - Groq")

# Fixed: Correct parameter name and model name
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0)

# Fixed: Correct context closing tag
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Questions:{input}
"""
)

def Vector_Embedding():
    # All things will be made in the session state
    if "vectordb" not in st.session_state:  
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        st.session_state.loader = PyPDFDirectoryLoader("pdfs/")
        st.session_state.documents = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.docs = st.session_state.text_splitter.split_documents(st.session_state.documents[:50])
        st.session_state.vectordb = Chroma.from_documents(st.session_state.docs, st.session_state.embeddings, collection_name="my_collection")

prompt1 = st.text_area("Prompt", "What is the capital of France?")

if st.button("Generate Embeddings"):
    Vector_Embedding()
    st.success("Embeddings Generated")

# FIXED: Only create retriever chain if vectordb exists
if "vectordb" in st.session_state:
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever = st.session_state.vectordb.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    
    if prompt1:
        start = time.process_time()
        try:
            response = retriever_chain.invoke({"input": prompt1})
            end = time.process_time()
            st.write(response['answer'])
            st.write(f"Time taken to answer the question is {end-start} seconds")
            
            with st.expander("Document Similarity Search"):
                # Find the relevant chunks
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
        except Exception as e:
            st.error(f"Error processing request: {e}")
else:
    if prompt1:
        st.warning("Please generate embeddings first by clicking 'Generate Embeddings' button!")






