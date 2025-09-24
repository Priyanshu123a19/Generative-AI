import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time



from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

##making a session and if the vector is not in the session state then we will be making the vector store
##otherwise we will be using the existing vector store that is already in the session state
##in this way it only gets created once and then we can use it multiple times without having to recreate it again and again
if "vector" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.loader = WebBaseLoader("https://docs.langchain.com/langsmith/observability")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vector = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

    

st.title("ChatGroq Demo")
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key, temperature=0)


prompt = ChatPromptTemplate.from_template("""
Answer the question based on the provoided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Question: {input}
"""
)                          

##creating the stuff document chain because it is the most straightforward way to combine documents and can get the response from the LLM that makes much more sense than the raw context
document_chain=create_stuff_documents_chain(llm=llm, prompt=prompt)
##the retriever chain helps us to retrieve the most relevant documents from the vector store based on the query
retriever=st.session_state.vector.as_retriever()

##this makes sure that we get the output from the retriever and then user the stuff document chain to get the final response that alredy comes from llm and makes much more clear sense
retriever_chain= create_retrieval_chain(retriever, document_chain)


prompt=st.text_input("Search the topic u want")
if prompt:
    start=time.process_time()
    response= retriever_chain.invoke({"input":prompt})
    print("Response time:", time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("\n")
