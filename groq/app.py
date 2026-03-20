import streamlit as st

import os

from langchain_groq import ChatGroq

from langchain_community.document_loaders import WebBaseLoader

from langchain_community.embeddings import OllamaEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains import create_retrieval_chain

from langchain_community.vectorstores import FAISS

import time



from dotenv import load_dotenv

load_dotenv()



## Load the groq api key

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")



if "vector" not in st.session_state:

    st.session_state.embeddings = OllamaEmbeddings(model="llama3.1")

    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")

    st.session_state.docs = st.session_state.loader.load()



    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])

    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)



st.title("ChatGroq Demo")

llm=ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"],

model="llama-3.1-8b-instant")



prompt=ChatPromptTemplate.from_template(

    """

    Answer the questions based on the provided context only.

    Please provide the most accurate response based on the question

    <context>

    {context}

    </context>

    Questions: {input}

    """

)



document_chain = create_stuff_documents_chain(llm, prompt)

retriever = st.session_state.vectors.as_retriever()

retrieval_chain = create_retrieval_chain(document_chain, retriever)



prompt=st.text_input("Input your prompt here")

if prompt:

    start = time.process_time()

    response = retrieval_chain.invoke({"input": prompt})

    print("Response time: ", time.process_time() - start)

    st.write(response['answer'])



    # With a streamlit expander

    with st.expander("Document Similarity Search"):

        # Find the relevant chunks

        if 'context' in response:

            for i, doc in enumerate(response['context']):

                st.write(doc.page_content)

                st.write("---------------------------------")











