from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

## Langsmith tracking
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_PROJECT"]="First Chatbot14"
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"

## Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question:{question}")
    ]
)

## Streamlit framework

st.title("Langchain Demo with Ollama")
input_text = st.text_input("Search the topic you want")

## Ollama LLM
llm = OllamaLLM(model="gemma3:1b")
output_parser=StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))