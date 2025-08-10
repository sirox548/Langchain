# IMPORTING ALL IMPORTANT LIBRAREIS
from dotenv import load_dotenv
load_dotenv() 

import os
import streamlit as st
from langchain.chat_models import ChatOpenAI      
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# GET API KEYS 
os.environ["OPENAI_API_KEY"]= os.getenv("OPENAI_API_KEY")
# - for langsmith Tracking
os.environ["LANGCHAIN_TRACING_V2"]= "true"
os.environ["LANGCHAIN_API_KEY"]= os.getenv("LANGCHAIN_API_KEY")


# Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user queries"),
        ("human", "Question: {question}")
    ]
)


# streamlit framework
st.title('Langchain Chatbot Demo with OPENAI API')
input_text = st.text_input("Search the topic you want")


# OpenAI LLm
llm= ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
output_parser= StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text})) 