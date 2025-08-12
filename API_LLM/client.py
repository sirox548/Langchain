import streamlit as st
import requests
import json


def get_openai_response(input_text):
    response = requests.post("http://localhost:9000/essay/invoke", 
                             json = {'input': {'topic': input_text}})
    return response.json()['output']['content']

def get_ollanma_response(input_text):
    response = requests.post("http://localhost:9000/poem/invoke", 
                             json = {'input': {'topic': input_text}})
    return response.json()['output']

# streamlit framework
st.title("Langchanin Demo with LLAMA2 and OPENAI")
# input request for openai
input_text = st.text_input("Write an essay on ")
# input request for llama2
input_text1 = st.text_input("Write a poem on")


# function calls
if input_text:
    st.write(get_openai_response(input_text))

if input_text1:
    st.write(get_ollanma_response(input_text1))