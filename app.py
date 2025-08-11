# Import necessary libraries
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# Load environment variables from a .env file.
# This makes API keys available as environment variables.

load_dotenv()


# Check for required environment variables and raise an error if they are missing.
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY environment variable is not set. Please check your .env file.")
    st.stop()
if not os.getenv("LANGCHAIN_API_KEY"):
    st.error("LANGCHAIN_API_KEY environment variable is not set. Please check your .env file.")
    st.stop()


# Set up LangSmith for tracing, which is great for debugging and monitoring.
# The `load_dotenv()` call above will load the keys if they are in your .env file.
os.environ["LANGCHAIN_TRACING_V2"] = "true"


# LangChain Components
# Define the Prompt Template.
# This template structures the input for the LLM.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("human", "Question: {question}")
    ]
)

# Initialize the OpenAI LLM.
# We're using `gpt-3.5-turbo` with a temperature of 0, which makes the
# responses more deterministic and factual, ideal for a general-purpose chatbot.
try:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
except OpenAIError as e:
    st.error(f"Error initializing OpenAI client: {e}")
    st.warning("Please verify that your OPENAI_API_KEY is correct and active.")
    st.stop()

# Initialize the Output Parser.
# This will parse the LLM's output into a simple string.
output_parser = StrOutputParser()

# Create the LangChain Expression Language (LCEL) chain.
# The chain passes the user's question through the prompt, then to the LLM,
# and finally parses the output.
chain = prompt | llm | output_parser


# Streamlit Application
# Set the title for the Streamlit app.
st.title('Langchain Chatbot Demo with OPENAI API')

# Create a text input box for the user.
input_text = st.text_input("Search the topic you want")

# Check if the user has entered any text.
if input_text:
    # Invoke the chain with the user's input and display the response.
    # The `invoke` method is used for a single, synchronous call.
    with st.spinner("Thinking..."):
        st.write(chain.invoke({'question': input_text}))
