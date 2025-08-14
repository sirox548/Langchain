# Import necessary libraries
import time
import os
import streamlit as st

# LangChain imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load Groq API Key from environment variable
# It's a good practice to handle cases where the key might not be set
try:
    groq_api_key = os.environ['GROQ_API_KEY']
except KeyError:
    st.error("GROQ_API_KEY environment variable not set. Please add it to your .env file.")
    st.stop() # Stop the app if the key is not found

# --- Streamlit Session State Initialization ---
# The logic below runs only once when the app is first loaded.
# It prevents expensive operations (like loading documents and creating embeddings)
# from running on every user interaction.
if "vectors" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="llama2")
    # testing https://bloodstemcell.hrsa.gov/about/advisory-council/charter
    # testing https://pmc.ncbi.nlm.nih.gov/articles/PMC10424908/#:~:text=The%20application's%20interface%20allows%20users,for%20non%2Dinstitutionalized%20US%20adults.
    st.session_state.loader = WebBaseLoader("https://pmc.ncbi.nlm.nih.gov/articles/PMC10424908/#:~:text=The%20application's%20interface%20allows%20users,for%20non%2Dinstitutionalized%20US%20adults.")
    st.session_state.text = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(
        st.session_state.text
    )
    # The key "vectors" is used here to match the variable name
    st.session_state.vectors = FAISS.from_documents(
        st.session_state.final_documents,
        st.session_state.embeddings
    )

st.title("ChatGroq Demo")

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")
# Define the prompt template for the RAG chain
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Questions: {input}
    """
)

# Create the document and retrieval chains
document_chain = create_stuff_documents_chain(llm, prompt)

# Corrected the key: used "vectors" to match the session state key
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# User input
user_prompt = st.text_input("Input your prompt here :")

# Handle the user's prompt
if user_prompt:
    with st.spinner("Thinking..."):
        start = time.process_time()
        try:
            response = retrieval_chain.invoke({"input": user_prompt})
            st.write(f"Response time: {time.process_time() - start:.2f} seconds")
            st.write(response['answer'])
            # Store the response in session state to use in the expander
            st.session_state.last_response = response
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.last_response = None

# Display the similarity search results if a response exists
if "last_response" in st.session_state and st.session_state.last_response:
    with st.expander("Document Similarity Search"):
        # The key is 'context' in the response dictionary.
        for i, text in enumerate(st.session_state.last_response['context']):
            st.write(text.page_content)
            st.write("-----------------------------")