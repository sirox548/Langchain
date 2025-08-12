from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langserve import add_routes
from langchain_community.llms import Ollama
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check for required environment variables.
# For a FastAPI app, raising an exception is a clean way to handle missing configs.
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")

if not os.getenv("LANGCHAIN_API_KEY"):
    raise ValueError("LANGCHAIN_API_KEY environment variable is not set. Please check your .env file.")

# Set up LangSmith for tracing, which is great for debugging and monitoring.
os.environ["LANGCHAIN_TRACING_V2"] = "true"

#  FastAPI App Creation 
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple FastAPI Server.",
    docs_url=None,
    redoc_url=None, 
    openapi_url=None
)



# Define LLM and Prompts 
# Instantiating the LLMs to use for this app
model = ChatOpenAI(model="gpt-3.5-turbo") 
# Ollama version llama2
llm = Ollama(model="llama2")

# Defining the prompt templates
prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words.")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} with 100 words.")

add_routes(
    app,
    model,
    path="/openai",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="default"
)

# First route for the essay prompt
add_routes(
    app, 
    prompt1 | model,
    path="/essay"
)

# Second route for the poem prompt
add_routes(
    app,
    prompt2 | llm,
    path="/poem"
)

# Main Execution Block 
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=9000)


