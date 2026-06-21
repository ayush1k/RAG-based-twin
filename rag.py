from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

# Setup the model
# Qwen/Qwen2.5-7B-Instruct is supported via the Hugging Face Serverless API's conversational task.
model_id = "Qwen/Qwen2.5-7B-Instruct"
token = os.getenv("HF_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id=model_id,
    huggingfacehub_api_token=token,
    task="conversational"
)

loader = DirectoryLoader(
    path='data',
    glob = '*.pdf',
    loader_cls=PyPDFLoader
)

chat = ChatHuggingFace(llm=llm)

# Execute and print results
messages = [HumanMessage(content="What is the capital of India?")]
result = chat.invoke(messages)
print(result.content)