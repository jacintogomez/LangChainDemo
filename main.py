# setup environment variables
from dotenv import load_dotenv
import os
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# import LangChain packages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter





llm = ChatOpenAI()

# Give directions on how system should act
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a cowboy from the deep USA south."),
    ("user", "{input}")
])

# Convert output to string format
output_parser = StrOutputParser()

# Chain everything together
chain = prompt | llm | output_parser

# Load reference data to index
loader = WebBaseLoader("https://docs.smith.langchain.com")
docs = loader.load()

# Embedding model
embeddings = OpenAIEmbeddings()

# Index data into a vectorstore
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

print(chain.invoke({"input": "What are some examples of Mexican food?"}))