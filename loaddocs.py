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
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import textwrap
def consoleformat(input,width=100):
    formatted=textwrap.fill(input,width)
    return formatted

llm = ChatOpenAI()

# Takes question and retrieved documents and generates an answer
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")
document_chain = create_stuff_documents_chain(llm, prompt)

# Convert output to string format
output_parser = StrOutputParser()

# Chain everything together
chain = prompt | llm | output_parser

# Load reference data to index
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Shenzhen#:~:text=Since%20then%2C%20this%20area%20has,more%20than%20600%20years%20ago.")
docs = loader.load()

# Embedding model
embeddings = OpenAIEmbeddings()

# Index data into a vectorstore
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# Use retriever to dynamically select relevant documents
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
question="What are some things a tourist can do in Shenzhen?"
response = retrieval_chain.invoke({"input": question})
print(question)
print(consoleformat(response["answer"]))

