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
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage



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
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Bitcoin")
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

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
retriever_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

chat_history = [HumanMessage(content="Is Satoshi Nakamoto the inventor of Bitcoin?"), AIMessage(content="Yes!")]
print(retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me when he did it"
}))