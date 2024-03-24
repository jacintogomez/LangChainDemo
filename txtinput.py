# setup environment variables
from dotenv import load_dotenv
import os
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import TextLoader
import textwrap

def consoleformat(input,width=100):
    formatted=textwrap.fill(input,width)
    return formatted

llm=ChatOpenAI()
prompt=ChatPromptTemplate.from_template("""Answer the following request based only on the provided context:

<context>
{context}
</context>

Question: {input}""")
documentchain=create_stuff_documents_chain(llm,prompt)
outputparser=StrOutputParser()
chain=prompt|llm|outputparser
loader=TextLoader('book.txt')
docs=loader.load()

embeddings=OpenAIEmbeddings()
tsplitter=RecursiveCharacterTextSplitter()
documents=tsplitter.split_documents(docs)
vec=FAISS.from_documents(documents,embeddings)

retriever=vec.as_retriever()
retrievalchain=create_retrieval_chain(retriever,documentchain)
question='Please summarize the given file in 5 sentences or less'
response=retrievalchain.invoke({'input':question})
print(question)
cleanans=response['answer']
print(consoleformat(cleanans))