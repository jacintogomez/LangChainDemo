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
from langchain_community.document_loaders import PyPDFLoader
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
loader=PyPDFLoader('book.pdf')
pages=loader.load_and_split()

# faissindex=FAISS.from_documents(pages,OpenAIEmbeddings())
# docs=faissindex.similarity_search('How will the community be engaged',k=2)
# for page in pages:
#     print('Page '+str(page.metadata['page'])+':',page.page_content[:300])

embeddings=OpenAIEmbeddings()
tsplitter=RecursiveCharacterTextSplitter()
documents=tsplitter.split_documents(pages)
vec=FAISS.from_documents(documents,embeddings)

retriever=vec.as_retriever()
retrievalchain=create_retrieval_chain(retriever,documentchain)
question='Who are the residents for which this contract was written?'
response=retrievalchain.invoke({'input':question})
print(question)
cleanans=response['answer']
print(consoleformat(cleanans))