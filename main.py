# setup environment variables
from dotenv import load_dotenv
import os
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# import LangChain packages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

print(chain.invoke({"input": "What are some examples of Mexican food?"}))