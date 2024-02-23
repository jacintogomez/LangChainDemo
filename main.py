from langchain_openai import ChatOpenAI

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

print(llm.invoke("what color is the sky usually?"))