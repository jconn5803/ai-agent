from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

# Set up an LLMs
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("What is the meaning of life?")
print(response)
