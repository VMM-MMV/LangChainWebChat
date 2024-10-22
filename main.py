import os
import sys
import time
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from dotenv import load_dotenv 
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

search_tool = DuckDuckGoSearchRun()

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.7,
    streaming=True
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that answers questions"),
    ("human", "{input}"),
    ("system", "Search results: {search_result}")
])

chain = (
    { 
        "input": RunnablePassthrough(),
        "search_result": lambda x: search_tool.run(x["input"])
    }
    | prompt
    | llm
    | StrOutputParser()
)

def precise_sleep(seconds):
    end_time = time.perf_counter() + seconds
    while time.perf_counter() < end_time:
        pass

def stream_output(text):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        precise_sleep(0.005)

def process_input(user_input):
    try:
        for chunk in chain.stream({"input": user_input}):
            stream_output(chunk)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")


user_input = input("\nYou: ").strip()
    
if user_input:
    print("\nBot: ", end=" ")
    process_input(user_input)