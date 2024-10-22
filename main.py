import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
import sys

from dotenv import load_dotenv 
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROK_TOKEN")

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.7,
    streaming=True
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that answers questions"),
    ("human", "{input}"),
])

chain = (
    { "input": RunnablePassthrough() }
    | prompt
    | llm
    | StrOutputParser()
)

def stream_output(text):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()

def process_input(user_input):
    try:
        for chunk in chain.stream({"input": user_input}):
            stream_output(chunk)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")


user_input = input("\nYou: ").strip()
    
if user_input:
    print("\nBot:", end=" ")
    process_input(user_input)