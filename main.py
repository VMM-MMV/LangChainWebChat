import os
import sys
import time
import json
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv 
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

search_tool = DuckDuckGoSearchRun()

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.7,
    streaming=True
)

extract_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional prompt analyst. "
                "Your task is to identify parts of the prompt that require real-time or constantly updating information, " 
                "extract those specific parts, don't repeat them, there have to be as few of them as possible, so we dont waste money on search requests, " 
                "then restructure them for easier searching on the internet"
                "and add them to a json list."
                "RETURN THE LIST LAST, WITH THE SEARCH TERMS ONLY, IF THERE ARE NONE MAKE THE ARRAY EMPTY. DON'T DO SIMILAR QUERIES"),
    ("human", "{input}"),
])

online_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that answers questions, which have online answers to them."),
    ("human", "{input}"),
    ("system", "Search results: {search_results}")
])

normal_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that answers questions"),
    ("human", "{input}"),
])

extract_chain = (
    extract_prompt
    | llm
    | StrOutputParser()
)

online_chain = (
    online_prompt
    | llm
    | StrOutputParser()
)

normal_chain = (
    normal_prompt
    | llm
    | StrOutputParser()
)

def extract_searchable_queries(user_input):
    res = extract_chain.invoke({"input": user_input})
    print(res)
    list_start, list_end = res.rfind("["), res.rfind("]")
    list_body = res[list_start:list_end+1].replace("[","").replace("]","").replace("'", '"')
    list_body = f"[{list_body}]"
    print(list_body)
    return json.loads(list_body)

# who is the president of moldova

# def get_online_answers(info):
#     print(info)
#     if info["topic"] != []:
#         return search_tool.run(info["topic"])
#     else:
#         return ""

# def get_online_answers(info):
#     if info != []:
#         return search_tool.run(info[0])
#     else:
#         return ""

def get_online_answers(info):
    def search_info(item):
        return search_tool.run(item)
    
    if info:
        results = []
        with ThreadPoolExecutor() as executor:
            future_to_item = { executor.submit(search_info, item): item for item in info }
            
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    results.append(f"Error searching '{item}': {exc}")  # Handle exceptions

        return "\n".join(results)  # Join all results into a single string
    else:
        return ""
    
def route_searched(online_response):
    if online_response == "":
        return normal_chain
    else:
        return online_chain

# chain = (
#     {"analysis" : lambda x: extract_searchable_queries(x["input"])} 
#     | 
#     {"search_results": lambda x: calculate_search(x)}
#     | 
#     search_chain
# )

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
    online_queries = extract_searchable_queries(user_input)
    search_results = get_online_answers(online_queries)
    llm = route_searched(search_results)
    for chunk in llm.stream({"input": user_input, "search_results": search_results}):
        stream_output(chunk)

# user_input = input("\nYou: ").strip()
user_input = "moldova president"
if user_input:
    print("\nBot: ", end=" ")
    process_input(user_input)