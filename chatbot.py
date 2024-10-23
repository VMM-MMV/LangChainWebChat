import os
import sys
import time
import json
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv 

class WebChatbot:
    def __init__(self):
        load_dotenv()
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

        self.search_tool = DuckDuckGoSearchRun()
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        self.llm = ChatGroq(
            temperature=0.7,
            streaming=True
        )

        self.extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a AI assitant. You will analyze the provided prompt and previous context to extract search queries.

            Your specific tasks:
            1. Identify any parts that need real-time or current information
            2. Extract those parts and convert them to search-friendly terms
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "search online:" + "{input}"),
        ])

        self.standardize_prompt = ChatPromptTemplate.from_messages([
            ("system", """You will be given some text that has in it some search queries.
             You will follow:
             1. You will return the queries in a python/JSON list.
             2. Only strings in the list.
             3. At most 3 items.
             4. If there are no items, return the list empty. 
             5. You will not generate anything other than that list."""),
            ("human", "search online:" + "{input}"),
        ])

        self.online_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant that answers questions, which have online answers to them."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("system", "Search results: {search_results}")
        ])

        self.normal_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant that answers questions"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        self.extract_chain = (
            self.extract_prompt
            | self.llm
            | StrOutputParser()
        )

        self.standardize_chain = (
            self.standardize_prompt
            | self.llm
            | StrOutputParser()
        )

        self.online_chain = (
            self.online_prompt
            | self.llm
            | StrOutputParser()
        )

        self.normal_chain = (
            self.normal_prompt
            | self.llm
            | StrOutputParser()
        )

    def extract_searchable_queries(self, user_input, chat_history):
        res = self.extract_chain.invoke({"input": user_input, "chat_history": chat_history})
        res = self.standardize_chain.invoke({"input": res})
        print(res)
        list_start, list_end = res.rfind("["), res.rfind("]")
        list_body = res[list_start:list_end+1].replace("[", "").replace("]", "")
        list_body = f"[{list_body}]"
        print(list_body)
        return json.loads(list_body)

    def get_online_answers(self, info):
        def search_info(item):
            return self.search_tool.run(item)

        if info:
            results = []
            with ThreadPoolExecutor() as executor:
                future_to_item = {executor.submit(search_info, item): item for item in info}

                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    result = future.result()
                    results.append(result)

            return "\n".join(results)
        else:
            return ""

    def route_searched(self, online_response):
        if online_response == "":
            return self.normal_chain
        else:
            return self.online_chain

    @staticmethod
    def precise_sleep(seconds):
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            pass

    def stream_output(self, text):
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            WebChatbot.precise_sleep(0.005)

    def process_input(self, user_input):
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        online_queries = self.extract_searchable_queries(user_input, chat_history)
        search_results = self.get_online_answers(online_queries)
        llm = self.route_searched(search_results)

        response = ""
        for chunk in llm.stream({"input": user_input, "search_results": search_results, "chat_history": chat_history}):
            response += chunk
            yield chunk

        self.memory.save_context(
            {"input": user_input},
            {"output": response}
        )

    def run(self):
        print("Welcome to the Web Chatbot!")
        print("Type 'quit' to exit the conversation.\n")

        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye! Thank you for chatting.")
                break

            if user_input:
                print("\nBot:", end=" ")
                for chunk in self.process_input(user_input):
                    print(chunk, end="")

if __name__ == "__main__":
    WebChatbot().run()
