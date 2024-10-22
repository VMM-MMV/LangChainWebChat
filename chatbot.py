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
            model="mixtral-8x7b-32768",
            temperature=0.7,
            streaming=True
        )

        self.extract_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional prompt analyst. "
                        "Your task is to identify parts of the prompt that require real-time or constantly updating information, " 
                        "extract those specific parts, " 
                        "then restructure them into common questions so they're more broad, for more accurate internet searching, "
                        "and add them to a json list."
                        "RETURN THE LIST LAST, WITH THE SEARCH TERMS ONLY, IF THERE ARE NONE MAKE THE LIST EMPTY."),
            ("human", "{input}"),
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

    def extract_searchable_queries(self, user_input):
        res = self.extract_chain.invoke({"input": user_input})
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
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as exc:
                        results.append(f"Error searching '{item}': {exc}")

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
        online_queries = self.extract_searchable_queries(user_input)
        search_results = self.get_online_answers(online_queries)
        llm = self.route_searched(search_results)
        chat_history = self.memory.load_memory_variables({})["chat_history"]

        response = ""
        for chunk in llm.stream({"input": user_input, "search_results": search_results, "chat_history": chat_history}):
            response += chunk
            # Yield each chunk for streaming in Streamlit
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
