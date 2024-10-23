import os
import sys
import time
import json
import logging
from typing import Generator, List, Optional, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ChatConfig:
    """Configuration settings for the chatbot"""
    temperature: float = 0.7
    max_search_results: int = 3
    type_delay: float = 0.005
    max_retries: int = 3
    timeout: float = 30.0

class PromptTemplates:
    """Container for all prompt templates used in the chatbot"""
    
    @staticmethod
    def get_extract_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a professional prompt analyst. Analyze the provided prompt and previous context to extract search queries.

            Tasks:
            1. Identify parts needing real-time or current information
            2. Extract those parts and convert to search-friendly terms
            3. Focus on factual queries that would benefit from current information
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "search online:{input}"),
        ])

    @staticmethod
    def get_standardize_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """Convert the text into a list of search queries following these rules:
             1. Return queries in a valid JSON list format
             2. Include only strings in the list
             3. Maximum 3 items
             4. Return empty list if no valid queries
             5. Generate only the list, nothing else"""),
            ("human", "search online:{input}"),
        ])

    @staticmethod
    def get_online_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant that provides accurate answers using online information."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("system", "Search results: {search_results}")
        ])

    @staticmethod
    def get_normal_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant that provides accurate and informative answers."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

class WebChatbot:
    """An AI chatbot with real-time web search capabilities"""

    def __init__(self, config: Optional[ChatConfig] = None):
        """Initialize the chatbot with optional custom configuration"""
        self.config = config or ChatConfig()
        self._initialize_environment()
        self._setup_components()
        self._setup_chains()

    def _initialize_environment(self) -> None:
        """Set up environment variables and API keys"""
        try:
            load_dotenv()
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            os.environ["GROQ_API_KEY"] = api_key
        except Exception as e:
            logger.error(f"Failed to initialize environment: {str(e)}")
            raise

    def _setup_components(self) -> None:
        """Initialize core components of the chatbot"""
        self.search_tool = DuckDuckGoSearchRun()
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        self.llm = ChatGroq(
            temperature=self.config.temperature,
            streaming=True,
            timeout=self.config.timeout
        )
        self.prompts = PromptTemplates()

    def _setup_chains(self) -> None:
        """Set up the processing chains"""
        self.extract_chain = (
            self.prompts.get_extract_prompt()
            | self.llm
            | StrOutputParser()
        )
        self.standardize_chain = (
            self.prompts.get_standardize_prompt()
            | self.llm
            | StrOutputParser()
        )
        self.online_chain = (
            self.prompts.get_online_prompt()
            | self.llm
            | StrOutputParser()
        )
        self.normal_chain = (
            self.prompts.get_normal_prompt()
            | self.llm
            | StrOutputParser()
        )

    @lru_cache(maxsize=100)
    def _parse_json_list(self, json_str: str) -> List[str]:
        """Parse JSON string into list with error handling and caching"""
        try:
            # Clean up the JSON string
            json_str = json_str.strip()
            list_start, list_end = json_str.find("["), json_str.rfind("]")
            if list_start == -1 or list_end == -1:
                return []
            cleaned_json = json_str[list_start:list_end + 1]
            return json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {str(e)}")
            return []

    async def extract_searchable_queries(self, user_input: str, chat_history: List[Dict[str, Any]]) -> List[str]:
        """Extract search queries from user input with retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                res = await self.extract_chain.ainvoke({
                    "input": user_input,
                    "chat_history": chat_history
                })
                standardized = await self.standardize_chain.ainvoke({"input": res})
                queries = self._parse_json_list(standardized)
                return queries[:self.config.max_search_results]
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.config.max_retries - 1:
                    logger.error("Failed to extract queries after all retries")
                    return []
                await asyncio.sleep(1)

    async def get_online_answers(self, queries: List[str]) -> str:
        """Perform concurrent web searches for queries"""
        if not queries:
            return ""

        async def search_info(query: str) -> str:
            try:
                return await self.search_tool.arun(query)
            except Exception as e:
                logger.error(f"Search failed for query '{query}': {str(e)}")
                return ""

        tasks = [search_info(query) for query in queries]
        results = await asyncio.gather(*tasks)
        return "\n".join(filter(None, results))

    def stream_output(self, text: str) -> Generator[str, None, None]:
        """Stream output with controlled timing"""
        for char in text:
            yield char
            time.sleep(self.config.type_delay)

    async def process_input(self, user_input: str) -> Generator[str, None, None]:
        """Process user input and generate response"""
        try:
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            queries = await self.extract_searchable_queries(user_input, chat_history)
            search_results = await self.get_online_answers(queries)
            
            chain = self.online_chain if search_results else self.normal_chain
            response = ""

            async for chunk in chain.astream({
                "input": user_input,
                "search_results": search_results,
                "chat_history": chat_history
            }):
                response += chunk
                yield chunk

            self.memory.save_context(
                {"input": user_input},
                {"output": response}
            )
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            yield "I apologize, but I encountered an error. Please try again."

    async def run(self) -> None:
        """Run the chatbot interface"""
        print("Welcome to the Web Chatbot!")
        print("Type 'quit' to exit the conversation.\n")

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ['quit', 'exit']:
                    print("\nGoodbye! Thank you for chatting.")
                    break

                if user_input:
                    print("\nBot:", end=" ")
                    async for chunk in self.process_input(user_input):
                        print(chunk, end="")
                        sys.stdout.flush()

            except KeyboardInterrupt:
                print("\n\nExiting gracefully...")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                print("\nAn unexpected error occurred. Please try again.")

if __name__ == "__main__":
    import asyncio
    chatbot = WebChatbot()
    asyncio.run(chatbot.run())