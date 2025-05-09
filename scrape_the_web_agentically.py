"""Agentic web scraping example using ScrapeGraphAI and LangGraph.

This module implements a web scraping agent that uses ScrapeGraphAI for intelligent
content extraction and LangGraph for orchestration. The agent can process multiple URLs
in parallel and extract information based on a specified keyword.
"""

import logging
from operator import or_
from typing import Any, Dict, List, Optional, TypedDict, Union

import nest_asyncio
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from pydantic_settings import BaseSettings
from scrapegraphai.graphs import SmartScraperGraph
from typing_extensions import Annotated

nest_asyncio.apply()

class Settings(BaseSettings):
    openai_api_key: str

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore'

def load_settings() -> Settings:
    """Load application settings from environment or .env file."""
    try:
        return Settings()
    except ValueError as e:
        logging.error(f"Configuration error: {e}. Ensure OPENAI_API_KEY is set in your environment or .env file.")
        raise

def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with consistent formatting."""
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def first_non_null(a: Any, b: Any) -> Any:
    """Return the first non-null value between two values."""
    return a if a is not None else b

class InputState(TypedDict):
    """Input schema for the LangGraph agent."""
    urls: Union[str, List[str]]
    keyword: str


class OverallState(TypedDict):
    """Overall state schema for the LangGraph agent."""
    initial_urls: List[str]
    current_url_index: int
    total_urls: int
    urls_to_scrape: List[str]
    extracted_info: Annotated[Optional[Dict[str, Any]], first_non_null]
    extracted_from_url: Annotated[Optional[str], first_non_null]
    is_information_found: Annotated[Optional[bool], or_]
    keyword: str

def initialize_state(state: Dict[str, Any], config: Dict[str, Any]) -> OverallState:
    """Initialize the agent's state from the input parameters."""
    logging.info("Executing node: initialize_state")
    logging.info(f"Input state: {state}")

    urls = state.get("urls", [])
    keyword = state.get("keyword", "")

    if not urls:
        logging.info("No URLs found in input state, using default URL")
        urls = ["https://python.langchain.com"]
    
    if not keyword:
        logging.info("No keyword found in input state, using default keyword")
        keyword = "How to track token usage for LLMs"

    initial_urls = [urls] if isinstance(urls, str) else urls
    
    logging.info(f"Initialized with URLs: {initial_urls} and keyword: '{keyword}'")
    
    return {
        "initial_urls": initial_urls,
        "current_url_index": 0,
        "total_urls": len(initial_urls),
        "urls_to_scrape": initial_urls.copy(),
        "keyword": keyword,
        "extracted_info": None,
        "extracted_from_url": None,
        "is_information_found": False,
    }


def scrape_manager(state: OverallState) -> OverallState:
    """Manage the URLs for scraping."""
    logging.info("Executing node: scrape_manager")

    urls_to_scrape = state.get("urls_to_scrape", [])
    total_urls = state.get("total_urls", 0)
    current_index = state.get("current_url_index", 0)
    
    if not urls_to_scrape:
        logging.info("No more URLs to process.")
        return state

    remaining = len(urls_to_scrape)
    processed = total_urls - remaining
    progress = (processed / total_urls) * 100 if total_urls > 0 else 0
    
    logging.info(
        f"URLs remaining: {remaining} out of {total_urls} "
        f"(Progress: {progress:.2f}%)"
    )
    
    return state

def send_to_scraper(state: OverallState) -> List[Send]:
    """Fan-out URLs to parallel scraper node instances."""
    urls_to_scrape = state.get("urls_to_scrape", [])
    if not urls_to_scrape:
        logging.info("No URLs left in the current batch to send to scraper.")
        return []

    logging.info(f"Sending {len(urls_to_scrape)} URLs to scraper node.")

    return [
        Send("scraper", {"url": url, "keyword": state["keyword"]})
        for url in urls_to_scrape
    ]


def scraper(state: OverallState, config: Dict[str, Any]) -> Dict[str, Any]:
    """Scrape a single URL using ScrapeGraphAI based on the keyword."""
    urls_to_scrape = state.get("urls_to_scrape", [])
    if not urls_to_scrape:
        logging.warning("Scraper node called with no URLs to scrape")
        return {
            "extracted_info": None,
            "extracted_from_url": None,
            "is_information_found": False,
            "urls_to_scrape": []
        }

    url = urls_to_scrape[0]
    remaining_urls = urls_to_scrape[1:]
    keyword = state["keyword"]
    
    logging.info(f"Executing node: scraper for URL: {url} with keyword: '{keyword}'")

    settings: Settings = config["configurable"]["settings"]
    api_key = settings.openai_api_key

    if not api_key:
        logging.error(f"OpenAI API key missing for scraping {url}.")
        return {
            "extracted_info": None, 
            "extracted_from_url": url, 
            "is_information_found": False
        }

    prompt = (
        f"Extract information related to '{keyword}'. Summarize the key points concisely. "
        f"If the keyword is not mentioned or relevant, state that clearly."
        f"Add full source url without modification where you found keyword."
    )

    graph_config = {
        "llm": {
            "api_key": api_key,
            "model": "gpt-4o",
            "temperature": 0.1,
        },
        "verbose": True,
        "headless": True,
    }

    information_found = False

    try:
        logging.info(f"Initializing SmartScraperGraph for: {url}")
        smart_scraper_graph = SmartScraperGraph(
            prompt=prompt,
            source=url,
            config=graph_config
        )

        logging.info(f"Running ScrapeGraphAI for: {url}")
        result = smart_scraper_graph.run()
        logging.info(f"ScrapeGraphAI finished for: {url}")
        logging.debug(f"ScrapeGraphAI raw result for {url}: {result}")

        if isinstance(result, dict):
            extracted_data = result
            summary = str(result.values()).lower()
            if (keyword.lower() in summary and 
                "not mentioned" not in summary and 
                "not relevant" not in summary):
                information_found = True
                logging.info(f"Found relevant information for '{keyword}' in {url}")
            else:
                logging.info(
                    f"Keyword '{keyword}' likely not found or relevant in {url} "
                    f"based on summary: {summary[:100]}..."
                )
        else:
            logging.warning(f"ScrapeGraphAI returned non-dict result for {url}: {type(result)}")
            extracted_data = {"raw_output": str(result)}

    except Exception as e:
        logging.error(f"Error running ScrapeGraphAI for {url}: {e}")
        extracted_data = {"error": str(e)}

    return {
        "extracted_info": extracted_data,
        "extracted_from_url": url,
        "is_information_found": information_found,
        "urls_to_scrape": remaining_urls
    }

def evaluate(state: OverallState) -> Dict[str, Any]:
    """Evaluate if relevant information was found in the latest scrape."""
    logging.info("Executing node: evaluate")

    if state.get("is_information_found"):
        logging.info(f"Relevant information found for '{state['keyword']}' from URL: {state['extracted_from_url']}")
        return {"is_information_found": True}
    else:
        logging.info("Relevant information not found in the last scraped URL.")
        return {"is_information_found": False}


def should_continue_scraping(state: OverallState) -> str:
    """Determine the next step in the workflow based on the current state."""
    logging.info("Executing node: should_continue_scraping (conditional edge)")

    if state.get("is_information_found"):
        logging.info("Relevant information found. Ending process.")
        return "end_process"
    elif state["current_url_index"] < state["total_urls"]:
        logging.info("Information not found yet, more URLs to process. Continuing.")
        return "continue_scraping"
    else:
        logging.info("Information not found and no more URLs left. Ending process.")
        return "end_process"

def create_graph(settings: Settings) -> CompiledStateGraph:
    """Build the LangGraph agent for web scraping with ScrapeGraphAI."""
    builder = StateGraph(OverallState, config_schema=Settings)

    builder.add_node("initialize_state", initialize_state)
    builder.add_node("scrape_manager", scrape_manager)
    builder.add_node("scraper", scraper)
    builder.add_node("evaluate", evaluate)

    builder.add_edge(START, "initialize_state")
    builder.add_edge("initialize_state", "scrape_manager")

    builder.add_conditional_edges(
        "scrape_manager",
        lambda state: "scraper" if state.get("urls_to_scrape") else "evaluate",
        {
            "scraper": "scraper",
            "evaluate": "evaluate"
        }
    )

    builder.add_edge("scraper", "evaluate")

    builder.add_conditional_edges(
        "evaluate",
        should_continue_scraping,
        {
            "continue_scraping": "scrape_manager",
            "end_process": END
        }
    )

    graph = builder.compile(
        checkpointer=None,
        interrupt_after=["evaluate"]
    )
    
    logging.info("ScrapeGraphAI LangGraph compiled successfully.")
    return graph

def main(urls: Union[str, List[str]], keyword: str) -> None:
    """Run the ScrapeGraphAI web scraping agent."""
    setup_logging()
    logging.info(f"Starting ScrapeGraphAI agent for URLs: {urls}, Keyword: {keyword}")

    extraction_tracker = {
        "found_information": False,
        "extracted_info": None,
        "source_url": None
    }

    try:
        settings = load_settings()
    except ValueError as e:
        logging.error(f"Failed to initialize settings: {e}")
        return

    graph = create_graph(settings)

    if isinstance(urls, str):
        url_list = [urls]
    else:
        url_list = urls
        
    inputs = {
        "urls": url_list,
        "keyword": keyword
    }
    config = {"configurable": {"settings": settings}}

    final_state = None
    logging.info("Invoking ScrapeGraphAI graph...")

    for event in graph.stream(inputs, config=config):
        for key, value in event.items():
            logging.debug(f"Graph Event: Node='{key}', State Update='{value}'")

            if key == "scraper" and isinstance(value, dict):
                if value.get("is_information_found", False):
                    extraction_tracker["found_information"] = True
                    extraction_tracker["extracted_info"] = value.get("extracted_info")
                    extraction_tracker["source_url"] = value.get("extracted_from_url")
                    logging.debug(f"Tracked successful extraction from {extraction_tracker['source_url']}")

            if isinstance(value, dict) and '__end__' in value:
                final_state = value
                logging.info("Graph execution finished.")
                break
            elif isinstance(value, dict):
                final_state = value

        if final_state and '__end__' in final_state:
            break

    logging.info("--- Final Agent State ---")

    found = extraction_tracker["found_information"]
    info = extraction_tracker["extracted_info"]
    source_url = extraction_tracker["source_url"]

    if found and info:
        logging.info(f"\n✅ Relevant Information Found for '{keyword}'!")
        logging.info(f"Source URL: {source_url}")
        logging.info(f"Extracted Information:\n---\n{info}\n---")
    else:
        logging.info(f"\n❌ Relevant information for '{keyword}' could not be found in the processed URLs.")
        if info:
            logging.info(f"Last extracted attempt from {source_url}:\n---\n{info}\n---")
        elif not final_state:
            logging.error("Graph execution did not produce a final state.")

if __name__ == "__main__":
    target_urls = [
        "https://python.langchain.com"
    ]
    search_keyword = "How to track token usage for LLMs"

    if not target_urls or not search_keyword:
        print("Please set the target_urls list and search_keyword variable.")
    else:
        main(target_urls, search_keyword)
