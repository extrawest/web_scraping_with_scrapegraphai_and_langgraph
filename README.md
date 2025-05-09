# 🧠 Agentic Web Scraping with ScrapeGraphAI and LangGraph

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]()
[![Maintainer](https://img.shields.io/static/v1?label=Yevhen%20Ruban&message=Maintainer&color=red)](mailto:yevhen.ruban@extrawest.com)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub release](https://img.shields.io/badge/release-v1.0.0-blue)

This repository contains an intelligent web scraping solution that uses ScrapeGraphAI for LLM-powered content extraction and LangGraph for orchestrating the scraping workflow. The system can intelligently crawl websites, extract content using natural language instructions, and search for specific information.




https://github.com/user-attachments/assets/a6b5abaa-980e-481b-b57b-0846e64ccc80




## 🚀 Features

- **LLM-Powered Extraction**: Uses OpenAI models to intelligently extract content based on natural language instructions
- **Parallel Processing**: Processes multiple URLs simultaneously using LangGraph's fan-out pattern
- **Flexible Prompting**: Customizable prompts for different scraping scenarios
- **Local Processing Control**: No remote servers continuing to consume credits
- **Progress Tracking**: Real-time progress updates during scraping
- **Error Handling**: Robust error handling for browser and API issues
- **Configurable**: Easy to configure for different websites and search terms

## 📋 Requirements

The code requires the following dependencies:
- Python 3.8+
- scrapegraphai
- langgraph
- nest_asyncio
- playwright
- pydantic-settings
- python-dotenv
- openai (for API access)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/extrawest/web_scraping_with_scrapegraphai_and_langgraph.git
cd web-scraping-scrapegraphai

# Install required packages
pip install -r requirements.txt

# Install Playwright browsers
playwright install

# Create a .env file with your configuration
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## 📝 Usage

### Command Line Usage

```bash
# Run the script directly
python scrape_the_web_agentically.py
```

### Configuration

You can modify the target URL and search keyword by editing the script:

```python
if __name__ == "__main__":
    target_urls = [
        "https://python.langchain.com"
    ]
    search_keyword = "How to track token usage for LLMs"

    if not target_urls or not search_keyword:
        print("Please set the target_urls list and search_keyword variable.")
    else:
        main(target_urls, search_keyword)
```

## 🧠 How It Works

The script uses a LangGraph workflow with ScrapeGraphAI to orchestrate the web scraping process:

1. **Initialization**: Sets up the initial state with the target URL and keyword
2. **Scrape Management**: Manages the URLs to be scraped
3. **Parallel Processing**: Uses LangGraph's fan-out pattern to process multiple URLs simultaneously
4. **LLM-Powered Extraction**: Uses OpenAI models to intelligently extract content from web pages
5. **Content Evaluation**: Determines if the extracted content contains the requested information
6. **Result Processing**: Formats and presents the extracted information

## 🔄 LangGraph Workflow

![langgraph_visualization](https://github.com/user-attachments/assets/f10f84db-0436-410e-8d67-cd60d8f8e87e)


The script uses LangGraph to create a structured workflow with the following nodes:

- `initialize_state`: Sets up the initial state with URLs and keyword
- `scrape_manager`: Manages the list of URLs to be scraped
- `scraper`: Extracts content from individual URLs using ScrapeGraphAI
- `evaluate`: Checks if the extracted content contains the requested information

The workflow continues until either the information is found or all URLs have been processed.

## 🤖 ScrapeGraphAI vs Firecrawl

ScrapeGraphAI offers several advantages over Firecrawl:

1. **LLM-Powered Extraction**: Uses OpenAI models to intelligently extract content based on natural language instructions
2. **Local Processing Control**: No remote servers continuing to consume credits
3. **More Flexible Scraping**: Natural language instructions allow for more nuanced content extraction
4. **Direct LLM-based Content Extraction**: Extracts content without requiring multiple API calls
