# -*- coding: utf-8 -*-
"""
A beginner-friendly Python script to fetch, parse, and analyze 10-K filings
from the U.S. Securities and Exchange Commission (SEC).

This script is designed for a YouTube tutorial, demonstrating web scraping,
text processing, and basic data analysis in a single, easy-to-follow file.

Author: Your Name
Contact: Your Contact Info
Version: 1.0
"""

# =============================================================================
# 1. IMPORTS - The Tools We Need
# =============================================================================
# These are the Python libraries we'll use.
# - `os`: To interact with the operating system (e.g., for environment variables).
# - `re`: Stands for "Regular Expressions," used for advanced text pattern matching.
# - `json`: To work with JSON data, a common format for web data.
# - `time`: To add delays in our code, which is polite when scraping websites.
# - `requests`: A popular library to make HTTP requests to websites and get data.
# - `BeautifulSoup`: A fantastic library for parsing HTML and XML documents.

import os
import re
import json
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# --- Optional LLM Imports (Advanced) ---
# These are for an advanced feature (summarization with AI).
# If you don't have them, the script will use a simpler fallback.
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Note: LangChain or OpenAI library not found. Using fallback summarizer.")


# =============================================================================
# 2. CONFIGURATION - Setting Up Our Script
# =============================================================================
# It's a good practice to keep settings in one place.

# The SEC requires you to identify yourself in your requests.
# Replace with your actual name and email.
USER_AGENT = "Your Name your.email@example.com"

# A simple file-based cache to avoid re-downloading the same data.
# This saves time and reduces load on the SEC's servers.
CACHE_FILE = "sec_filings_cache.json"


# =============================================================================
# 3. CACHING - Saving and Loading Data
# =============================================================================
# Web scraping can be slow. Caching saves the data we've already downloaded.


def load_cache():
    """Loads the cache from a JSON file if it exists."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}  # Return empty cache if file is corrupted
    return {}


def save_cache(cache):
    """Saves the given cache dictionary to a JSON file."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)


# =============================================================================
# 4. SEC DATA FETCHING - Getting Data from the SEC Website
# =============================================================================
# These functions handle the logic of finding and downloading company filings.


def get_company_info(ticker):
    """
    Fetches company metadata from the SEC, including the CIK.
    The CIK (Central Index Key) is a unique identifier for each company.

    Args:
        ticker (str): The company's stock ticker (e.g., "AAPL" for Apple).

    Returns:
        str: The 10-digit CIK for the company, or None if not found.
    """
    print(f"Fetching company list to find CIK for {ticker}...")
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
        all_companies = response.json()

        # Find the CIK for the given ticker
        for company in all_companies.values():
            if company["ticker"].upper() == ticker.upper():
                cik = str(company["cik_str"]).zfill(10)
                print(f"Found CIK: {cik}")
                return cik
        print(f"Error: Ticker '{ticker}' not found.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching company list: {e}")
        return None


def get_latest_10k_filing_url(cik):
    """
    Finds the URL for the most recent '10-K' annual filing for a given CIK.

    Args:
        cik (str): The company's 10-digit CIK.

    Returns:
        str: The URL to the primary document of the latest 10-K filing, or None.
    """
    if not cik:
        return None

    print(f"Fetching submissions for CIK {cik}...")
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        submissions = response.json()

        # Find the first '10-K' in the list of recent filings
        recent_filings = submissions["filings"]["recent"]
        for i, form in enumerate(recent_filings["form"]):
            if form == "10-K":
                accession_num = recent_filings["accessionNumber"][i].replace("-", "")
                primary_doc = recent_filings["primaryDocument"][i]
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_num}/{primary_doc}"
                print(f"Found latest 10-K filing: {filing_url}")
                return filing_url

        print("Error: No 10-K filing found in recent submissions.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching submissions: {e}")
        return None
    except KeyError:
        print(
            "Error: Could not parse submissions JSON. The structure might have changed."
        )
        return None


def fetch_filing_html(url, ticker, year):
    """
    Fetches the HTML content of a filing from a URL, using a cache.

    Args:
        url (str): The URL of the filing to download.
        ticker (str): The company ticker, used as a cache key.
        year (int): The filing year, used as a cache key.

    Returns:
        str: The HTML content of the filing, or None if fetching fails.
    """
    cache = load_cache()
    cache_key = f"{ticker}_{year}"

    if cache_key in cache:
        print("Found filing in cache. Loading from file.")
        return cache[cache_key]

    if not url:
        return None

    print("Filing not in cache. Downloading from SEC...")
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        html_content = response.text

        # Save to cache before returning
        cache[cache_key] = html_content
        save_cache(cache)

        return html_content
    except requests.exceptions.RequestException as e:
        print(f"Error downloading filing: {e}")
        return None


# =============================================================================
# 5. TEXT EXTRACTION - Pulling Information from HTML
# =============================================================================
# This is where we parse the messy HTML to find the sections we care about.


def extract_text_from_html(html_content):
    """
    Uses BeautifulSoup to extract clean, readable text from HTML.

    Args:
        html_content (str): The raw HTML of the filing.

    Returns:
        str: The cleaned text content.
    """
    print("Extracting text from HTML using BeautifulSoup...")
    soup = BeautifulSoup(html_content, "html.parser")

    # Get all text and join with newlines
    text = soup.get_text("\n")

    # Clean up whitespace and non-breaking spaces
    text = re.sub(r"\xa0", " ", text)  # Replace non-breaking spaces
    text = re.sub(r"\s*\n\s*", "\n", text)  # Normalize newlines

    return text


def extract_section(full_text, start_pattern, end_pattern):
    """
    Extracts a specific section from the text of the filing, like "Risk Factors".
    It uses regular expressions to find the start and end of a section.

    Args:
        full_text (str): The entire text of the filing.
        start_pattern (str): A regex pattern to find the start of the section.
        end_pattern (str): A regex pattern to find the start of the *next* section.

    Returns:
        str: The extracted text of the section, or a "not found" message.
    """
    # Use re.IGNORECASE to match "Item 1", "item 1", "ITEM 1", etc.
    start_match = re.search(start_pattern, full_text, re.IGNORECASE)

    if not start_match:
        return "Section not found."

    # Find the start of the *next* item to define the end of our section
    end_match = re.search(end_pattern, full_text[start_match.end() :], re.IGNORECASE)

    if end_match:
        section_text = full_text[
            start_match.start() : start_match.end() + end_match.start()
        ]
    else:
        # If no end pattern is found, take a fixed number of characters
        section_text = full_text[start_match.start() : start_match.start() + 20000]

    return section_text.strip()


# =============================================================================
# 6. SUMMARIZATION - Making Sense of the Text
# =============================================================================
# These functions summarize the long, extracted sections into key points.


def fallback_summarizer(section_text, num_sentences=5):
    """
    A simple summarizer that just extracts the first few sentences.
    This is used if the AI-based summarizer isn't available.
    """
    # Split text into sentences
    sentences = re.split(r"(?<=[.!?])\s+", section_text)

    # Get the first `num_sentences` non-empty sentences
    summary = "\n- ".join([s.strip() for s in sentences if s.strip()][:num_sentences])
    return "- " + summary if summary else "Could not generate a summary."


def summarize_with_llm(section_text):
    """
    Summarizes a section using an external Large Language Model (LLM) like GPT.
    This is an advanced feature and requires an API key.

    Args:
        section_text (str): The text to summarize.

    Returns:
        str: A bullet-point summary from the LLM.
    """
    if not LANGCHAIN_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
        return None  # Signal that LLM is not available

    print("Summarizing with LLM...")

    # The prompt tells the AI exactly what we want it to do.
    prompt_template = """
    You are a financial analyst. Summarize the following section of a 10-K report
    into 5 concise, insightful bullet points. Focus on the most critical information.
    
    Section Text:
    ---
    {text}
    ---
    
    Summary:
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    chain = prompt | llm

    try:
        response = chain.invoke(
            {"text": section_text[:15000]}
        )  # Limit text to avoid high costs
        return response.content
    except Exception as e:
        print(f"LLM summarization failed: {e}")
        return None  # Fallback to simple summarizer if API fails


# =============================================================================
# 7. MAIN APPLICATION LOGIC - Putting It All Together
# =============================================================================
# This is where we define the main workflow of our script.


def run_analyzer():
    """The main function that orchestrates the entire process."""

    # --- Step 1: Get User Input ---
    print("\n--- SEC 10-K Financial Report Analyzer ---")
    ticker = input("Enter a company ticker (e.g., AAPL, GOOGL, MSFT): ").strip().upper()
    year = datetime.now().year - 1  # Default to last year

    # --- Step 2: Fetch Company Info and Filing ---
    cik = get_company_info(ticker)
    if not cik:
        return  # Stop if we can't find the company

    filing_url = get_latest_10k_filing_url(cik)
    if not filing_url:
        return  # Stop if no 10-K is found

    html_content = fetch_filing_html(filing_url, ticker, year)
    if not html_content:
        return  # Stop if download fails

    # --- Step 3: Extract Text and Sections ---
    full_text = extract_text_from_html(html_content)

    print("\nExtracting key sections from the filing...")

    # Regex patterns to find the start of each section.
    # The `.` in "Item 1." is escaped as `\.`
    business_section = extract_section(full_text, r"Item 1\.\s+Business", r"Item 1A\.")
    risk_factors_section = extract_section(
        full_text, r"Item 1A\.\s+Risk Factors", r"Item 1B\."
    )
    mda_section = extract_section(
        full_text, r"Item 7\.\s+Management's Discussion and Analysis", r"Item 7A\."
    )

    # --- Step 4: Summarize and Display Results ---
    sections = {
        "Business Overview (Item 1)": business_section,
        "Risk Factors (Item 1A)": risk_factors_section,
        "Management's Discussion and Analysis (Item 7)": mda_section,
    }

    print("\n--- Analysis Results ---\n")
    for title, text in sections.items():
        print(f"--- {title} ---\n")

        # Try LLM summary first, then use fallback
        summary = summarize_with_llm(text)
        if not summary:
            print("(Using fallback summarizer)")
            summary = fallback_summarizer(text)

        print(summary)
        print("\n" + "=" * 50 + "\n")


# =============================================================================
# 8. SCRIPT EXECUTION - Running the App
# =============================================================================
# This standard Python construct ensures the `run_analyzer` function is called
# only when the script is executed directly (not when imported as a module).

if __name__ == "__main__":
    run_analyzer()

# --- Example of How to Run ---
