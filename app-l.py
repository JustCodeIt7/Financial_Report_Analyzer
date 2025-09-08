import os
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
import json
import re


# Function to get CIK from ticker
def get_cik(ticker):
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "myemail@example.com"}
    response = requests.get(url, headers=headers)
    tickers = response.json()
    for key, value in tickers.items():
        if value["ticker"] == ticker.upper():
            return str(value["cik_str"]).zfill(10)
    return None


# Function to get 10-K URL for ticker and year (or latest if year=None)
def get_10k_url(cik, year=None):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {"User-Agent": "myemail@example.com"}
    response = requests.get(url, headers=headers)
    data = response.json()
    filings = data["filings"]["recent"]
    for i in range(len(filings["form"])):
        if filings["form"][i] == "10-K":
            filing_year = int(filings["filingDate"][i][:4])
            if year is None or filing_year == year:
                acc_num = filings["accessionNumber"][i]
                doc = filings["primaryDocument"][i]
                cik_num = int(cik)
                return (
                    f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc_num.replace('-', '')}/{doc}",
                    filing_year,
                )
    return None, None


# Function to fetch HTML
def fetch_10k_html(ticker, year=None):
    cik = get_cik(ticker)
    url, _ = get_10k_url(cik, year)
    if not url:
        return None
    headers = {"User-Agent": "myemail@example.com"}
    response = requests.get(url, headers=headers)
    return response.text


def extract_full_text(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()


# Setup LangChain for summarization
def setup_llm_chain(template):
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo")
    # llm = ChatOllama(model="llama3.2", temperature=0.3)
    prompt = PromptTemplate(input_variables=["text"], template=template)
    return LLMChain(llm=llm, prompt=prompt)


# Streamlit UI
st.title("10-K Financial Report Analyzer")

# api_key = st.sidebar.text_input("OpenAI API Key", type="password")
ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL)")
year = st.number_input(
    "Enter Year (optional, leave 0 for latest)", min_value=0, value=0
)
year = year if year > 0 else None

if st.button("Analyze"):
    if ticker:
        html = fetch_10k_html(ticker, year)
        if html:
            full_text = extract_full_text(html)

            # Investment thesis
            thesis_template = "Write a 1-paragraph investment thesis based on the following 10-K filing: {text}"
            thesis_chain = setup_llm_chain(thesis_template)
            thesis = thesis_chain.invoke({"text": full_text})

            # Top 5 risks
            risk_template = "Extract the top 5 risks from the following text, each with a severity (low, medium, high): {text}"
            risk_chain = setup_llm_chain(risk_template)
            top_risks = risk_chain.invoke({"text": full_text})

            # Display outputs
            st.subheader("Investment Thesis")
            st.write(thesis)

            st.subheader("Top 5 Risks")
            st.write(top_risks)

            # Generate Markdown
            md_content = f"# 10-K Analysis for {ticker.upper()}\n\n"
            md_content += f"## Investment Thesis\n{thesis}\n\n"
            md_content += f"## Top 5 Risks\n{top_risks}\n"

            st.download_button(
                "Download Markdown", md_content, file_name=f"{ticker}_10K_analysis.md"
            )
        else:
            st.error("Could not retrieve 10-K filing. Check ticker and year.")
