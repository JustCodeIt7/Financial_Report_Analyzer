import requests
from bs4 import BeautifulSoup
import sqlite3
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
import json
import re

# Setup SQLite cache
conn = sqlite3.connect("cache.db")
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS filings
             (ticker TEXT, year INTEGER, html TEXT, PRIMARY KEY (ticker, year))""")
conn.commit()


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


# Function to fetch HTML, with cache
def fetch_10k_html(ticker, year=None):
    c.execute(
        "SELECT html FROM filings WHERE ticker=? AND year=?", (ticker.upper(), year)
    )
    row = c.fetchone()
    if row:
        return row[0]
    cik = get_cik(ticker)
    url, fetched_year = get_10k_url(cik, year)
    headers = {"User-Agent": "myemail@example.com"}
    response = requests.get(url, headers=headers)
    html = response.text
    year_to_cache = year if year else fetched_year
    c.execute(
        "INSERT INTO filings (ticker, year, html) VALUES (?, ?, ?)",
        (ticker.upper(), year_to_cache, html),
    )
    conn.commit()
    return html


# Function to extract sections from HTML
def extract_sections(html):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()
    sections = {}

    # Simple regex to find sections
    business_pattern = re.search(r"Item 1\.\s*Business", text, re.IGNORECASE)
    risk_pattern = re.search(r"Item 1A\.\s*Risk Factors", text, re.IGNORECASE)
    mda_pattern = re.search(
        r"Item 7\.\s*Management.s Discussion and Analysis", text, re.IGNORECASE
    )

    if business_pattern:
        start = business_pattern.end()
        end = risk_pattern.start() if risk_pattern else len(text)
        sections["Business Overview"] = text[start:end].strip()[:5000]  # Limit length

    if risk_pattern:
        start = risk_pattern.end()
        end = mda_pattern.start() if mda_pattern else len(text)
        sections["Risk Factors"] = text[start:end].strip()[:5000]

    if mda_pattern:
        start = mda_pattern.end()
        end = len(text)  # Assume end of doc
        sections["MD&A"] = text[start:end].strip()[:5000]

    return sections


# Setup LangChain for summarization
def setup_llm_chain(api_key, template):
    llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo")
    prompt = PromptTemplate(input_variables=["text"], template=template)
    return LLMChain(llm=llm, prompt=prompt)


# Streamlit UI
st.title("10-K Financial Report Analyzer")

api_key = st.sidebar.text_input("OpenAI API Key", type="password")
ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL)")
year = st.number_input(
    "Enter Year (optional, leave 0 for latest)", min_value=0, value=0
)
year = year if year > 0 else None

if st.button("Analyze"):
    if not api_key:
        st.warning("Please enter OpenAI API Key")
    else:
        html = fetch_10k_html(ticker, year)
        sections = extract_sections(html)

        # Summarize each section into 10 bullets
        bullet_template = (
            "Summarize the following text into exactly 10 bullet points: {text}"
        )
        bullet_chain = setup_llm_chain(api_key, bullet_template)

        outputs = {}
        for section, text in sections.items():
            if text:
                summary = bullet_chain.run(text=text)
                outputs[section] = summary

        # Top 5 risks
        risk_template = "Extract the top 5 risks from the following text, each with a severity (low, medium, high): {text}"
        risk_chain = setup_llm_chain(api_key, risk_template)
        top_risks = ""
        if "Risk Factors" in sections:
            top_risks = risk_chain.run(text=sections["Risk Factors"])

        # 1-paragraph thesis
        thesis_template = "Write a 1-paragraph investment thesis based on the following sections: {text}"
        thesis_chain = setup_llm_chain(api_key, thesis_template)
        all_text = json.dumps(sections)
        thesis = thesis_chain.run(text=all_text)

        # Display outputs
        for section, bullets in outputs.items():
            st.subheader(section)
            st.write(bullets)

        st.subheader("Top 5 Risks")
        st.write(top_risks)

        st.subheader("Investment Thesis")
        st.write(thesis)

        # Generate Markdown
        md_content = f"# 10-K Analysis for {ticker.upper()}\n\n"
        for section, bullets in outputs.items():
            md_content += f"## {section}\n{bullets}\n\n"
        md_content += f"## Top 5 Risks\n{top_risks}\n\n"
        md_content += f"## Investment Thesis\n{thesis}\n"

        st.download_button(
            "Download Markdown", md_content, file_name=f"{ticker}_10K_analysis.md"
        )
