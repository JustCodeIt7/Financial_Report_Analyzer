import os
import re
import json
import time
from datetime import datetime
import requests
import streamlit as st

# Optional LangChain (summarization). If not available or no API key, fallback summarizer will be used.
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# -----------------------------
# Configuration
# -----------------------------
USER_AGENT = (
    "YourName Contact@Email.com"  # Replace with real contact per SEC guidelines
)
SEC_BASE = "https://data.sec.gov"


# -----------------------------
# SEC Data Fetching
# -----------------------------
def fetch_ticker_map():
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers={"User-Agent": USER_AGENT})
    return r.json()


def cik_from_ticker(ticker: str):
    ticker = ticker.upper()
    data = fetch_ticker_map()
    for entry in data.values():
        if entry["ticker"].upper() == ticker:
            return str(entry["cik_str"]).zfill(10)
    return None


def get_company_submissions(cik: str):
    url = f"{SEC_BASE}/submissions/CIK{cik}.json"
    r = requests.get(url, headers={"User-Agent": USER_AGENT})
    return r.json()


def find_10k_for_year(submissions: dict, year: int):
    filings = submissions.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    dates = filings.get("filingDate", [])
    acc_nums = filings.get("accessionNumber", [])
    primary_docs = filings.get("primaryDocument", [])
    for form, date, acc, doc in zip(forms, dates, acc_nums, primary_docs):
        if form == "10-K" and date.startswith(str(year)):
            return acc, doc
    return None, None


def accession_to_url(acc: str, cik: str, primary_doc: str):
    acc_nodashes = acc.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodashes}/{primary_doc}"


def download_filing_html(url: str):
    r = requests.get(url, headers={"User-Agent": USER_AGENT})
    return r.text


# -----------------------------
# Section Extraction
# -----------------------------
HTML_EXTRACTION_PROMPT = """
From the provided HTML of a 10-K filing, extract the full text for the following sections:
- Item 1: Business
- Item 1A: Risk Factors
- Item 7: Management's Discussion and Analysis of Financial Condition and Results of Operations (MD&A)

Identify the start of each section by looking for patterns like "Item 1.", "Item 1A.", and "Item 7.".
Extract all text content belonging to that section until the start of the next major "Item".

Return the extracted content in a single JSON object with the following keys:
"Business Overview", "Risk Factors", "MD&A".

HTML Content:
{html_content}
"""


def extract_sections_with_llm(html: str):
    """
    Extracts key sections from 10-K HTML using an LLM.
    """
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(HTML_EXTRACTION_PROMPT)
    parser = JsonOutputParser()
    chain = prompt | llm | parser

    # To avoid exceeding token limits, we can truncate the HTML.
    # The key sections are usually in the first part of the document.
    max_chars = 300000  # A reasonable limit, can be adjusted
    truncated_html = html[:max_chars]

    try:
        extracted_json = chain.invoke({"html_content": truncated_html})
        # Ensure all keys are present
        for key in ["Business Overview", "Risk Factors", "MD&A"]:
            if key not in extracted_json:
                extracted_json[key] = f"(LLM failed to extract section: {key})"
        return extracted_json
    except Exception as e:
        st.error(f"Failed to extract sections with LLM: {e}")
        return {
            "Business Overview": "(Extraction Failed)",
            "Risk Factors": "(Extraction Failed)",
            "MD&A": "(Extraction Failed)",
        }


# -----------------------------
# Summarization
# -----------------------------
DEFAULT_BULLET_PROMPT = """
You are an analyst. Summarize the following 10-K section into exactly 10 concise, insight-rich bullets.
Focus on: business model, growth drivers, competitive positioning, financial highlights, strategic priorities.
Avoid boilerplate. Each bullet <= 25 words.
Section:
{section_text}
Return bullets as a plain list prefixed with '- '.
"""

RISK_PROMPT = """
Extract the 5 most material distinct risks from the Risk Factors section below.
For each risk produce:
- Short title (max 8 words)
- One-line description (<=25 words)
- Severity (High, Medium, Low) – judge from wording ('significant', 'material', 'could adversely', etc.)
Return JSON list with objects: title, description, severity.
Section:
{risk_text}
"""

THESIS_PROMPT = """
Write one investment thesis paragraph (max 120 words) synthesizing the company's position, growth levers, key risks, and outlook based on the extracted summaries.
Use neutral, professional tone.
Bullets Data:
{bullets_json}
Risks Data:
{risks_json}
"""


def clean_text(text: str):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def simple_sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text)


def fallback_bullets(section_text, count=10):
    sentences = simple_sentence_split(section_text)[:count]
    bullets = []
    for s in sentences:
        s = clean_text(s)
        if s:
            bullets.append("- " + s[:180])
    while len(bullets) < count:
        bullets.append("- (insufficient content)")
    return "\n".join(bullets[:count])


def heuristic_risks(risk_text, top=5):
    paragraphs = [p for p in risk_text.split("\n") if len(p.split()) > 8]
    scored = []
    for p in paragraphs:
        score = 0
        lw = p.lower()
        for kw in ["significant", "material", "adverse", "substantial", "severe"]:
            if kw in lw:
                score += 2
        for kw in ["may", "could", "risk"]:
            if kw in lw:
                score += 1
        scored.append((score, p))
    scored.sort(reverse=True, key=lambda x: x[0])
    risks = []
    for score, para in scored[:top]:
        sev = "High" if score >= 5 else "Medium" if score >= 3 else "Low"
        title = " ".join(para.split()[:6])
        risks.append(
            {"title": title, "description": clean_text(para)[:160], "severity": sev}
        )
    return risks


def get_llm():
    # Check if the Ollama service is available
    ollama_llm = ChatOllama(model="phi4-mini", temperature=0)
    ollama_llm.invoke("test")  # A quick check
    return ollama_llm


def llm_summarize(section_text, prompt_template):
    llm = get_llm()
    if not llm:
        return None
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    out = chain.invoke({"section_text": section_text})
    return out.content.strip()


def llm_json(section_text, prompt_template):
    llm = get_llm()
    if not llm:
        return None
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    out = chain.invoke({"risk_text": section_text})
    txt = out.content.strip()
    # Try to extract JSON block
    m = re.search(r"\[.*\]", txt, re.DOTALL)
    if m:
        return json.loads(m.group(0))

    return None


def llm_thesis(bullets_json, risks_json):
    llm = get_llm()
    if not llm:
        return None
    prompt = ChatPromptTemplate.from_template(THESIS_PROMPT)
    chain = prompt | llm
    out = chain.invoke(
        {
            "bullets_json": json.dumps(bullets_json, indent=2),
            "risks_json": json.dumps(risks_json, indent=2),
        }
    )
    return out.content.strip()


def summarize_sections(sections):
    results = {}
    risks_struct = []
    for name, text in sections.items():
        if not text or "(Extraction Failed)" in text or "(section not found)" in text:
            results[name] = {"bullets": "- (section not found or extraction failed)"}
            continue

        bullets = llm_summarize(text, DEFAULT_BULLET_PROMPT)
        if not bullets:
            bullets = fallback_bullets(text)
        results[name] = {"bullets": bullets}

        if name == "Risk Factors":
            parsed = llm_json(text, RISK_PROMPT)
            if not parsed:
                parsed = heuristic_risks(text)
            risks_struct = parsed

    thesis = llm_thesis({k: v["bullets"] for k, v in results.items()}, risks_struct)
    if not thesis:
        thesis = "Thesis: Company exhibits identifiable risks and strategic drivers; deeper LLM analysis requires API key."

    return results, risks_struct, thesis


# -----------------------------
# Markdown Export
# -----------------------------
def build_markdown(ticker, year, accession, summaries, risks, thesis):
    md = []
    md.append(f"# {ticker} {year} 10-K Highlights")
    md.append(f"*Accession:* {accession}")
    md.append("")
    for section, data in summaries.items():
        md.append(f"## {section}")
        md.append(data["bullets"])
        md.append("")
    md.append("## Top 5 Risks")
    for r in risks:
        md.append(f"- **{r['title']}** ({r['severity']}): {r['description']}")
    md.append("")
    md.append("## Investment Thesis")
    md.append(thesis)
    return "\n".join(md)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="10-K Analyzer", layout="wide")
st.title("10-K Financial Report Analyzer (LLM-Powered)")

with st.sidebar:
    st.markdown("### Input")
    ticker = st.text_input("Ticker", value="AAPL").upper()
    year = st.number_input(
        "Filing Year",
        min_value=2005,
        max_value=datetime.now().year,
        value=datetime.now().year - 1,
    )
    run = st.button("Fetch & Analyze")
    st.markdown("---")
    st.markdown("Environment:")
    st.write("LLM Available:", "Yes" if get_llm() else "No")


if run:
    st.write(f"Fetching {ticker} {year} 10-K ...")
    cik = cik_from_ticker(ticker)
    if not cik:
        st.error(f"Could not find CIK for ticker {ticker}.")
        st.stop()

    submissions = get_company_submissions(cik)
    accession, doc = find_10k_for_year(submissions, int(year))
    if not accession:
        st.error(f"Could not find 10-K for {ticker} in {year}.")
        st.stop()

    st.success(f"Filing Accession: {accession} (CIK {cik})")
    url = accession_to_url(accession, cik, doc)
    html = download_filing_html(url)
    with st.expander("Raw HTML (truncated)"):
        st.code(html[:5000])

    st.write("Extracting key sections using LLM...")
    sections = extract_sections_with_llm(html)

    col1, col2, col3 = st.columns(3)
    col1.subheader("Business Overview")
    col1.write(
        sections["Business Overview"][:1500]
        + ("..." if len(sections["Business Overview"]) > 1500 else "")
    )
    col2.subheader("Risk Factors")
    col2.write(
        sections["Risk Factors"][:1500]
        + ("..." if len(sections["Risk Factors"]) > 1500 else "")
    )
    col3.subheader("MD&A")
    col3.write(
        sections["MD&A"][:1500] + ("..." if len(sections["MD&A"]) > 1500 else "")
    )

    st.write("Summarizing (may use LLM if available)...")
    start = time.time()
    summaries, risks, thesis = summarize_sections(sections)
    duration = time.time() - start
    st.info(f"Summarization complete in {duration:.2f}s")

    st.header("Section Summaries")
    for section, data in summaries.items():
        st.subheader(section)
        st.text(data["bullets"])

    st.header("Top 5 Risks")
    for r in risks:
        st.markdown(f"- **{r['title']}** ({r['severity']}): {r['description']}")

    st.header("Investment Thesis")
    st.write(thesis)

    md_content = build_markdown(ticker, year, accession, summaries, risks, thesis)
    st.download_button(
        "Download Markdown",
        data=md_content,
        file_name=f"{ticker}_{year}_10K_summary.md",
        mime="text/markdown",
    )

    with st.expander("Generated Markdown"):
        st.code(md_content)
