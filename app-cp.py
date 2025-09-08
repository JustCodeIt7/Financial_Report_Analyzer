import os
import re
import json
import time
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import streamlit as st

# Optional LangChain (summarization). If not available or no API key, fallback summarizer will be used.
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate


# -----------------------------
# Configuration
# -----------------------------
USER_AGENT = (
    "YourName Contact@Email.com"  # Replace with real contact per SEC guidelines
)
SEC_BASE = "https://data.sec.gov"
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


# -----------------------------
# SEC Data Fetching
# -----------------------------
def fetch_ticker_map():
    # Cache locally
    path = CACHE_DIR / "company_tickers.json"
    if path.exists():
        return json.loads(path.read_text())
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers={"User-Agent": USER_AGENT})
    data = r.json()
    path.write_text(json.dumps(data))
    return data


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
# Filing Caching
# -----------------------------
def get_or_fetch_filing(ticker: str, year: int):
    cik = cik_from_ticker(ticker)
    submissions = get_company_submissions(cik)
    acc, doc = find_10k_for_year(submissions, year)
    url = accession_to_url(acc, cik, doc)
    html = download_filing_html(url)

    return filing_id, html, acc, cik


# -----------------------------
# Section Extraction
# -----------------------------
ITEM_PATTERNS = {
    "business": r"item\s+1\.*\s*(business)",
    "risk_factors": r"item\s+1a\.*\s*(risk factors?)",
    "mdna": r"item\s+7\.*\s*(management['’]s discussion and analysis|management.?s discussion and analysis)",
}

STOP_ITEM_REGEX = r"item\s+([1-9][0-9]?[a-z]?)\."  # generic


def clean_text(text: str):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_sections(html: str):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")
    # Normalize
    norm = re.sub(r"\xa0", " ", text)
    lines = norm.splitlines()
    joined = "\n".join([l for l in lines if l.strip()])

    lower = joined.lower()

    def find_section(item_regex):
        pattern = re.compile(item_regex, re.IGNORECASE)
        matches = list(pattern.finditer(lower))
        if not matches:
            return ""
        start = matches[0].start()
        # Find next item after start
        stop_pattern = re.compile(STOP_ITEM_REGEX, re.IGNORECASE)
        following = list(stop_pattern.finditer(lower, start + 10))
        end = len(joined)
        for m in following:
            # ensure it's a different item number, not the same occurrence or sub-head (like 7A)
            if m.start() > start:
                end = m.start()
                break
        section_raw = joined[start:end]
        return clean_text(section_raw)

    sections = {
        "Business Overview": find_section(ITEM_PATTERNS["business"]),
        "Risk Factors": find_section(ITEM_PATTERNS["risk_factors"]),
        "MD&A": find_section(ITEM_PATTERNS["mdna"]),
    }
    return sections


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
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    llm = ChatOllama(model="llama3.2", temperature=0.2)
    return llm


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
        try:
            return json.loads(m.group(0))
        except:
            return None
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
        if not text:
            results[name] = {"bullets": "- (section not found)"}
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
st.title("10-K Financial Report Analyzer (Simplified Demo)")

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
    st.write("OpenAI Key:", "Yes" if os.getenv("OPENAI_API_KEY") else "No")

if run:
    st.write(f"Fetching {ticker} {year} 10-K ...")
    filing_id, html, accession, cik = get_or_fetch_filing(ticker, int(year))
    st.success(f"Filing Accession: {accession} (CIK {cik})")
    with st.expander("Raw HTML (truncated)"):
        st.code(html[:5000])

    st.write("Extracting key sections...")
    sections = extract_sections(html)
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
