import os
import json
import time
from datetime import datetime
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
# LLM-based HTML Content Extraction
# -----------------------------

HTML_EXTRACTION_PROMPT = """
You are an expert HTML content extractor. Extract all relevant data from the provided HTML content and structure it into a clear, organized JSON format.

Focus on identifying and extracting:
1. Document metadata (title, company name, filing type, date)
2. Section headings and their hierarchy
3. Key business sections (Item 1 - Business, Item 1A - Risk Factors, Item 7 - MD&A)
4. Tables with financial data
5. Links and references
6. Text content organized by sections
7. Any embedded structured data

For SEC 10-K filings specifically, pay attention to:
- Item numbers and section titles
- Business description content
- Risk factors enumeration
- Management discussion and analysis
- Financial tables and metrics

Structure the output as a comprehensive JSON object with nested elements representing the document hierarchy. Ensure all text content is clean and properly formatted.

HTML Content:
{html_content}

Return only a valid JSON object with the extracted and structured data:
"""


def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    # Fallback to Ollama if available
    try:
        return ChatOllama(model="llama3.2", temperature=0.1)
    except:
        return None


def extract_html_content_with_llm(html: str):
    """Extract structured content from HTML using LLM"""
    llm = get_llm()
    if not llm:
        return fallback_html_extraction(html)

    # Clean and truncate HTML for LLM processing
    soup = BeautifulSoup(html, "html.parser")
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Get clean text with some HTML structure preserved
    clean_html = str(soup)

    # Truncate if too long (adjust based on LLM context limits)
    if len(clean_html) > 50000:
        clean_html = clean_html[:50000] + "... [truncated]"

    try:
        prompt = ChatPromptTemplate.from_template(HTML_EXTRACTION_PROMPT)
        chain = prompt | llm
        response = chain.invoke({"html_content": clean_html})

        # Extract JSON from response
        content = response.content.strip()

        # Try to find JSON block
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()

        # Parse JSON
        extracted_data = json.loads(content)
        return extracted_data

    except Exception as e:
        print(f"LLM extraction failed: {e}")
        return fallback_html_extraction(html)


def fallback_html_extraction(html: str):
    """Fallback extraction method when LLM is not available"""
    soup = BeautifulSoup(html, "html.parser")

    # Basic extraction without regex
    title = soup.find("title")
    title_text = title.get_text().strip() if title else "Unknown Document"

    # Find all headings
    headings = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        headings.append({"level": int(tag.name[1]), "text": tag.get_text().strip()})

    # Get all text content
    full_text = soup.get_text(separator="\n").strip()

    # Basic section identification
    sections = {"business": "", "risk_factors": "", "mdna": ""}

    # Simple text-based section detection
    text_lower = full_text.lower()

    # Find business section
    business_start = text_lower.find("item 1") and text_lower.find("business")
    if business_start != -1:
        # Get next 5000 characters as a rough section
        sections["business"] = full_text[business_start : business_start + 5000]

    return {
        "metadata": {"title": title_text, "extraction_method": "fallback"},
        "headings": headings,
        "sections": sections,
        "full_text": full_text[:10000],  # Truncate for performance
    }


def extract_sections_from_structured_data(extracted_data):
    """Convert LLM-extracted structured data to the expected sections format"""
    sections = {"Business Overview": "", "Risk Factors": "", "MD&A": ""}

    # Try to find sections in the structured data
    if "sections" in extracted_data:
        for section_key, section_data in extracted_data["sections"].items():
            section_lower = section_key.lower()
            if "business" in section_lower:
                if isinstance(section_data, dict) and "content" in section_data:
                    sections["Business Overview"] = section_data["content"]
                elif isinstance(section_data, str):
                    sections["Business Overview"] = section_data
            elif "risk" in section_lower:
                if isinstance(section_data, dict) and "content" in section_data:
                    sections["Risk Factors"] = section_data["content"]
                elif isinstance(section_data, str):
                    sections["Risk Factors"] = section_data
            elif "md&a" in section_lower or "management" in section_lower:
                if isinstance(section_data, dict) and "content" in section_data:
                    sections["MD&A"] = section_data["content"]
                elif isinstance(section_data, str):
                    sections["MD&A"] = section_data

    # If sections are empty, try to extract from full text
    if not any(sections.values()) and "full_text" in extracted_data:
        full_text = extracted_data["full_text"]
        # Use simple keyword-based extraction as fallback
        sections = simple_section_extraction(full_text)

    return sections


def simple_section_extraction(text: str):
    """Simple keyword-based section extraction as ultimate fallback"""
    sections = {"Business Overview": "", "Risk Factors": "", "MD&A": ""}

    text_lower = text.lower()

    # Find business section
    business_keywords = ["item 1", "business", "overview", "operations"]
    for keyword in business_keywords:
        start = text_lower.find(keyword)
        if start != -1:
            sections["Business Overview"] = text[start : start + 3000]
            break

    # Find risk factors
    risk_keywords = ["risk factors", "item 1a", "risks"]
    for keyword in risk_keywords:
        start = text_lower.find(keyword)
        if start != -1:
            sections["Risk Factors"] = text[start : start + 3000]
            break

    # Find MD&A
    mdna_keywords = ["management's discussion", "item 7", "md&a"]
    for keyword in mdna_keywords:
        start = text_lower.find(keyword)
        if start != -1:
            sections["MD&A"] = text[start : start + 3000]
            break

    return sections


# Updated extract_sections function
def extract_sections(html: str):
    """Main function to extract sections using LLM-based approach"""
    # Extract structured data using LLM
    extracted_data = extract_html_content_with_llm(html)

    # Convert to expected sections format
    sections = extract_sections_from_structured_data(extracted_data)

    return sections


# -----------------------------
# Summarization (keeping existing functions)
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
    return " ".join(text.split()).strip()


def simple_sentence_split(text):
    import re

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
    import re

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
st.title("10-K Financial Report Analyzer (LLM-Enhanced)")

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
    st.write("Extraction Method:", "LLM-Enhanced" if get_llm() else "Fallback")

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
