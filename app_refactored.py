# =========================================================
# STEP 1. Imports & Optional Dependencies
# =========================================================
import os
import re
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, List, Optional, Any

import requests
from bs4 import BeautifulSoup
import streamlit as st

# Optional LangChain wrappers; handle absence gracefully for tutorial clarity
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None  # type: ignore

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None  # type: ignore

try:
    from langchain.prompts import ChatPromptTemplate
except ImportError:
    ChatPromptTemplate = None  # type: ignore


# =========================================================
# STEP 2. Configuration Constants
# =========================================================
USER_AGENT = (
    "YourName Contact@Email.com"  # Replace with real contact per SEC fair-use guidance
)
SEC_BASE = "https://data.sec.gov"

# Regex patterns for section extraction
ITEM_PATTERNS = {
    "business": r"item\s+1\.*\s*(business)",
    "risk_factors": r"item\s+1a\.*\s*(risk factors?)",
    "mdna": r"item\s+7\.*\s*(management['’]s discussion and analysis|management.?s discussion and analysis)",
}
STOP_ITEM_REGEX = r"item\s+([1-9][0-9]?[a-z]?)\."

# Prompt templates (kept as raw multi-line strings for readability)
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


# =========================================================
# STEP 3. Data Structures
# =========================================================
@dataclass
class FilingReference:
    """Holds minimal identity metadata for a filing."""

    cik: str
    accession: str
    primary_doc: str
    url: str


@dataclass
class UIState:
    """Captured input parameters from the sidebar."""

    ticker: str
    year: int
    run: bool
    has_openai_key: bool


# =========================================================
# STEP 4. Simple Utility Functions
# =========================================================
def clean_text(text: str) -> str:
    """Normalize whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def simple_sentence_split(text: str) -> List[str]:
    """Naive sentence splitter for fallback summarization."""
    return re.split(r"(?<=[.!?])\s+", text)


# =========================================================
# STEP 5. SEC Data Fetching & Caching Layer
# =========================================================
@st.cache_data(show_spinner=False)
def fetch_ticker_map() -> Dict[str, Any]:
    """Retrieve the master ticker->CIK mapping JSON from SEC."""
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    return r.json()


def cik_from_ticker(ticker: str) -> Optional[str]:
    """Lookup CIK (zero-padded) for a given ticker symbol."""
    data = fetch_ticker_map()
    t = ticker.upper()
    for entry in data.values():
        if entry.get("ticker", "").upper() == t:
            return str(entry["cik_str"]).zfill(10)
    return None


@st.cache_data(show_spinner=False)
def get_company_submissions(cik: str) -> Dict[str, Any]:
    """Fetch submissions JSON for a given CIK."""
    url = f"{SEC_BASE}/submissions/CIK{cik}.json"
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    return r.json()


def find_10k_for_year(
    submissions: dict, year: int
) -> Tuple[Optional[str], Optional[str]]:
    """Locate the accession number & primary document for the desired year's 10-K."""
    filings = submissions.get("filings", {}).get("recent", {})
    for form, date, acc, doc in zip(
        filings.get("form", []),
        filings.get("filingDate", []),
        filings.get("accessionNumber", []),
        filings.get("primaryDocument", []),
    ):
        if form == "10-K" and date.startswith(str(year)):
            return acc, doc
    return None, None


def accession_to_url(acc: str, cik: str, primary_doc: str) -> str:
    """Construct the direct SEC archive URL to the filing's primary document."""
    acc_nodashes = acc.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodashes}/{primary_doc}"


def download_filing_html(url: str) -> str:
    """Download the raw HTML for a filing."""
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    return r.text


def get_filing(ticker: str, year: int) -> Tuple[str, FilingReference]:
    """High-level convenience: from ticker+year -> Filing HTML & metadata."""
    cik = cik_from_ticker(ticker)
    if not cik:
        raise ValueError(f"Unable to find CIK for ticker '{ticker}'.")
    submissions = get_company_submissions(cik)
    acc, doc = find_10k_for_year(submissions, year)
    if not acc or not doc:
        raise ValueError(f"No 10-K filing found for {ticker} in {year}.")
    url = accession_to_url(acc, cik, doc)
    html = download_filing_html(url)
    return html, FilingReference(cik=cik, accession=acc, primary_doc=doc, url=url)


# =========================================================
# STEP 6. Section Extraction Logic
# =========================================================
def extract_sections(html: str) -> Dict[str, str]:
    """
    Parse the filing HTML and heuristically extract select sections.
    NOTE: Real-world production systems should use more robust parsing (XBRL / structured docs).
    """
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")
    normalized = re.sub(r"\xa0", " ", text)
    lines = [l for l in normalized.splitlines() if l.strip()]
    joined = "\n".join(lines)
    lowered = joined.lower()

    def slice_section(pattern: str) -> str:
        """Capture text between an 'Item X' marker and the next 'Item Y' marker."""
        compiled = re.compile(pattern, re.IGNORECASE)
        matches = list(compiled.finditer(lowered))
        if not matches:
            return ""
        start = matches[0].start()

        stop_pattern = re.compile(STOP_ITEM_REGEX, re.IGNORECASE)
        following = list(stop_pattern.finditer(lowered, start + 10))
        end = len(joined)
        for m in following:
            if m.start() > start:
                end = m.start()
                break
        section_raw = joined[start:end]
        return clean_text(section_raw)

    return {
        "Business Overview": slice_section(ITEM_PATTERNS["business"]),
        "Risk Factors": slice_section(ITEM_PATTERNS["risk_factors"]),
        "MD&A": slice_section(ITEM_PATTERNS["mdna"]),
    }


# =========================================================
# STEP 7. Summarization Helpers (Fallback + Heuristics)
# =========================================================
def fallback_bullets(section_text: str, count: int = 10) -> str:
    """Create simple bullets from the first N sentences if no LLM available."""
    sentences = [
        clean_text(s) for s in simple_sentence_split(section_text) if s.strip()
    ]
    chosen = sentences[:count]
    bullets = ["- " + s[:180] for s in chosen]
    while len(bullets) < count:
        bullets.append("- (insufficient content)")
    return "\n".join(bullets[:count])


def heuristic_risks(risk_text: str, top: int = 5) -> List[Dict[str, str]]:
    """
    Crude heuristic risk extraction based on keyword scoring.
    Produces a list of risk objects: title, description, severity.
    """
    paragraphs = [p for p in risk_text.split("\n") if len(p.split()) > 8]
    scored: List[Tuple[int, str]] = []
    for p in paragraphs:
        lw = p.lower()
        score = sum(
            2
            for kw in ["significant", "material", "adverse", "substantial", "severe"]
            if kw in lw
        )
        score += sum(1 for kw in ["may", "could", "risk"] if kw in lw)
        scored.append((score, p))
    scored.sort(reverse=True, key=lambda x: x[0])
    risks: List[Dict[str, str]] = []
    for score, para in scored[:top]:
        sev = "High" if score >= 5 else "Medium" if score >= 3 else "Low"
        title = " ".join(para.split()[:6])
        risks.append(
            {"title": title, "description": clean_text(para)[:160], "severity": sev}
        )
    return risks


# =========================================================
# STEP 8. LLM Integration (Optional)
# =========================================================
def get_llm():
    """
    Try to return an LLM client (OpenAI or Ollama) if available.
    Priority or selection logic can be adjusted easily for tutorial demonstration.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    llm = ChatOllama(model="llama3.2", temperature=0.2)
    return llm


def llm_invoke_single_var(
    llm, template: str, var_name: str, content: str
) -> Optional[str]:
    """Helper to run a one-variable prompt template."""
    if not llm or ChatPromptTemplate is None:
        return None
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    try:
        out = chain.invoke({var_name: content})
        return out.content.strip()
    except Exception:
        return None


def llm_summarize_section(section_text: str) -> Optional[str]:
    return llm_invoke_single_var(
        get_llm(), DEFAULT_BULLET_PROMPT, "section_text", section_text
    )


def llm_extract_risks(risk_text: str) -> Optional[List[Dict[str, str]]]:
    """Attempt structured risk extraction via LLM returning JSON."""
    raw = llm_invoke_single_var(get_llm(), RISK_PROMPT, "risk_text", risk_text)
    if not raw:
        return None
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        if isinstance(parsed, list):
            return parsed
    except Exception:
        return None
    return None


def llm_write_thesis(
    bullets_json: Dict[str, str], risks_json: List[Dict[str, str]]
) -> Optional[str]:
    if not ChatPromptTemplate:
        return None
    llm = get_llm()
    if not llm:
        return None
    prompt = ChatPromptTemplate.from_template(THESIS_PROMPT)
    chain = prompt | llm
    try:
        out = chain.invoke(
            {
                "bullets_json": json.dumps(bullets_json, indent=2),
                "risks_json": json.dumps(risks_json, indent=2),
            }
        )
        return out.content.strip()
    except Exception:
        return None


# =========================================================
# STEP 9. High-Level Summarization Orchestrator
# =========================================================
def summarize_sections(
    sections: Dict[str, str],
) -> Tuple[Dict[str, Dict[str, str]], List[Dict[str, str]], str]:
    """
    For each extracted section:
      - Attempt LLM bullet summary; fallback to heuristic bullet creation.
      - For Risk Factors, attempt LLM JSON extraction; fallback to heuristic risk scoring.
      - Build an overall thesis paragraph.
    """
    results: Dict[str, Dict[str, str]] = {}
    risks_struct: List[Dict[str, str]] = []

    for name, text in sections.items():
        if not text:
            results[name] = {"bullets": "- (section not found)"}
            continue

        bullets = llm_summarize_section(text)
        if not bullets:
            bullets = fallback_bullets(text)

        results[name] = {"bullets": bullets}

        if name == "Risk Factors":
            parsed_risks = llm_extract_risks(text)
            if not parsed_risks:
                parsed_risks = heuristic_risks(text)
            risks_struct = parsed_risks

    thesis = llm_write_thesis(
        {k: v["bullets"] for k, v in results.items()}, risks_struct
    )
    if not thesis:
        thesis = "Thesis: Company exhibits identifiable risks and strategic drivers; deeper LLM analysis requires API key or local model."
    return results, risks_struct, thesis


# =========================================================
# STEP 10. Markdown Export
# =========================================================
def build_markdown(
    ticker: str,
    year: int,
    filing: FilingReference,
    summaries: Dict[str, Dict[str, str]],
    risks: List[Dict[str, str]],
    thesis: str,
) -> str:
    """Assemble a Markdown report for download or sharing."""
    md = []
    md.append(f"# {ticker} {year} 10-K Highlights")
    md.append(f"*Accession:* {filing.accession}  |  *CIK:* {filing.cik}")
    md.append(f"*Source URL:* {filing.url}")
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


# =========================================================
# STEP 11. Streamlit UI Components
# =========================================================
def configure_page():
    """Initial page setup for Streamlit."""
    st.set_page_config(page_title="10-K Analyzer", layout="wide")
    st.title("10-K Financial Report Analyzer (Refactored Tutorial Version)")
    st.caption(
        "Educational example: Rapid filing section extraction + heuristic / LLM summaries."
    )


def sidebar_inputs() -> UIState:
    """Render sidebar inputs and return the collected UI state."""
    with st.sidebar:
        st.header("Input")
        ticker = st.text_input("Ticker", value="AAPL").upper().strip()
        year = st.number_input(
            "Filing Year",
            min_value=2005,
            max_value=datetime.now().year,
            value=datetime.now().year - 1,
        )
        run = st.button("Fetch & Analyze")
        st.markdown("---")
        has_key = bool(os.getenv("OPENAI_API_KEY"))
        st.write("OpenAI Key:", "✅" if has_key else "❌")
        return UIState(ticker=ticker, year=int(year), run=run, has_openai_key=has_key)


def render_sections_preview(sections: Dict[str, str], truncate: int = 1500):
    """Show truncated raw section text for transparency before summarization."""
    col1, col2, col3 = st.columns(3)
    col1.subheader("Business Overview")
    col1.write(
        sections["Business Overview"][:truncate]
        + ("..." if len(sections["Business Overview"]) > truncate else "")
    )
    col2.subheader("Risk Factors")
    col2.write(
        sections["Risk Factors"][:truncate]
        + ("..." if len(sections["Risk Factors"]) > truncate else "")
    )
    col3.subheader("MD&A")
    col3.write(
        sections["MD&A"][:truncate]
        + ("..." if len(sections["MD&A"]) > truncate else "")
    )


def render_summaries(summaries: Dict[str, Dict[str, str]]):
    """Display bullet summaries for each section."""
    st.header("Section Summaries")
    for section, data in summaries.items():
        st.subheader(section)
        st.text(data["bullets"])


def render_risks(risks: List[Dict[str, str]]):
    """Display structured risk output."""
    st.header("Top 5 Risks")
    if not risks:
        st.info("No risks extracted.")
        return
    for r in risks:
        st.markdown(f"- **{r['title']}** ({r['severity']}): {r['description']}")


def render_thesis(thesis: str):
    """Display synthesized investment thesis."""
    st.header("Investment Thesis")
    st.write(thesis)


# =========================================================
# STEP 12. Orchestrating the App Logic
# =========================================================
def run_analysis(ui: UIState):
    """Execute the full fetch -> extract -> summarize -> display pipeline."""
    try:
        st.write(f"Fetching {ui.ticker} {ui.year} 10-K ...")
        html, filing_ref = get_filing(ui.ticker, ui.year)
        st.success(f"Filing Accession: {filing_ref.accession} (CIK {filing_ref.cik})")
        with st.expander("Raw HTML (first 5000 chars)"):
            st.code(html[:5000])

        st.write("Extracting key sections...")
        sections = extract_sections(html)
        render_sections_preview(sections)

        st.write("Summarizing (LLM if available, else fallback heuristics)...")
        start = time.time()
        summaries, risks, thesis = summarize_sections(sections)
        elapsed = time.time() - start
        st.info(f"Summarization complete in {elapsed:.2f} seconds.")

        render_summaries(summaries)
        render_risks(risks)
        render_thesis(thesis)

        md_content = build_markdown(
            ui.ticker, ui.year, filing_ref, summaries, risks, thesis
        )
        st.download_button(
            "Download Markdown Report",
            data=md_content,
            file_name=f"{ui.ticker}_{ui.year}_10K_summary.md",
            mime="text/markdown",
        )
        with st.expander("Generated Markdown"):
            st.code(md_content)

    except Exception as e:
        st.error(f"Error: {e}")


# =========================================================
# STEP 13. Entry Point
# =========================================================
def main():
    """
    Main entry point for:
      - Local script execution: python streamlit_10k_app.py
      - Streamlit execution: streamlit run streamlit_10k_app.py
    """
    configure_page()
    ui_state = sidebar_inputs()
    if ui_state.run:
        run_analysis(ui_state)
    else:
        st.info("Enter a ticker & year, then click 'Fetch & Analyze' to begin.")


# Standard Python entry guard.
if __name__ == "__main__":
    # NOTE: When running via `streamlit run`, Streamlit executes the file top-to-bottom.
    # The guard still allows conventional script execution if needed.
    main()
