# 10-K Analyzer (Simple Tutorial Version)
# Goal: Minimal, ~under 500 lines, few moving parts, easy to explain in a YouTube video.
# Features:
# - Input: Ticker + Year
# - Fetch latest (or that year) 10-K from SEC
# - Extract 3 sections: Business (Item 1), Risk Factors (Item 1A), MD&A (Item 7)
# - Summarize each into 10 bullets (simple heuristic or optional OpenAI)
# - Derive Top 5 Risks + severity (heuristic)
# - Generate 1-paragraph thesis
# - Export Markdown
# - Basic caching (files in .cache/)
# - Light SEC rate limiting (sleep)
#
# NOTE: Intentionally minimal: little error handling, naive regex, heuristic summarization.
# Good enough for teaching; can be improved later.

import os, re, time, json, requests, streamlit as st
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import tool
from langchain_openai import ChatOpenAI


# ----------------------------- CONFIG ---------------------------------
USER_AGENT = os.getenv("SEC_USER_AGENT", "Tutorial10KAnalyzer (email@example.com)")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")  # Optional
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# ----------------------------- SIMPLE RATE LIMIT ----------------------
_last_req = 0.0


def sec_get(url):
    global _last_req
    # Conservative: 5 requests / second max
    elapsed = time.time() - _last_req
    if elapsed < 0.2:
        time.sleep(0.2 - elapsed)
    _last_req = time.time()
    r = requests.get(url, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r


# ----------------------------- TICKER -> CIK --------------------------
TICKER_MAP_FILE = CACHE_DIR / "company_tickers.json"


def load_ticker_map():
    if TICKER_MAP_FILE.exists():
        return json.loads(TICKER_MAP_FILE.read_text())
    url = "https://www.sec.gov/files/company_tickers.json"
    data = sec_get(url).json()
    TICKER_MAP_FILE.write_text(json.dumps(data))
    return data


def ticker_to_cik(ticker: str):
    ticker = ticker.upper().strip()
    data = load_ticker_map()
    for entry in data.values():
        if entry["ticker"].upper() == ticker:
            return str(entry["cik_str"]).zfill(10)
    return None


# ----------------------------- FETCH 10-K -----------------------------
@tool
def fetch_10k_html(ticker: str, year: int):
    """Fetches the 10-K HTML for a given ticker and year."""
    # Cache path
    cache_html = CACHE_DIR / f"{ticker}_{year}.html"
    if cache_html.exists():
        return cache_html.read_text()
    cik = ticker_to_cik(ticker)
    subs_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    subs = sec_get(subs_url).json()
    recent = subs.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accs = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primaries = recent.get("primaryDocument", [])
    target_idx = None
    # Find matching year 10-K else first 10-K
    for i, f in enumerate(forms):
        if f == "10-K":
            fyear = int(dates[i].split("-")[0])
            if fyear == year:
                target_idx = i
                break
    if target_idx is None:
        for i, f in enumerate(forms):
            if f == "10-K":
                target_idx = i
                break
    acc = accs[target_idx]
    primary = primaries[target_idx]
    acc_nodash = acc.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{primary}"
    html = sec_get(url).text
    cache_html.write_text(html)
    return html


# ----------------------------- HTML -> TEXT ---------------------------
@tool
def html_to_text(html: str):
    """Converts HTML to clean text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    return "\n".join(lines)


# ----------------------------- SECTION EXTRACTION ---------------------
SECTION_REGEX = {
    "business": [r"\bItem\s+1\.*\s+Business\b"],
    "risks": [r"\bItem\s+1A\.*\s+Risk\s+Factors\b"],
    "mdna": [r"\bItem\s+7\.*\s+Management'?s?\s+Discussion.*?Analysis\b"],
}
NEXT_ITEM = re.compile(r"\bItem\s+([0-9]{1,2}[A]?)\b", re.IGNORECASE)


@tool
def extract_sections(text: str):
    """Extracts the Business, Risk Factors, and MD&A sections from a 10-K filing."""
    sections = {}
    for key, patterns in SECTION_REGEX.items():
        start = None
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                start = m.start()
                break
        if start is None:
            continue
        next_m = None
        for nm in NEXT_ITEM.finditer(text, start + 10):
            if nm.start() > start:
                next_m = nm
                break
        chunk = text[start : next_m.start()] if next_m else text[start:]
        sections[key] = re.sub(r"[ \t]+", " ", chunk.strip())
    return sections


# ----------------------------- SUMMARIZATION --------------------------
KEYWORDS = [
    "growth",
    "revenue",
    "profit",
    "margin",
    "customer",
    "market",
    "demand",
    "supply",
    "competition",
    "regulation",
    "strategy",
    "cash",
    "liquidity",
    "cost",
    "capital",
    "technology",
    "inflation",
    "risk",
    "operations",
]


def simple_bullets(text: str, n=10):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    scored = []
    for s in sentences:
        if 25 <= len(s) <= 260:
            kw = sum(1 for k in KEYWORDS if k in s.lower())
            score = kw * 10 + min(len(s), 200)
            scored.append((score, s.strip()))
    scored.sort(reverse=True, key=lambda x: x[0])
    out = []
    used_frag = set()
    for _, s in scored:
        frag = s.lower()[:60]
        if frag in used_frag:
            continue
        used_frag.add(frag)
        out.append(s)
        if len(out) == n:
            break
    return out


def openai_chat(prompt: str, max_tokens=600):
    if not OPENAI_KEY:
        return ""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You write concise financial bullets."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


@tool
def summarize_section(name: str, text: str):
    """Summarizes a section of a 10-K filing into 10 concise bullets."""
    if OPENAI_KEY:
        prompt = f"Summarize this 10-K section into EXACTLY 10 concise bullets (<=25 words, no numbering):\n\n{text[:20000]}"
        raw = openai_chat(prompt)
        bullets = [l.strip("-• \t") for l in raw.splitlines() if l.strip()]
        if len(bullets) >= 10:
            return bullets[:10]
    return simple_bullets(text, 10)


# ----------------------------- RISKS ----------------------------------
RISK_SEV_WORDS = {
    "high": [
        "material",
        "significant",
        "severe",
        "critical",
        "going concern",
        "substantial",
    ],
    "medium": ["could", "may", "uncertain", "volatility", "challenging"],
    "low": ["limited", "manageable", "mitigated"],
}


def heuristic_risk_candidates(text: str):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    picks = []
    for l in lines:
        low = l.lower()
        if any(
            word in low
            for word in [
                "risk",
                "uncertain",
                "competition",
                "regulation",
                "supply",
                "volatility",
                "cyber",
            ]
        ):
            if 40 < len(l) < 400:
                picks.append(l)
    # Dedup crude
    uniq = []
    seen = set()
    for p in picks:
        key = p[:70].lower()
        if key not in seen:
            uniq.append(p)
            seen.add(key)
    return uniq[:50]


def score_severity(line: str):
    low = line.lower()
    score = 0
    for w in RISK_SEV_WORDS["high"]:
        if w in low:
            score += 3
    for w in RISK_SEV_WORDS["medium"]:
        if w in low:
            score += 1
    for w in RISK_SEV_WORDS["low"]:
        if w in low:
            score -= 1
    if score >= 4:
        return "High"
    if score >= 2:
        return "Medium"
    return "Low"


@tool
def top_risks(risk_text: str):
    """Identifies the top 5 risks from the Risk Factors section of a 10-K filing."""
    cands = heuristic_risk_candidates(risk_text)
    scored = []
    rank_map = {"High": 3, "Medium": 2, "Low": 1}
    for c in cands:
        sev = score_severity(c)
        scored.append((rank_map[sev], sev, c))
    scored.sort(reverse=True, key=lambda x: (x[0], len(x[2])))
    out = []
    for _, sev, t in scored[:5]:
        name = t.split(".")[0]
        if len(name) > 80:
            name = name[:77] + "..."
        out.append(
            {
                "risk": name,
                "severity": sev,
                "rationale": (t[:180] + "..." if len(t) > 180 else t),
            }
        )
    return out


# ----------------------------- THESIS ---------------------------------
@tool
def build_thesis(business_bullets, mdna_bullets, risks):
    """Builds an investment thesis based on the summarized sections and top risks."""
    if OPENAI_KEY:
        risk_list = "\n".join(f"- {r['risk']} ({r['severity']})" for r in risks)
        prompt = f"""
Combine these into ONE investment thesis paragraph (<=150 words):
Business:
{chr(10).join("- " + b for b in business_bullets)}
MD&A:
{chr(10).join("- " + b for b in mdna_bullets)}
Risks:
{risk_list}
"""
        txt = openai_chat(prompt, max_tokens=220)
        # Small clean
        return txt.strip()
    # Fallback heuristic
    return (
        "Investment Thesis: The company shows "
        + "; ".join(business_bullets[:3])
        + ". Operational/financial themes: "
        + "; ".join(mdna_bullets[:3])
        + ". Key risks: "
        + ", ".join(f"{r['risk']} ({r['severity']})" for r in risks)
        + "."
    )


# ----------------------------- MARKDOWN EXPORT ------------------------
def export_md(ticker, year, sections, bullets, risks, thesis):
    lines = []
    lines.append(f"# {ticker} {year} 10-K Analysis")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}_")
    lines.append("")
    lines.append("## Investment Thesis")
    lines.append(thesis.strip())
    lines.append("")
    label_map = {
        "business": "Business Overview (Item 1)",
        "risks": "Risk Factors (Item 1A)",
        "mdna": "MD&A (Item 7)",
    }
    order = ["business", "risks", "mdna"]
    for k in order:
        if k in bullets:
            lines.append(f"## {label_map[k]}")
            for b in bullets[k]:
                lines.append(f"- {b}")
            lines.append("")
            if k == "risks":
                lines.append("### Top 5 Risks")
                for r in risks:
                    lines.append(
                        f"- **{r['risk']}** (Severity: {r['severity']}) — {r['rationale']}"
                    )
                lines.append("")
    return "\n".join(lines)


# ----------------------------- LANGCHAIN AGENT ------------------------
def create_agent():
    tools = [
        fetch_10k_html,
        html_to_text,
        extract_sections,
        summarize_section,
        top_risks,
        build_thesis,
    ]
    prompt = ChatPromptTemplate.from_template(
        """You are a financial analyst. Your task is to analyze a 10-K filing.

        1. Fetch the 10-K HTML for the given ticker and year.
        2. Convert the HTML to text.
        3. Extract the Business, Risk Factors, and MD&A sections.
        4. Summarize each section into 10 concise bullets.
        5. Identify the top 5 risks from the Risk Factors section.
        6. Build an investment thesis based on the summarized sections and top risks.
        7. Return a JSON object with the following keys: a. \"sections\": the extracted sections b. \"bullets\": the summarized bullets for each section c. \"risks\": the top 5 risks d. \"thesis\": the investment thesis

        {input}"""
    )
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor


# ----------------------------- STREAMLIT UI ---------------------------
st.set_page_config(page_title="Simple 10-K Analyzer", layout="wide")
st.title("Simple 10-K Analyzer (Tutorial Build)")
st.caption(
    "Fetch a 10-K, extract sections, make bullets, list top risks, and produce a thesis. Educational demo."
)
ticker = st.text_input("Ticker", value="AAPL")
year = st.number_input(
    "Year (filing year - approximate)", min_value=1995, max_value=2100, value=2024
)
run = st.button("Fetch & Analyze", type="primary")

if run:
    agent_executor = create_agent()
    query = f"Analyze the 10-K filing for {ticker} for the year {year}."
    with st.spinner("Analyzing..."):
        result = agent_executor.invoke({"input": query})

    # The result is a dictionary with the output key.
    # The value of the output key is a JSON string.
    analysis = json.loads(result["output"])

    st.markdown("## Investment Thesis")
    st.write(analysis["thesis"])

    st.markdown("## Section Bullet Summaries")
    col1, col2, col3 = st.columns(3)
    for idx, (k, title) in enumerate(
        [("business", "Business"), ("risks", "Risk Factors"), ("mdna", "MD&A")]
    ):
        if k in analysis["bullets"]:
            col = [col1, col2, col3][idx % 3]
            with col:
                st.markdown(f"### {title}")
                for b in analysis["bullets"][k]:
                    st.markdown(f"- {b}")

    if analysis["risks"]:
        st.markdown("## Top 5 Risks")
        for r in analysis["risks"]:
            st.markdown(
                f"- **{r['risk']}** (Severity: {r['severity']}) — {r['rationale']}"
            )

    md = export_md(
        ticker.upper(),
        int(year),
        analysis["sections"],
        analysis["bullets"],
        analysis["risks"],
        analysis["thesis"],
    )
    st.download_button(
        "Download Markdown",
        data=md,
        file_name=f"{ticker.upper()}_{year}_10K_analysis.md",
        mime="text/markdown",
    )

st.markdown("---")
st.caption(
    "No investment advice. Educational demo. Add robust error handling, better parsing, vector search, and advanced LLM logic in a production version."
)
