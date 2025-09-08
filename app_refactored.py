import os, re, json, time, requests
from datetime import datetime
from bs4 import BeautifulSoup
import streamlit as st

# -----------------------------
# Config
# -----------------------------
USER_AGENT = "Your Name Contact@Email.com"
SEC_BASE = "https://data.sec.gov"
HEADERS = {"User-Agent": USER_AGENT}


# -----------------------------
# Caching helpers
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60)
def cached_get_json(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False, ttl=60 * 60)
def cached_get_text(url):
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.text


# -----------------------------
# SEC helpers
# -----------------------------
@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def fetch_ticker_map():
    return cached_get_json("https://www.sec.gov/files/company_tickers.json")


def cik_from_ticker(ticker: str):
    ticker = (ticker or "").strip().upper()
    data = fetch_ticker_map()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker:
            return str(entry.get("cik_str")).zfill(10)
    return None


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_company_submissions(cik: str):
    return cached_get_json(f"{SEC_BASE}/submissions/CIK{cik}.json")


def find_10k_for_year(submissions: dict, year: int):
    filings = submissions.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    dates = filings.get("filingDate", [])
    acc_nums = filings.get("accessionNumber", [])
    docs = filings.get("primaryDocument", [])
    for f, d, acc, doc in zip(forms, dates, acc_nums, docs):
        if f == "10-K" and str(d).startswith(str(year)):
            return acc, doc
    return None, None


def accession_to_url(acc: str, cik: str, primary_doc: str):
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{primary_doc}"


def get_filing_html(ticker: str, year: int):
    cik = cik_from_ticker(ticker)
    if not cik:
        raise ValueError("CIK not found for ticker.")
    subs = get_company_submissions(cik)
    acc, doc = find_10k_for_year(subs, year)
    if not acc or not doc:
        raise ValueError("10-K for that year not found.")
    url = accession_to_url(acc, cik, doc)
    html = cached_get_text(url)
    return html, acc, cik


# -----------------------------
# Text utils
# -----------------------------
def clean_text(s: str):
    return re.sub(r"\s+", " ", s or "").strip()


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text)


# -----------------------------
# Section extraction
# -----------------------------
ITEM_PATTERNS = {
    "Business Overview": r"item\s+1\.*\s*(business)",
    "Risk Factors": r"item\s+1a\.*\s*(risk factors?)",
    "MD&A": r"item\s+7\.*\s*(management['’]s discussion and analysis|management.?s discussion and analysis)",
}
STOP_ITEM_REGEX = r"item\s+([1-9][0-9]?[a-z]?)\."


def extract_sections(html: str):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n").replace("\xa0", " ")
    joined = "\n".join([l for l in text.splitlines() if l.strip()])
    lower = joined.lower()

    def find_section(item_regex):
        m = list(re.finditer(item_regex, lower, re.IGNORECASE))
        if not m:
            return ""
        start = m[0].start()
        nxt = list(re.finditer(STOP_ITEM_REGEX, lower[start + 10 :], re.IGNORECASE))
        end = len(joined)
        if nxt:
            end = start + 10 + nxt[0].start()
        return clean_text(joined[start:end])

    return {name: find_section(rx) for name, rx in ITEM_PATTERNS.items()}


# -----------------------------
# OpenAI (optional)
# -----------------------------
def openai_chat(system_prompt: str, user_prompt: str, model="gpt-4o-mini", temp=0.2):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": temp,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


# -----------------------------
# Summarization and fallbacks
# -----------------------------
BULLET_INSTR = (
    "You are an analyst. Summarize the 10-K section into exactly 10 concise, insight-rich bullets. "
    "Focus: business model, growth drivers, competitive positioning, financial highlights, strategy. "
    "Avoid boilerplate. Each bullet <= 25 words. Return as '- ' bullets only."
)
RISK_INSTR = (
    "Extract the 5 most material distinct risks from the 10-K Risk Factors. "
    "For each: JSON object with title (<=8 words), description (<=25 words), severity (High/Medium/Low). "
    "Return a JSON array only."
)
THESIS_INSTR = (
    "Write one investment thesis paragraph (<=120 words) synthesizing position, growth levers, key risks, outlook. "
    "Neutral, professional tone."
)


def llm_bullets(section_text):
    user = f"{BULLET_INSTR}\n\nSection:\n{section_text}"
    return openai_chat("10-K summarizer", user)


def llm_risks(risk_text):
    user = f"{RISK_INSTR}\n\nSection:\n{risk_text}"
    raw = openai_chat("10-K risk extractor", user)
    if not raw:
        return None
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    try:
        return json.loads(m.group(0) if m else raw)
    except Exception:
        return None


def llm_thesis(bullets_dict, risks_list):
    user = f"{THESIS_INSTR}\n\nBullets:\n{json.dumps(bullets_dict, indent=2)}\n\nRisks:\n{json.dumps(risks_list, indent=2)}"
    return openai_chat("10-K thesis writer", user)


def fallback_bullets(section_text, count=10):
    sents = [clean_text(s) for s in sentence_split(section_text) if clean_text(s)]
    sents = sents[:count]
    out = ["- " + (s[:180]) for s in sents]
    while len(out) < count:
        out.append("- (insufficient content)")
    return "\n".join(out[:count])


def heuristic_risks(risk_text, top=5):
    paras = [p.strip() for p in risk_text.split("\n") if len(p.split()) > 8]
    scored = []
    for p in paras:
        lw = p.lower()
        score = 2 * sum(
            k in lw
            for k in ["significant", "material", "adverse", "substantial", "severe"]
        ) + 1 * sum(k in lw for k in ["may", "could", "risk"])
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    res = []
    for score, para in scored[:top]:
        sev = "High" if score >= 5 else "Medium" if score >= 3 else "Low"
        title = " ".join(para.split()[:6])
        res.append(
            {"title": title, "description": clean_text(para)[:160], "severity": sev}
        )
    return res


def summarize_sections(sections):
    results, risks_list = {}, []
    for name, text in sections.items():
        if not text:
            results[name] = {"bullets": "- (section not found)"}
            continue
        bullets = llm_bullets(text) or fallback_bullets(text)
        results[name] = {"bullets": bullets}
        if name == "Risk Factors":
            risks_list = llm_risks(text) or heuristic_risks(text)
    thesis = llm_thesis({k: v["bullets"] for k, v in results.items()}, risks_list)
    if not thesis:
        thesis = "Thesis: Company shows identifiable risks and strategic drivers; deeper LLM analysis requires OPENAI_API_KEY."
    return results, risks_list, thesis


# -----------------------------
# Markdown export
# -----------------------------
def build_markdown(ticker, year, accession, summaries, risks, thesis):
    md = [f"# {ticker} {year} 10-K Highlights", f"*Accession:* {accession}", ""]
    for section, data in summaries.items():
        md += [f"## {section}", data["bullets"], ""]
    md.append("## Top 5 Risks")
    for r in risks:
        md.append(f"- **{r['title']}** ({r['severity']}): {r['description']}")
    md += ["", "## Investment Thesis", thesis]
    return "\n".join(md)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="10-K Analyzer (Simplified)", layout="wide")
st.title("10-K Financial Report Analyzer (Simplified)")

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
    st.caption(f"OpenAI Key: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")

if run:
    try:
        st.write(f"Fetching {ticker} {int(year)} 10-K ...")
        html, accession, cik = get_filing_html(ticker, int(year))
        st.success(f"Filing Accession: {accession} (CIK {cik})")

        with st.expander("Raw HTML (truncated)"):
            st.code(html[:6000])

        st.write("Extracting key sections...")
        sections = extract_sections(html)

        col1, col2, col3 = st.columns(3)
        col1.subheader("Business Overview")
        col1.write(
            (sections["Business Overview"] or "")[:1500]
            + ("..." if len(sections["Business Overview"]) > 1500 else "")
        )
        col2.subheader("Risk Factors")
        col2.write(
            (sections["Risk Factors"] or "")[:1500]
            + ("..." if len(sections["Risk Factors"]) > 1500 else "")
        )
        col3.subheader("MD&A")
        col3.write(
            (sections["MD&A"] or "")[:1500]
            + ("..." if len(sections["MD&A"]) > 1500 else "")
        )

        st.write("Summarizing (uses OpenAI if available)...")
        t0 = time.time()
        summaries, risks, thesis = summarize_sections(sections)
        st.info(f"Summarization complete in {time.time() - t0:.2f}s")

        st.header("Section Summaries")
        for section, data in summaries.items():
            st.subheader(section)
            st.text(data["bullets"])

        st.header("Top 5 Risks")
        for r in risks:
            st.markdown(f"- **{r['title']}** ({r['severity']}): {r['description']}")

        st.header("Investment Thesis")
        st.write(thesis)

        md_content = build_markdown(
            ticker, int(year), accession, summaries, risks, thesis
        )
        st.download_button(
            "Download Markdown",
            data=md_content,
            file_name=f"{ticker}_{int(year)}_10K_summary.md",
            mime="text/markdown",
        )
        with st.expander("Generated Markdown"):
            st.code(md_content)
    except Exception as e:
        st.error(f"Error: {e}")
