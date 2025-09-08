# requirements.txt
"""
streamlit>=1.28.0
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.3
sqlite3  # Built-in with Python
pandas>=2.0.0
python-dateutil>=2.8.0
openai>=1.0.0  # For LLM summarization
PyPDF2>=3.0.0
reportlab>=4.0.0
"""

import streamlit as st
import requests
import sqlite3
import json
import time
import re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
from typing import Dict, List, Optional, Tuple
import hashlib
import os
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Filing:
    ticker: str
    cik: str
    filing_date: str
    form_type: str
    filing_url: str
    html_url: str


@dataclass
class ExtractedSections:
    business_overview: str
    risk_factors: str
    mda: str  # Management's Discussion and Analysis


@dataclass
class AnalysisResult:
    business_bullets: List[str]
    risk_bullets: List[str]
    mda_bullets: List[str]
    top_risks: List[Dict[str, str]]  # risk, severity, explanation
    thesis: str


class SECClient:
    """SEC EDGAR client with rate limiting and caching"""

    def __init__(self):
        self.base_url = "https://www.sec.gov"
        self.headers = {
            "User-Agent": "Financial Analyzer v1.0 (educational@example.com)",
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov",
        }
        self.last_request_time = 0
        self.min_request_interval = 0.1  # SEC allows 10 requests per second

    def _rate_limit(self):
        """Ensure we don't exceed SEC rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    def get_company_cik(self, ticker: str) -> Optional[str]:
        """Get CIK for a given ticker symbol"""
        self._rate_limit()

        url = f"{self.base_url}/cgi-bin/browse-edgar"
        params = {"action": "getcompany", "ticker": ticker.upper(), "output": "xml"}

        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "xml")
            cik_element = soup.find("CIK")
            if cik_element:
                return cik_element.text.zfill(10)  # Zero-pad to 10 digits
        except Exception as e:
            logger.error(f"Error fetching CIK for {ticker}: {e}")

        return None

    def get_latest_10k(
        self, cik: str, ticker: str, year: Optional[int] = None
    ) -> Optional[Filing]:
        """Get the latest 10-K filing for a company"""
        self._rate_limit()

        url = f"{self.base_url}/cgi-bin/browse-edgar"
        params = {
            "action": "getcompany",
            "CIK": cik,
            "type": "10-K",
            "dateb": "",
            "owner": "exclude",
            "count": "10",
        }

        if year:
            # Set date range for the specific year
            params["dateb"] = f"{year}1231"
            params["datea"] = f"{year}0101"

        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Find the first 10-K filing
            filing_table = soup.find("table", {"class": "tableFile2"})
            if not filing_table:
                return None

            rows = filing_table.find_all("tr")[1:]  # Skip header
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 4:
                    form_type = cols[0].text.strip()
                    if form_type == "10-K":
                        filing_date = cols[3].text.strip()
                        documents_link = cols[1].find("a")["href"]

                        # Get the actual HTML document URL
                        html_url = self._get_html_document_url(documents_link)
                        if html_url:
                            return Filing(
                                ticker=ticker,
                                cik=cik,
                                filing_date=filing_date,
                                form_type=form_type,
                                filing_url=urljoin(self.base_url, documents_link),
                                html_url=html_url,
                            )
        except Exception as e:
            logger.error(f"Error fetching 10-K for CIK {cik}: {e}")

        return None

    def _get_html_document_url(self, documents_link: str) -> Optional[str]:
        """Get the actual HTML document URL from the documents page"""
        self._rate_limit()

        try:
            response = requests.get(
                urljoin(self.base_url, documents_link), headers=self.headers
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Look for the main 10-K document (usually .htm file)
            document_table = soup.find("table", {"class": "tableFile"})
            if document_table:
                rows = document_table.find_all("tr")[1:]  # Skip header
                for row in rows:
                    cols = row.find_all("td")
                    if len(cols) >= 3:
                        doc_type = cols[3].text.strip()
                        if "10-K" in doc_type and not doc_type.startswith("EX-"):
                            doc_link = cols[2].find("a")["href"]
                            return urljoin(self.base_url, doc_link)
        except Exception as e:
            logger.error(f"Error getting HTML document URL: {e}")

        return None

    def fetch_filing_content(self, html_url: str) -> Optional[str]:
        """Fetch the HTML content of a filing"""
        self._rate_limit()

        try:
            response = requests.get(html_url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error fetching filing content: {e}")

        return None


class FilingExtractor:
    """Extract key sections from 10-K filings"""

    def __init__(self):
        # Common patterns for section identification
        self.business_patterns = [
            r"(?i)item\s+1[\.\s]*business",
            r"(?i)business\s*$",
            r"(?i)our\s+business",
            r"(?i)the\s+business",
        ]

        self.risk_patterns = [
            r"(?i)item\s+1a[\.\s]*risk\s+factors",
            r"(?i)risk\s+factors",
            r"(?i)factors\s+that\s+may\s+affect",
        ]

        self.mda_patterns = [
            r"(?i)item\s+7[\.\s]*management[\'\s]*s\s+discussion",
            r"(?i)management[\'\s]*s\s+discussion\s+and\s+analysis",
            r"(?i)md&a",
            r"(?i)financial\s+condition\s+and\s+results",
        ]

    def extract_sections(self, html_content: str) -> ExtractedSections:
        """Extract key sections from 10-K HTML content"""
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text()

        business_overview = self._extract_section(text, self.business_patterns)
        risk_factors = self._extract_section(text, self.risk_patterns)
        mda = self._extract_section(text, self.mda_patterns)

        return ExtractedSections(
            business_overview=business_overview, risk_factors=risk_factors, mda=mda
        )

    def _extract_section(self, text: str, patterns: List[str]) -> str:
        """Extract a section based on patterns"""
        text_lines = text.split("\n")
        section_start = -1

        # Find section start
        for i, line in enumerate(text_lines):
            for pattern in patterns:
                if re.search(pattern, line.strip()):
                    section_start = i
                    break
            if section_start != -1:
                break

        if section_start == -1:
            return "Section not found"

        # Find section end (next item or significant break)
        section_end = len(text_lines)
        next_item_patterns = [
            r"(?i)item\s+\d+[a-z]*[\.\s]",
            r"(?i)part\s+[iv]+",
            r"(?i)table\s+of\s+contents",
        ]

        for i in range(section_start + 1, len(text_lines)):
            line = text_lines[i].strip()
            if len(line) > 100:  # Skip short lines
                for pattern in next_item_patterns:
                    if re.search(pattern, line):
                        section_end = i
                        break
            if section_end != len(text_lines):
                break

        # Extract and clean the section
        section_lines = text_lines[section_start:section_end]
        section_text = "\n".join(section_lines)

        # Clean up the text
        section_text = re.sub(r"\s+", " ", section_text)
        section_text = re.sub(r"[^\w\s\.\,\;\:\!\?\-\(\)\$\%]", "", section_text)

        return section_text[:10000]  # Limit to first 10k characters


class DatabaseManager:
    """Manage SQLite cache for filings"""

    def __init__(self, db_path: str = "filings_cache.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS filings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                cik TEXT NOT NULL,
                filing_date TEXT NOT NULL,
                form_type TEXT NOT NULL,
                filing_url TEXT NOT NULL,
                html_url TEXT NOT NULL,
                content_hash TEXT UNIQUE,
                raw_content TEXT,
                business_overview TEXT,
                risk_factors TEXT,
                mda TEXT,
                cached_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT NOT NULL,
                business_bullets TEXT,
                risk_bullets TEXT,
                mda_bullets TEXT,
                top_risks TEXT,
                thesis TEXT,
                created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (content_hash) REFERENCES filings (content_hash)
            )
        """)

        conn.commit()
        conn.close()

    def get_cached_filing(
        self, ticker: str, year: Optional[int] = None
    ) -> Optional[Dict]:
        """Get cached filing data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT * FROM filings 
            WHERE ticker = ? 
            ORDER BY filing_date DESC 
            LIMIT 1
        """

        cursor.execute(query, (ticker.upper(),))
        row = cursor.fetchone()
        conn.close()

        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))

        return None

    def cache_filing(self, filing: Filing, content: str, sections: ExtractedSections):
        """Cache filing data"""
        content_hash = hashlib.md5(content.encode()).hexdigest()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO filings 
                (ticker, cik, filing_date, form_type, filing_url, html_url, 
                 content_hash, raw_content, business_overview, risk_factors, mda)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    filing.ticker,
                    filing.cik,
                    filing.filing_date,
                    filing.form_type,
                    filing.filing_url,
                    filing.html_url,
                    content_hash,
                    content,
                    sections.business_overview,
                    sections.risk_factors,
                    sections.mda,
                ),
            )
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
        finally:
            conn.close()

        return content_hash

    def get_cached_analysis(self, content_hash: str) -> Optional[Dict]:
        """Get cached analysis results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM analysis_results 
            WHERE content_hash = ? 
            ORDER BY created_date DESC 
            LIMIT 1
        """,
            (content_hash,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))

        return None

    def cache_analysis(self, content_hash: str, analysis: AnalysisResult):
        """Cache analysis results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO analysis_results 
                (content_hash, business_bullets, risk_bullets, mda_bullets, top_risks, thesis)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    content_hash,
                    json.dumps(analysis.business_bullets),
                    json.dumps(analysis.risk_bullets),
                    json.dumps(analysis.mda_bullets),
                    json.dumps(analysis.top_risks),
                    analysis.thesis,
                ),
            )
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
        finally:
            conn.close()


class LLMAnalyzer:
    """Analyze extracted sections using LLM (mock implementation)"""

    def __init__(self):
        pass

    def analyze_sections(
        self, sections: ExtractedSections, ticker: str
    ) -> AnalysisResult:
        """
        Analyze extracted sections and generate insights
        Note: This is a mock implementation. In production, you would use OpenAI API
        or another LLM service.
        """

        # Mock analysis - in production, replace with actual LLM calls
        business_bullets = self._generate_mock_bullets(
            sections.business_overview, "business"
        )
        risk_bullets = self._generate_mock_bullets(sections.risk_factors, "risk")
        mda_bullets = self._generate_mock_bullets(sections.mda, "mda")

        top_risks = self._generate_mock_top_risks(sections.risk_factors)
        thesis = self._generate_mock_thesis(ticker, sections)

        return AnalysisResult(
            business_bullets=business_bullets,
            risk_bullets=risk_bullets,
            mda_bullets=mda_bullets,
            top_risks=top_risks,
            thesis=thesis,
        )

    def _generate_mock_bullets(self, content: str, section_type: str) -> List[str]:
        """Generate mock bullet points"""
        if not content or content == "Section not found":
            return [f"No {section_type} information available in the filing."]

        # Simple keyword-based bullet generation (mock)
        bullets = []
        content_lower = content.lower()

        if section_type == "business":
            if "revenue" in content_lower:
                bullets.append(
                    "• Company generates revenue through multiple business segments"
                )
            if "products" in content_lower or "services" in content_lower:
                bullets.append("• Offers diverse products and services to customers")
            if "market" in content_lower:
                bullets.append("• Operates in competitive market environments")
            if "customers" in content_lower:
                bullets.append(
                    "• Serves a broad customer base across different sectors"
                )
            if "technology" in content_lower:
                bullets.append("• Leverages technology to deliver solutions")

        elif section_type == "risk":
            if "market" in content_lower:
                bullets.append("• Faces market volatility and competitive pressures")
            if "regulatory" in content_lower or "regulation" in content_lower:
                bullets.append(
                    "• Subject to regulatory changes and compliance requirements"
                )
            if "economic" in content_lower:
                bullets.append("• Exposed to economic downturns and market conditions")
            if "cyber" in content_lower or "security" in content_lower:
                bullets.append("• Cybersecurity threats pose operational risks")

        elif section_type == "mda":
            if "revenue" in content_lower and "increase" in content_lower:
                bullets.append(
                    "• Revenue showed growth trends during the reporting period"
                )
            if "profit" in content_lower or "margin" in content_lower:
                bullets.append("• Profitability metrics were discussed in detail")
            if "cash" in content_lower:
                bullets.append("• Cash flow and liquidity position were analyzed")

        # Pad with generic bullets if needed
        while len(bullets) < 10:
            bullets.append(
                f"• Additional {section_type} details available in full filing"
            )

        return bullets[:10]

    def _generate_mock_top_risks(self, risk_content: str) -> List[Dict[str, str]]:
        """Generate mock top risks"""
        if not risk_content or risk_content == "Section not found":
            return [
                {
                    "risk": "No risk information available",
                    "severity": "Unknown",
                    "explanation": "Risk factors section not found in filing",
                }
            ]

        mock_risks = [
            {
                "risk": "Market Competition",
                "severity": "High",
                "explanation": "Intense competition may impact market share and pricing power",
            },
            {
                "risk": "Regulatory Changes",
                "severity": "Medium",
                "explanation": "Evolving regulations could affect operations and compliance costs",
            },
            {
                "risk": "Economic Conditions",
                "severity": "Medium",
                "explanation": "Economic downturns may reduce demand for products/services",
            },
            {
                "risk": "Technology Disruption",
                "severity": "High",
                "explanation": "Rapid technological changes could make current offerings obsolete",
            },
            {
                "risk": "Supply Chain Issues",
                "severity": "Medium",
                "explanation": "Disruptions in supply chain could affect operations and costs",
            },
        ]

        return mock_risks[:5]

    def _generate_mock_thesis(self, ticker: str, sections: ExtractedSections) -> str:
        """Generate mock investment thesis"""
        return f"{ticker} operates in a complex business environment with various growth opportunities and risk factors. Based on the analysis of their latest 10-K filing, the company demonstrates both strategic positioning in their core markets and exposure to industry-wide challenges. Investors should consider the balance between growth potential and risk factors when evaluating this investment opportunity. The company's business model, competitive position, and risk management approach provide a foundation for long-term value creation, though market conditions and operational execution remain key variables."


def export_to_markdown(ticker: str, filing_date: str, analysis: AnalysisResult) -> str:
    """Export analysis results to Markdown format"""

    markdown_content = f"""# Financial Analysis Report: {ticker}
## Filing Date: {filing_date}
## Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Investment Thesis

{analysis.thesis}

---

## Business Overview
"""

    for bullet in analysis.business_bullets:
        markdown_content += f"{bullet}\n"

    markdown_content += "\n---\n\n## Risk Factors\n"

    for bullet in analysis.risk_bullets:
        markdown_content += f"{bullet}\n"

    markdown_content += "\n---\n\n## Management Discussion & Analysis\n"

    for bullet in analysis.mda_bullets:
        markdown_content += f"{bullet}\n"

    markdown_content += "\n---\n\n## Top 5 Risks by Severity\n"

    for i, risk in enumerate(analysis.top_risks, 1):
        markdown_content += f"""
### {i}. {risk["risk"]} (Severity: {risk["severity"]})
{risk["explanation"]}
"""

    markdown_content += (
        f"\n---\n\n*Report generated by Financial Report Analyzer v1.0*\n"
    )

    return markdown_content


# Streamlit App
def main():
    st.set_page_config(
        page_title="Financial Report Analyzer", page_icon="📊", layout="wide"
    )

    st.title("📊 Financial Report Analyzer (10-K Assistant)")
    st.markdown("Analyze SEC 10-K filings with AI-powered insights")

    # Initialize components
    sec_client = SECClient()
    extractor = FilingExtractor()
    db_manager = DatabaseManager()
    analyzer = LLMAnalyzer()

    # Sidebar for inputs
    with st.sidebar:
        st.header("Analysis Parameters")

        ticker = st.text_input(
            "Stock Ticker", value="AAPL", help="Enter company ticker symbol"
        )

        year = st.selectbox(
            "Year (Optional)",
            options=[None] + list(range(2024, 2019, -1)),
            format_func=lambda x: "Latest Available" if x is None else str(x),
            help="Select specific year or latest available",
        )

        use_cache = st.checkbox(
            "Use Cache", value=True, help="Use cached data when available"
        )

        analyze_button = st.button("🔍 Analyze Filing", type="primary")

    if analyze_button and ticker:
        with st.spinner(f"Analyzing {ticker} 10-K filing..."):
            # Step 1: Get CIK
            st.info("Step 1: Looking up company information...")
            cik = sec_client.get_company_cik(ticker.upper())

            if not cik:
                st.error(f"Could not find CIK for ticker: {ticker}")
                return

            st.success(f"Found CIK: {cik}")

            # Step 2: Check cache
            cached_data = None
            if use_cache:
                st.info("Step 2: Checking cache...")
                cached_data = db_manager.get_cached_filing(ticker.upper(), year)
                if cached_data:
                    st.success("Found cached filing data")

            # Step 3: Fetch filing
            if not cached_data:
                st.info("Step 3: Fetching latest 10-K filing...")
                filing = sec_client.get_latest_10k(cik, ticker.upper(), year)

                if not filing:
                    st.error(f"Could not find 10-K filing for {ticker}")
                    return

                st.success(f"Found 10-K filing dated: {filing.filing_date}")

                # Step 4: Download content
                st.info("Step 4: Downloading filing content...")
                content = sec_client.fetch_filing_content(filing.html_url)

                if not content:
                    st.error("Could not download filing content")
                    return

                # Step 5: Extract sections
                st.info("Step 5: Extracting key sections...")
                sections = extractor.extract_sections(content)

                # Cache the data
                content_hash = db_manager.cache_filing(filing, content, sections)
                st.success("Filing data cached successfully")

            else:
                # Use cached data
                filing = Filing(
                    ticker=cached_data["ticker"],
                    cik=cached_data["cik"],
                    filing_date=cached_data["filing_date"],
                    form_type=cached_data["form_type"],
                    filing_url=cached_data["filing_url"],
                    html_url=cached_data["html_url"],
                )

                sections = ExtractedSections(
                    business_overview=cached_data["business_overview"],
                    risk_factors=cached_data["risk_factors"],
                    mda=cached_data["mda"],
                )

                content_hash = cached_data["content_hash"]

            # Step 6: Analyze sections
            st.info("Step 6: Generating AI analysis...")

            cached_analysis = None
            if use_cache:
                cached_analysis = db_manager.get_cached_analysis(content_hash)

            if cached_analysis:
                analysis = AnalysisResult(
                    business_bullets=json.loads(cached_analysis["business_bullets"]),
                    risk_bullets=json.loads(cached_analysis["risk_bullets"]),
                    mda_bullets=json.loads(cached_analysis["mda_bullets"]),
                    top_risks=json.loads(cached_analysis["top_risks"]),
                    thesis=cached_analysis["thesis"],
                )
                st.success("Using cached analysis")
            else:
                analysis = analyzer.analyze_sections(sections, ticker.upper())
                db_manager.cache_analysis(content_hash, analysis)
                st.success("Analysis completed and cached")

        # Display results
        st.header(f"📋 Analysis Results for {ticker.upper()}")

        # Filing info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ticker", ticker.upper())
        with col2:
            st.metric("Filing Date", filing.filing_date)
        with col3:
            st.metric("Form Type", filing.form_type)

        # Investment thesis
        st.subheader("🎯 Investment Thesis")
        st.write(analysis.thesis)

        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Business Overview", "Risk Factors", "MD&A", "Top Risks"]
        )

        with tab1:
            st.subheader("Business Overview Highlights")
            for bullet in analysis.business_bullets:
                st.write(bullet)

        with tab2:
            st.subheader("Risk Factor Highlights")
            for bullet in analysis.risk_bullets:
                st.write(bullet)

        with tab3:
            st.subheader("MD&A Highlights")
            for bullet in analysis.mda_bullets:
                st.write(bullet)

        with tab4:
            st.subheader("Top 5 Risks by Severity")
            for i, risk in enumerate(analysis.top_risks, 1):
                with st.expander(f"{i}. {risk['risk']} (Severity: {risk['severity']})"):
                    st.write(risk["explanation"])

        # Export functionality
        st.subheader("📤 Export Options")

        if st.button("📄 Generate Markdown Report"):
            markdown_content = export_to_markdown(
                ticker.upper(), filing.filing_date, analysis
            )

            st.download_button(
                label="⬇️ Download Markdown Report",
                data=markdown_content,
                file_name=f"{ticker.upper()}_analysis_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
            )

            with st.expander("Preview Markdown Content"):
                st.code(markdown_content, language="markdown")

        # Link to original filing
        st.subheader("🔗 Original Filing")
        st.markdown(f"[View Original 10-K Filing]({filing.html_url})")


if __name__ == "__main__":
    main()
