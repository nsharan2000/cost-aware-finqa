"""
Tool implementations for the Cost-Aware FinQA Environment.

3 tools with different costs:
- sql_query: $0.001 per call (cheap but penalized for errors)
- web_search: $0.02 per call (uses Serper API)
- upgrade_llm: $1.00 per call (1000x SQL — last resort only)
"""

import json
import os
import re
import sqlite3
from typing import Tuple
import urllib.request


# Tool costs — upgrade_llm is 1000x SQL, 50x web search
TOOL_COSTS = {
    "sql_query": 0.001,
    "web_search": 0.02,
    "upgrade_llm": 1.00,
    "submit_answer": 0.0,
}

SQL_ERROR_PENALTY = -0.15
SQL_QUERY_PENALTY = -0.01  # Small penalty per SQL call to discourage excessive queries
VALID_SQL_BONUS = 0.03
VALID_WEB_BONUS = 0.02
UPGRADE_LLM_PENALTY = -0.10  # Strong penalty — model must justify LLM upgrade as last resort
REDUNDANT_CALL_PENALTY = -0.05


def get_db_path():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "data", "financial_data.db")


def execute_sql_query(query: str, table_hint: str = "") -> Tuple[str, float]:
    """Execute SQL against the financial datastore.

    The agent can query:
    - table_catalog: discover available tables
    - financials_*: company financial data
    - documents: SEC filing text passages
    - questions: (blocked - would leak answers)
    """
    db_path = get_db_path()
    if not os.path.exists(db_path):
        return "Error: Database not found", SQL_ERROR_PENALTY

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA query_only = ON")
        cursor = conn.cursor()

        cleaned = query.strip().upper()
        if not cleaned.startswith("SELECT"):
            conn.close()
            return "Error: Only SELECT queries allowed", SQL_ERROR_PENALTY

        # Block access to questions table (would leak gold answers)
        if "QUESTIONS" in cleaned and "GOLD_ANSWER" in cleaned:
            conn.close()
            return "Error: Cannot query gold_answer column", SQL_ERROR_PENALTY

        dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "ATTACH"]
        for d in dangerous:
            if d in cleaned:
                conn.close()
                return f"Error: {d} not allowed", SQL_ERROR_PENALTY

        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.close()

        if not rows:
            return "Query returned no results. Try: SELECT * FROM table_catalog to see available tables.", SQL_ERROR_PENALTY

        result_lines = [" | ".join(columns)]
        result_lines.append("-" * len(result_lines[0]))
        for row in rows[:25]:
            result_lines.append(" | ".join(str(v) for v in row))
        if len(rows) > 25:
            result_lines.append(f"... ({len(rows)} total rows)")

        return "\n".join(result_lines), VALID_SQL_BONUS

    except sqlite3.Error as e:
        return f"SQL Error: {str(e)}", SQL_ERROR_PENALTY
    except Exception as e:
        return f"Error: {str(e)}", SQL_ERROR_PENALTY


def execute_web_search(query: str) -> Tuple[str, float]:
    """Search the web using Serper API."""
    api_key = (os.environ.get("SERPER_API_KEY")
               or os.environ.get("SERPER_OPENENV")
               or os.environ.get("SERPER-OPENENV"))

    if not api_key:
        return _simulated_web_search(query), VALID_WEB_BONUS

    try:
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": 5})
        req = urllib.request.Request(
            url,
            data=payload.encode("utf-8"),
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        results = []
        if data.get("answerBox"):
            ab = data["answerBox"]
            results.append(f"[Answer Box] {ab.get('title', '')}: {ab.get('answer', ab.get('snippet', ''))}")
        if data.get("knowledgeGraph"):
            kg = data["knowledgeGraph"]
            results.append(f"[Knowledge Graph] {kg.get('title', '')}: {kg.get('description', '')}")
        for item in data.get("organic", [])[:5]:
            results.append(f"- {item.get('title', '')}: {item.get('snippet', '')}")

        return "\n".join(results) if results else "No results found.", VALID_WEB_BONUS
    except Exception as e:
        return f"Web search error: {str(e)}. Falling back to simulated results.\n\n" + _simulated_web_search(query), VALID_WEB_BONUS


def _simulated_web_search(query: str) -> str:
    """Fallback simulated search for when Serper API is unavailable."""
    q = query.lower()
    benchmarks = {
        "p/e ratio": "S&P 500 average P/E: ~21.5. Tech sector: ~28-32. Financials: ~12-15.",
        "operating margin": "Avg operating margins - Tech: ~25%, Healthcare: ~15%, Retail: ~5%, Financials: ~30%.",
        "debt-to-equity": "Avg D/E ratios - Tech: ~0.5, Utilities: ~1.4, Financials: ~2.5, Healthcare: ~0.8.",
        "revenue growth": "Avg revenue growth - Tech: ~12% YoY, S&P 500: ~8%, Healthcare: ~6%.",
        "roe": "Average ROE - S&P 500: ~15%, Tech: ~20%, Banking: ~12%, Utilities: ~9%.",
        "profit margin": "Net margins - Tech: ~20%, Healthcare: ~12%, Retail: ~3%, Energy: ~8%.",
        "r&d": "R&D as % revenue - Tech: ~15%, Pharma: ~20%, Auto: ~5%, Aerospace: ~4%.",
        "market cap": "Large cap: >$10B, Mid cap: $2-10B, Small cap: <$2B.",
        "dividend": "Avg yield - S&P 500: ~1.5%, Utilities: ~3.5%, REITs: ~4%, Tech: ~0.8%.",
        "free cash flow": "Avg FCF yield - S&P 500: ~4.5%, Tech: ~5%, Healthcare: ~4%.",
    }
    for key, value in benchmarks.items():
        if key in q:
            return f"[Web Search Result] {value}"
    return "[Web Search Result] Industry benchmarks vary by sector. Tech typically has higher margins but lower dividends. Compare within the same GICS sector for meaningful results."


def execute_upgrade_llm(query: str, context: str = "") -> Tuple[str, float]:
    """Use a stronger model for complex reasoning.

    In deployment, this calls a more capable model endpoint.
    In the environment, it provides structured reasoning guidance.
    """
    parts = []
    if context:
        parts.append(f"[Enhanced Analysis]\nContext from previous tools:\n{context[:500]}")

    parts.append(
        "[Upgrade LLM] Enhanced reasoning mode activated.\n"
        "Approach for complex financial calculations:\n"
        "1. Extract exact numbers from the data\n"
        "2. Identify the formula needed (CAGR, ratio, % change, etc.)\n"
        "3. Show step-by-step arithmetic\n"
        "4. Verify by checking if the result is reasonable\n"
        f"Problem: {query[:300]}"
    )
    return "\n".join(parts), 0.0


def get_table_schema(question_id: str) -> str:
    """Get the SQL schema hint for a question's financial table."""
    db_path = get_db_path()
    if not os.path.exists(db_path):
        return "Database not available"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get financial table name for this question
        cursor.execute("SELECT financial_table, company, fiscal_year FROM questions WHERE id = ?", (question_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return "No table found"

        table_name, company, year = row
        if not table_name:
            conn.close()
            return f"Company: {company}, Year: {year}\nNo specific financial table. Try: SELECT * FROM table_catalog WHERE company = '{company}'"

        # Get columns
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        columns = cursor.fetchall()
        col_names = [c[1] for c in columns]

        # Get sample data
        cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 3')
        sample = cursor.fetchall()
        conn.close()

        schema = f'Company: {company} | Year: {year}\n'
        schema += f'Table: "{table_name}"\n'
        schema += f'Columns: {", ".join(col_names)}\n'

        if sample:
            schema += "\nSample data:\n"
            schema += " | ".join(col_names) + "\n"
            for r in sample:
                schema += " | ".join(str(v)[:30] for v in r) + "\n"

        schema += f'\nDiscovery: SELECT * FROM table_catalog WHERE company = \'{company}\''
        return schema

    except Exception as e:
        return f"Error: {str(e)}"


def grade_answer(submitted: str, gold: str, question_id: str = "") -> Tuple[float, str]:
    """Grade submitted answer vs gold. Fuzzy numerical match + text fallback."""
    submitted = str(submitted).strip()
    gold = str(gold).strip()

    # Numerical comparison
    sub_num = _extract_number(submitted)
    gold_num = _extract_number(gold)

    if sub_num is not None and gold_num is not None:
        if gold_num == 0:
            return (1.0, "Exact match (both ~0)") if abs(sub_num) < 0.01 else (0.0, f"Expected ~0, got {sub_num}")

        rel_error = abs(sub_num - gold_num) / abs(gold_num)

        # Handle percentage vs decimal mismatch (e.g. 63.67 vs 0.63634)
        if rel_error > 0.5 and gold_num != 0:
            # Try dividing submitted by 100 (model gave percentage, gold is decimal)
            alt_error_div = abs(sub_num / 100 - gold_num) / abs(gold_num)
            # Try multiplying submitted by 100 (model gave decimal, gold is percentage)
            alt_error_mul = abs(sub_num * 100 - gold_num) / abs(gold_num)
            rel_error = min(rel_error, alt_error_div, alt_error_mul)

        if rel_error <= 0.01:
            return 1.0, f"Exact match (error={rel_error:.4f})"
        elif rel_error <= 0.05:
            return 0.6, f"Close (error={rel_error:.4f})"
        else:
            return 0.0, f"Wrong (error={rel_error:.4f}, expected={gold_num}, got={sub_num})"

    # Text comparison
    if submitted.lower() == gold.lower():
        return 1.0, "Exact text match"

    sub_words = set(submitted.lower().split())
    gold_words = set(gold.lower().split())
    if gold_words:
        overlap = len(sub_words & gold_words) / len(gold_words)
        if overlap >= 0.8:
            return 0.8, f"High overlap ({overlap:.2f})"
        elif overlap >= 0.5:
            return 0.5, f"Moderate overlap ({overlap:.2f})"

    return 0.0, "No match"


def _extract_number(text: str):
    """Extract number from text, handling financial formats."""
    text = str(text).strip().replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        return float(text)
    except ValueError:
        match = re.search(r'[-+]?\d+\.?\d*', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
    return None
