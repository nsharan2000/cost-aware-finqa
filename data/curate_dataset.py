"""
Curate 200 questions from FinQA and build a realistic financial datastore.

Architecture:
1. DATASTORE (large, real-world):
   - company_financials: One table per company with all their financial data
   - documents: SEC filing text passages organized by company/year
2. QUESTIONS (small):
   - 200 questions that query over the datastore
   - Each linked to a company and specific financial report
"""

import json
import sqlite3
import hashlib
import re
import os
import urllib.request
from collections import defaultdict


def download_finqa_raw():
    """Download FinQA train split from GitHub."""
    print("Loading FinQA from GitHub repo...")
    url = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    return data


def parse_filename(filename):
    """Extract company ticker and year from filename like 'AAL/2014/page_18.pdf'."""
    parts = filename.split("/")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "UNKNOWN", "0000"


def classify_question(question, program, pre_text, post_text):
    """Classify into tool category and difficulty."""
    program = program or ""
    question_lower = question.lower()

    # Count program steps
    steps = [s.strip() for s in program.split("),") if s.strip()] if program else []
    num_steps = len(steps)

    # Text dependency
    text_keywords = ["why", "factor", "reason", "driven", "cause", "explain", "describe",
                     "strategy", "risk", "outlook", "plan", "expect", "believe", "management",
                     "primarily", "due to", "result of", "impact"]
    has_text_dep = any(kw in question_lower for kw in text_keywords)

    # External data need
    external_keywords = ["industry", "sector", "market", "peer", "competitor", "benchmark",
                        "average", "compare", "typical", "standard"]
    needs_external = any(kw in question_lower for kw in external_keywords)

    # Category
    if needs_external:
        category = "web_search"
    elif num_steps >= 3:
        category = "llm_upgrade"
    elif has_text_dep:
        category = "vector_search"
    else:
        category = "sql_primary"

    # Difficulty
    if num_steps <= 1 and not has_text_dep and not needs_external:
        difficulty = "easy"
    elif num_steps <= 2 or has_text_dep:
        difficulty = "medium"
    else:
        difficulty = "hard"

    return category, difficulty


def sanitize_col_name(name):
    """Make a string safe for SQL column names."""
    name = str(name).strip()
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    if not name or name[0].isdigit():
        name = "col_" + name
    return name[:60]


def build_datastore(raw_data, selected_questions, db_path):
    """Build the SQLite datastore with company financials and documents."""
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # =========================================================
    # TABLE 1: questions (the small table)
    # =========================================================
    cursor.execute("""
        CREATE TABLE questions (
            id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            gold_answer TEXT NOT NULL,
            category TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            task TEXT NOT NULL,
            program TEXT,
            company TEXT,
            fiscal_year TEXT,
            report_page TEXT,
            financial_table TEXT
        )
    """)

    # =========================================================
    # TABLE 2: documents (SEC filing text passages - the doc store)
    # =========================================================
    cursor.execute("""
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            fiscal_year TEXT NOT NULL,
            report_page TEXT,
            section TEXT NOT NULL,
            content TEXT NOT NULL,
            question_id TEXT,
            FOREIGN KEY (question_id) REFERENCES questions(id)
        )
    """)
    cursor.execute("CREATE INDEX idx_docs_company ON documents(company)")
    cursor.execute("CREATE INDEX idx_docs_company_year ON documents(company, fiscal_year)")

    # =========================================================
    # COMPANY FINANCIAL TABLES (one per company)
    # =========================================================
    # Group questions by company
    company_tables = defaultdict(list)  # company -> list of (table_data, question)

    for q in selected_questions:
        company = q["company"]
        table = q.get("table", [])
        if table and len(table) >= 2:
            company_tables[company].append((table, q))

    # Create one financial table per company, with a report_id to distinguish different reports
    created_tables = {}  # (company, report_idx) -> table_name
    question_table_map = {}  # question_id -> table_name

    for company, items in company_tables.items():
        safe_company = sanitize_col_name(company).lower()

        for idx, (table, q) in enumerate(items):
            table_name = f"financials_{safe_company}_{idx}"

            # Parse headers
            headers = table[0]
            col_names = []
            seen = {}
            for h in headers:
                c = sanitize_col_name(h) if h.strip() else "metric"
                if c in seen:
                    seen[c] += 1
                    c = f"{c}_{seen[c]}"
                else:
                    seen[c] = 0
                col_names.append(c)

            # Add metadata columns
            all_cols = ["_company", "_fiscal_year", "_report_page"] + col_names
            col_defs = ", ".join([f'"{c}" TEXT' for c in all_cols])

            try:
                cursor.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_defs})')

                # Insert rows
                placeholders = ", ".join(["?" for _ in all_cols])
                for row in table[1:]:
                    padded = list(row) + [""] * (len(col_names) - len(row))
                    padded = padded[:len(col_names)]
                    values = [company, q["fiscal_year"], q["report_page"]] + padded
                    cursor.execute(f'INSERT INTO "{table_name}" VALUES ({placeholders})', values)

                question_table_map[q["id"]] = table_name
                created_tables[(company, idx)] = table_name
            except Exception as e:
                print(f"Warning: Could not create table {table_name}: {e}")

    # =========================================================
    # TABLE 3: table_catalog (helps agent discover available tables)
    # =========================================================
    cursor.execute("""
        CREATE TABLE table_catalog (
            table_name TEXT PRIMARY KEY,
            company TEXT,
            description TEXT,
            columns TEXT,
            row_count INTEGER,
            sample_data TEXT
        )
    """)

    for (company, idx), table_name in created_tables.items():
        try:
            cursor.execute(f'PRAGMA table_info("{table_name}")')
            cols = [row[1] for row in cursor.fetchall()]

            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            row_count = cursor.fetchone()[0]

            cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 2')
            sample = cursor.fetchall()
            sample_str = json.dumps([list(r) for r in sample])

            cursor.execute(
                "INSERT OR REPLACE INTO table_catalog VALUES (?, ?, ?, ?, ?, ?)",
                (table_name, company,
                 f"Financial data for {company}",
                 json.dumps(cols), row_count, sample_str[:500])
            )
        except Exception:
            pass

    # =========================================================
    # Insert questions and documents
    # =========================================================
    task_map = {"easy": "basic_retrieval", "medium": "analytical_reasoning", "hard": "strategic_research"}

    for q in selected_questions:
        task = task_map[q["difficulty"]]
        fin_table = question_table_map.get(q["id"], "")

        cursor.execute("""
            INSERT OR REPLACE INTO questions
            (id, question, gold_answer, category, difficulty, task, program, company, fiscal_year, report_page, financial_table)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            q["id"], q["question"], str(q["answer"]),
            q["category"], q["difficulty"], task,
            q.get("program", ""), q["company"], q["fiscal_year"],
            q["report_page"], fin_table
        ))

        # Insert text passages as documents
        for passage in (q.get("pre_text", []) or []):
            if passage and passage.strip():
                cursor.execute(
                    "INSERT INTO documents (company, fiscal_year, report_page, section, content, question_id) VALUES (?, ?, ?, ?, ?, ?)",
                    (q["company"], q["fiscal_year"], q["report_page"], "pre_text", passage.strip(), q["id"])
                )
        for passage in (q.get("post_text", []) or []):
            if passage and passage.strip():
                cursor.execute(
                    "INSERT INTO documents (company, fiscal_year, report_page, section, content, question_id) VALUES (?, ?, ?, ?, ?, ?)",
                    (q["company"], q["fiscal_year"], q["report_page"], "post_text", passage.strip(), q["id"])
                )

    conn.commit()

    # Print stats
    cursor.execute("SELECT COUNT(*) FROM questions")
    print(f"Questions: {cursor.fetchone()[0]}")
    cursor.execute("SELECT COUNT(*) FROM documents")
    print(f"Documents: {cursor.fetchone()[0]}")
    cursor.execute("SELECT COUNT(*) FROM table_catalog")
    print(f"Financial tables: {cursor.fetchone()[0]}")
    cursor.execute("SELECT DISTINCT company FROM questions")
    companies = cursor.fetchall()
    print(f"Companies: {len(companies)} ({', '.join(c[0] for c in companies[:10])}...)")

    conn.close()
    print(f"Database saved to {db_path}")


def curate_questions(raw_data, target=200):
    """Select and classify questions from raw FinQA data."""

    # Flatten and classify
    all_items = []
    for item in raw_data:
        qa = item.get("qa", {})
        question = qa.get("question", "")
        if not question.strip():
            continue

        company, year = parse_filename(item.get("filename", ""))
        program = qa.get("program", "")
        pre_text = item.get("pre_text", [])
        post_text = item.get("post_text", [])

        category, difficulty = classify_question(question, program, pre_text, post_text)

        all_items.append({
            "id": hashlib.md5(f"{question}_{company}_{year}".encode()).hexdigest()[:12],
            "question": question,
            "answer": qa.get("exe_ans", qa.get("answer", "")),
            "program": program,
            "company": company,
            "fiscal_year": year,
            "report_page": item.get("filename", "").split("/")[-1].replace(".pdf", ""),
            "category": category,
            "difficulty": difficulty,
            "table": item.get("table", []),
            "pre_text": pre_text,
            "post_text": post_text,
        })

    # Target distribution
    targets = {
        "sql_primary": {"easy": 45, "medium": 25, "hard": 10},
        "vector_search": {"easy": 15, "medium": 25, "hard": 10},
        "web_search": {"easy": 5, "medium": 15, "hard": 20},
        "llm_upgrade": {"easy": 5, "medium": 5, "hard": 20},
    }

    selected = []
    used_ids = set()

    for cat, diffs in targets.items():
        pool = [q for q in all_items if q["category"] == cat and q["id"] not in used_ids]

        for diff, count in diffs.items():
            diff_pool = [q for q in pool if q["difficulty"] == diff]

            take = min(count, len(diff_pool))
            selected.extend(diff_pool[:take])
            for q in diff_pool[:take]:
                used_ids.add(q["id"])

            # Fill shortfall from sql_primary pool
            shortfall = count - take
            if shortfall > 0:
                fill_pool = [q for q in all_items if q["id"] not in used_ids]
                extras = fill_pool[:shortfall]
                for e in extras:
                    e["category"] = cat
                    e["difficulty"] = diff
                    used_ids.add(e["id"])
                selected.extend(extras)

    # Pad to target if needed
    remaining = [q for q in all_items if q["id"] not in used_ids]
    while len(selected) < target and remaining:
        selected.append(remaining.pop(0))

    # Augment web_search questions with external context needs
    benchmarks = [
        "S&P 500 average P/E ratio", "industry average operating margin",
        "sector average debt-to-equity", "market average ROE",
        "industry revenue growth rate", "sector average profit margin",
        "market capitalization benchmark", "industry R&D spending ratio",
    ]
    for q in selected:
        if q["category"] == "web_search":
            bench = benchmarks[hash(q["id"]) % len(benchmarks)]
            q["original_question"] = q["question"]
            q["question"] = q["question"].rstrip("?.") + f". How does this compare to the {bench}?"
            q["external_context"] = bench

    # Stats
    cat_counts = defaultdict(int)
    diff_counts = defaultdict(int)
    for q in selected:
        cat_counts[q["category"]] += 1
        diff_counts[q["difficulty"]] += 1
    print(f"Selected {len(selected)} questions")
    print(f"By category: {dict(cat_counts)}")
    print(f"By difficulty: {dict(diff_counts)}")

    return selected


def save_questions_json(questions, path):
    """Save curated questions as JSON (without raw table data to keep it small)."""
    output = []
    for q in questions:
        output.append({
            "id": q["id"],
            "question": q["question"],
            "answer": str(q["answer"]),
            "category": q["category"],
            "difficulty": q["difficulty"],
            "task": {"easy": "basic_retrieval", "medium": "analytical_reasoning", "hard": "strategic_research"}[q["difficulty"]],
            "program": q.get("program", ""),
            "company": q["company"],
            "fiscal_year": q["fiscal_year"],
            "report_page": q.get("report_page", ""),
            "external_context": q.get("external_context", ""),
            "original_question": q.get("original_question", ""),
        })
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(output)} questions to {path}")


def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))

    print("Step 1: Downloading FinQA...")
    raw = download_finqa_raw()
    print(f"Downloaded {len(raw)} examples")

    print("\nStep 2: Curating 200 questions...")
    questions = curate_questions(raw, target=200)

    print("\nStep 3: Saving questions JSON...")
    save_questions_json(questions, os.path.join(data_dir, "finqa_200.json"))

    print("\nStep 4: Building datastore...")
    build_datastore(raw, questions, os.path.join(data_dir, "financial_data.db"))

    print("\nDone!")


if __name__ == "__main__":
    main()
