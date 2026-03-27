# semantic_layer.yaml — Deep Dive

> **What it is, why it exists, and exactly how every line of it gets used by the code.**

---

## The core problem it solves

When a user types:

> *"Show me all unpaid bills from Mumbai vendors last quarter"*

A database has no idea what "unpaid", "bills", or "last quarter" means.
It only knows table names and column names like `invoices`, `status`, `due_date`.

Without a semantic layer, you would send the raw question directly to the LLM along with the raw DDL (CREATE TABLE statements).  The LLM would then have to guess:

| User word | Raw DDL has | LLM must guess correctly |
|---|---|---|
| "bills" | `invoices` table | ✅ maybe |
| "unpaid" | `status` column with 7 enum values | ❌ often wrong |
| "last quarter" | `invoice_date DATE` column | ❌ no idea what the date range is |
| "revenue" | nothing – it's a business concept | ❌ completely blind |
| "top vendors" | `vendors` table | ❌ top by what metric? |

`semantic_layer.yaml` encodes all of this **business knowledge** in a structured file so the LLM gets precise, unambiguous context on every single call.

---

## Where the file lives and who reads it

```
cashflo/
├── semantic_layer.yaml          ← the file
└── engine/
    └── semantic_layer.py        ← the reader (SemanticLayer class)
```

The `SemanticLayer` class in `engine/semantic_layer.py` opens and parses the YAML at startup:

```python
# engine/semantic_layer.py  line 110-115
def __init__(self, yaml_path: str | Path | None = None) -> None:
    if yaml_path is None:
        yaml_path = Path(__file__).parent.parent / "semantic_layer.yaml"
    self._path: Path = Path(yaml_path)
    with open(self._path, encoding="utf-8") as fh:
        self._cfg: dict[str, Any] = yaml.safe_load(fh)   # ← reads the whole YAML
```

From there it splits the parsed dictionary into 6 named attributes:

```python
self.tables          = self._cfg.get("tables", {})
self.metrics         = self._cfg.get("metrics", {})
self.synonyms        = self._cfg.get("synonyms", {})
self.ambiguity_rules = self._cfg.get("ambiguity_rules", {})
self.temporal_cfg    = self._cfg.get("temporal", {})
# relationships split into direct_joins + multi_hops
```

Each attribute maps to a top-level section in the YAML.

---

## The 6 sections of the YAML

### Section 1 — `tables`

```yaml
tables:

  invoices:
    description: "Invoices submitted by vendors for payment. Each invoice links
                  to a vendor, a company, and a PO."
    synonyms: [bill, bills, payable, payables, invoice, invoices, tax_invoice]
    columns:
      grand_total: { type: decimal, desc: "Invoice value including all taxes in INR" }
      status:      { type: enum,
                     values: [received, validated, approved, rejected, on_hold, paid, partially_paid],
                     desc: "AP workflow status: received=logged; validated=3-way match passed; ..." }
      due_date:    { type: date, desc: "Payment due date" }
```

**What is it?**
Every table in the database gets a human-readable description, a list of alternative names users might say, and per-column metadata (type, description, allowed values for enums).

**How the code uses it — `build_prompt_context()` (line 184)**

The `SemanticLayer.build_prompt_context()` method loops over every table and formats it into a structured text block:

```python
for tbl_name, tbl in self.tables.items():
    lines.append(f"\nTable: {tbl_name}")
    lines.append(f"  Description: {tbl.get('description', '')}")
    syns = tbl.get("synonyms", [])
    if syns:
        lines.append(f"  Synonyms: {', '.join(syns)}")
    lines.append("  Columns:")
    for col, meta in tbl.get("columns", {}).items():
        col_type = meta.get("type", "")
        desc     = meta.get("desc", "")
        vals     = meta.get("values", [])
        val_str  = f" [{', '.join(vals)}]" if vals else ""
        lines.append(f"    {col} ({col_type}){val_str}: {desc}")
```

The output of that loop becomes part of the string injected into every LLM prompt:

```
=== DATABASE TABLES ===

Table: invoices
  Description: Invoices submitted by vendors for payment...
  Synonyms: bill, bills, payable, payables, invoice, invoices, tax_invoice
  Columns:
    grand_total (decimal): Invoice value including all taxes in INR
    status (enum) [received, validated, approved, rejected, on_hold, paid, partially_paid]: AP workflow status...
    due_date (date): Payment due date
```

**Effect:** The LLM now knows that `invoices.status` is an enum with exactly 7 allowed values.  It will never hallucinate `status = 'overdue'` (which doesn't exist in the real data).

---

### Section 2 — `relationships`

```yaml
relationships:
  direct:
    - { from: invoices, to: vendors,         join: "invoices.vendor_id = vendors.id" }
    - { from: invoices, to: purchase_orders, join: "invoices.po_id = purchase_orders.id" }
    - { from: payments, to: invoices,         join: "payments.invoice_id = invoices.id" }
    ...

  multi_hop:
    - path: "invoices to departments"
      via:  "invoices.po_id = purchase_orders.id AND purchase_orders.department_id = departments.id"
      description: "To find the department for an invoice, go through the linked purchase order."
```

**What is it?**
The `direct` list enumerates every foreign-key join path.  The `multi_hop` list explains joins that require going through an intermediate table — which the LLM would not be able to derive reliably from the DDL alone.

**How the code reads it — `__init__()` (line 123)**

```python
_rel = self._cfg.get("relationships", {})
if isinstance(_rel, dict):
    self.direct_joins: list[dict] = _rel.get("direct", [])
    self.multi_hops:   list[dict] = _rel.get("multi_hop", [])
```

**How it appears in the prompt — `build_prompt_context()` (line 212)**

```python
lines.append("\n=== JOIN RELATIONSHIPS ===")
for rel in self.direct_joins:
    lines.append(f"  {rel['from']} → {rel['to']}  ON  {rel['join']}")
for hop in self.multi_hops:
    lines.append(f"  [multi-hop] {hop.get('path', '')}")
    lines.append(f"    via: {hop.get('via', '')}")
```

Resulting prompt section:

```
=== JOIN RELATIONSHIPS ===
  invoices → vendors          ON  invoices.vendor_id = vendors.id
  invoices → purchase_orders  ON  invoices.po_id = purchase_orders.id
  payments → invoices         ON  payments.invoice_id = invoices.id
  [multi-hop] invoices to departments
    via: invoices.po_id = purchase_orders.id AND purchase_orders.department_id = departments.id
```

**Effect:** The question *"Show invoices for the Engineering department"* requires:
`invoices → purchase_orders → departments`
There is no direct foreign key from `invoices` to `departments`.  Without the multi_hop entry, the LLM would either get the join wrong or hallucinate a direct FK.  With it, the LLM is told the exact 2-table chain to use.

---

### Section 3 — `metrics`

```yaml
metrics:

  revenue:
    sql: "SUM(invoices.grand_total) FILTER (WHERE invoices.status = 'paid')"
    description: "Total value of fully paid invoices (INR)."
    synonyms: [income, earnings, total_paid, amount_paid]

  outstanding:
    sql: "SUM(invoices.grand_total) FILTER (WHERE invoices.status IN ('approved','validated','on_hold'))"
    description: "Total unpaid invoice value in the payment pipeline (INR)."
    synonyms: [unpaid, outstanding_amount, pending_payment, due, dues, amount_due]

  overdue:
    sql: "SUM(invoices.grand_total) FILTER (WHERE invoices.status IN ('approved','validated','on_hold') AND invoices.due_date < DATE('now'))"
    description: "Outstanding invoice value where the due date has already passed."
    synonyms: [past_due, late, overdue_amount]
```

**What is it?**
Pre-defined business aggregations.  Each metric has a canonical SQL fragment, a plain-English description, and synonyms that users might say instead.

**How the code reads it** — stored in `self.metrics` by `__init__()`.

**Two ways the code uses metrics:**

**A. `find_metric()` (line 144)** — direct lookup:

```python
def find_metric(self, name: str) -> dict | None:
    name_lower = name.lower()
    if name_lower in self.metrics:              # exact name match
        return self.metrics[name_lower]
    for m_name, m_def in self.metrics.items():  # synonym match
        if name_lower in [s.lower() for s in m_def.get("synonyms", [])]:
            return m_def
    return None
```

So `find_metric("income")` returns the full `revenue` metric dict because `income` is in its synonyms list.

**B. `build_prompt_context()` (line 221)** — injected into every prompt:

```python
lines.append("\n=== BUSINESS METRICS ===")
for m_name, m_def in self.metrics.items():
    lines.append(f"  {m_name}: {m_def.get('description', '')}")
    lines.append(f"    SQL fragment: {m_def.get('sql', '')}")
    syns = m_def.get("synonyms", [])
    if syns:
        lines.append(f"    Synonyms: {', '.join(syns)}")
```

Resulting prompt section:

```
=== BUSINESS METRICS ===
  revenue: Total value of fully paid invoices (INR).
    SQL fragment: SUM(invoices.grand_total) FILTER (WHERE invoices.status = 'paid')
    Synonyms: income, earnings, total_paid, amount_paid

  outstanding: Total unpaid invoice value in the payment pipeline (INR).
    SQL fragment: SUM(invoices.grand_total) FILTER (WHERE invoices.status IN ('approved','validated','on_hold'))
    Synonyms: unpaid, outstanding_amount, pending_payment, due, dues, amount_due
```

**Effect:** The question *"What was our revenue last quarter?"* maps to:
```sql
SELECT SUM(invoices.grand_total)
FROM invoices
WHERE invoices.status = 'paid'
  AND invoice_date BETWEEN '2025-10-01' AND '2025-12-31'
```
Without the metrics section, the LLM might generate `SUM(grand_total)` with no status filter — counting rejected and on-hold invoices as "revenue", which is wrong.

---

### Section 4 — `synonyms`

```yaml
synonyms:
  # Table aliases
  bills:    invoices
  bill:     invoices
  receipts: grns
  orders:   purchase_orders
  po:       purchase_orders

  # Status conditions  (these expand to WHERE clause fragments)
  unpaid:    "invoices.status IN ('received','validated','approved','on_hold')"
  outstanding: "invoices.status IN ('approved','validated','on_hold')"
  paid:      "invoices.status = 'paid'"
  overdue:   "invoices.due_date < DATE('now') AND invoices.status NOT IN ('paid','rejected')"
  watchlist: "vendors.is_watchlist = 1"
  high_risk: "vendors.is_watchlist = 1"
  blocked:   "vendors.is_watchlist = 1"

  # Column aliases
  value:  grand_total
  amount: grand_total
  total:  grand_total

  # Entity aliases
  client:   companies
  supplier: vendors
```

**What is it?**
A flat word → meaning dictionary at the global level (not per-table).  Two kinds:
- **Table aliases** — map casual words to table names
- **Condition expansions** — map one adjective to a full WHERE clause fragment

**How the code reads it** — `__init__()` builds a lower-case lookup map:

```python
self._syn_map: dict[str, str] = {
    k.lower(): v for k, v in self.synonyms.items()
}
```

**How it is used — `resolve_synonym()` (line 140):**

```python
def resolve_synonym(self, word: str) -> str | None:
    return self._syn_map.get(word.lower())
```

Example calls:
```python
sl.resolve_synonym("bills")      # → "invoices"
sl.resolve_synonym("unpaid")     # → "invoices.status IN ('received','validated','approved','on_hold')"
sl.resolve_synonym("watchlist")  # → "vendors.is_watchlist = 1"
sl.resolve_synonym("receipts")   # → "grns"
```

**How it appears in the prompt — `build_prompt_context()` (line 229):**

```python
lines.append("\n=== VOCABULARY / SYNONYMS ===")
for word, meaning in self.synonyms.items():
    lines.append(f"  '{word}' → {meaning}")
```

Resulting prompt section:
```
=== VOCABULARY / SYNONYMS ===
  'bills'      → invoices
  'unpaid'     → invoices.status IN ('received','validated','approved','on_hold')
  'watchlist'  → vendors.is_watchlist = 1
  'overdue'    → invoices.due_date < DATE('now') AND invoices.status NOT IN ('paid','rejected')
```

**Effect — two real question examples:**

| Question | Without synonyms | With synonyms |
|---|---|---|
| "Show all unpaid bills" | LLM guesses `status != 'paid'` (misses on_hold) | LLM sees exact `IN (...)` clause |
| "List watchlist vendors" | LLM might try `WHERE watchlist = true` | LLM uses `WHERE vendors.is_watchlist = 1` |

---

### Section 5 — `temporal`

```yaml
temporal:
  today:
    sql: "DATE('now')"

  this_month:
    start: "DATE('now', 'start of month')"
    end:   "DATE('now')"

  last_month:
    start: "DATE('now', 'start of month', '-1 month')"
    end:   "DATE('now', 'start of month', '-1 day')"

  this_quarter:
    description: "Current calendar quarter Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec"
    sql_start: "DATE(strftime('%Y','now') || '-' || CASE ... END || '-01')"

  last_quarter:
    description: "Previous calendar quarter"
    sql_hint: "Compute by subtracting 3 months from the start of the current quarter"

  this_year: ...
  last_year: ...
```

**What is it?**
Date-range definitions for common time expressions.  But notice: the YAML only defines *templates* — it does not compute actual dates.  The real computation happens in Python.

**How the code uses it — `TemporalResolver` class (line 30)**

`TemporalResolver` is a separate class that uses `date.today()` to calculate concrete ISO date strings at the moment a query runs:

```python
class TemporalResolver:
    def __init__(self) -> None:
        self._today: date = date.today()    # ← snapshot of today when engine starts

    def last_quarter(self) -> tuple[str, str]:
        """First → last day of the previous calendar quarter."""
        q_start_month = ((self._today.month - 1) // 3) * 3 + 1
        if q_start_month == 1:
            prev_month = 10
            prev_year  = self._today.year - 1
        else:
            prev_month = q_start_month - 3
            prev_year  = self._today.year
        start = date(prev_year, prev_month, 1)
        end   = date(self._today.year, q_start_month, 1) - timedelta(days=1)
        return start.isoformat(), end.isoformat()
```

The `SemanticLayer` class creates one `TemporalResolver` instance:

```python
self._temporal = TemporalResolver()
```

And exposes it as a property:

```python
@property
def temporal(self) -> TemporalResolver:
    return self._temporal
```

**How resolved dates appear in the prompt — `build_prompt_context()` (line 234):**

```python
td = self._temporal.as_context_dict()
lines.append(f"  today:        {td['today']}")
for key in ("this_month", "last_month", "this_quarter", "last_quarter", ...):
    start, end = td[key]
    lines.append(f"  {key:<14}: {start}  →  {end}")
```

Resulting prompt section (on 2026-03-27):

```
=== TEMPORAL EXPRESSIONS (resolved for today) ===
  today:          2026-03-27
  this_month    : 2026-03-01  →  2026-03-27
  last_month    : 2026-02-01  →  2026-02-28
  this_quarter  : 2026-01-01  →  2026-03-27
  last_quarter  : 2025-10-01  →  2025-12-31
  this_year     : 2026-01-01  →  2026-03-27
  last_year     : 2025-01-01  →  2025-12-31
```

**Effect:** The question *"How many invoices were raised last month?"* becomes:

```sql
SELECT COUNT(*) FROM invoices
WHERE invoice_date BETWEEN '2026-02-01' AND '2026-02-28'
```

The LLM doesn't have to figure out what "last month" means in SQLite date arithmetic — the exact date range is already in the prompt.

---

### Section 6 — `ambiguity_rules`

```yaml
ambiguity_rules:

  top_vendors:
    trigger: "top vendors"
    options:
      - { label: "by total invoice value", default: true,  sql_order: "SUM(invoices.grand_total) DESC" }
      - { label: "by invoice count",       default: false, sql_order: "COUNT(invoices.id) DESC" }
      - { label: "by payment received",    default: false, sql_order: "SUM(payments.amount) DESC" }
      - { label: "by vendor rating",       default: false, sql_order: "vendors.rating ASC" }
    default_assumption: "Assuming 'top vendors' means highest total invoice value."

  outstanding_vs_overdue:
    trigger: "outstanding"
    default_assumption: "Assuming 'outstanding' means invoices not yet paid (status in approved/validated/on_hold)."

  compare_quarters:
    trigger: "compare quarter"
    default_assumption: "Assuming comparison is between this quarter and the immediately preceding quarter."
```

**What is it?**
A list of known ambiguous phrasings.  Each rule has a `trigger` (the phrase that makes a question ambiguous) and a `default_assumption` (what the engine assumes when it decides not to ask the user).

**How the code uses it — `detect_ambiguity()` (line 163):**

```python
def detect_ambiguity(self, question: str) -> dict | None:
    q_words = set(re.findall(r"\w+", question.lower()))
    for rule in self.ambiguity_rules.values():
        trigger = rule.get("trigger", "").lower()
        if not trigger:
            continue
        if set(trigger.split()).issubset(q_words):   # ← all trigger words must appear
            return rule
    return None
```

The word-subset check means `trigger: "top vendors"` matches:
- ✅ *"Who are our top 5 vendors?"*  (has "top" and "vendors", "5" doesn't matter)
- ✅ *"Show me the top vendors by state"*
- ❌ *"Show vendor list"*  (missing "top")

**How the engine acts on it — `nlp_to_sql.py` line 63:**

```python
# Step 1 – Ambiguity detection
ambiguity_rule = self._semantic.detect_ambiguity(question)
if ambiguity_rule:
    result.assumption = ambiguity_rule.get("default_assumption", "")
    log.warning("ambiguity",
                f"Ambiguous question → {result.assumption[:80]}")
```

And in the LLM prompt:
```python
if ambiguity_rule:
    assumption = ambiguity_rule.get("default_assumption", "")
    ambiguity_note = (
        f"NOTE – AMBIGUOUS QUESTION: {assumption}\n"
        "Include this as a SQL comment: -- ASSUMPTION: <assumption>"
    )
```

**Effect:** For *"Who are our top 5 vendors?"* the LLM receives:

```
NOTE – AMBIGUOUS QUESTION: Assuming 'top vendors' means highest total invoice value.
Include this as a SQL comment: -- ASSUMPTION: <assumption>
```

And the generated SQL will include:
```sql
-- ASSUMPTION: Ordering by total invoice value (SUM of grand_total)
SELECT v.name, SUM(i.grand_total) AS total_value
FROM vendors v JOIN invoices i ON i.vendor_id = v.id
GROUP BY v.id ORDER BY total_value DESC
LIMIT 5
```

The Streamlit UI then shows a yellow warning box with the assumption so the user knows what was assumed.

---

## The complete data flow — one question traced end-to-end

**Question:** *"Show me all unpaid bills from last month"*

```
1. app.py  receives the question from the chat input
        │
        ▼
2. ConversationSession.ask()  passes it to NLPtoSQLEngine.query()
        │
        ▼
3. NLPtoSQLEngine.query()  calls:

   a) semantic.detect_ambiguity("Show me all unpaid bills from last month")
      → q_words = {show, me, all, unpaid, bills, from, last, month}
      → checks trigger "top vendors": {top, vendors} ⊄ q_words  → no match
      → checks trigger "outstanding": {outstanding} ⊄ q_words   → no match
      → returns None  (no ambiguity)

   b) cache.lookup("Show me all unpaid bills from last month")
      → cache miss (first time)

   c) _generate_sql() builds the prompt:
      ┌──────────────────────────────────────────────────────────┐
      │ [SYSTEM_PROMPT: rules for SQL generation]               │
      │                                                          │
      │ === DATABASE TABLES ===                                  │
      │ Table: invoices                                          │
      │   Synonyms: bill, bills, payable, ...   ← from tables   │
      │   Columns:                                               │
      │     status (enum) [received,validated,...,paid,...]: ... │
      │     invoice_date (date): Date printed on the invoice     │
      │     grand_total (decimal): Invoice value including taxes  │
      │                                                          │
      │ === JOIN RELATIONSHIPS ===                               │
      │   invoices → vendors  ON  invoices.vendor_id=vendors.id  │
      │   ...                                                    │
      │                                                          │
      │ === BUSINESS METRICS ===                                 │
      │   outstanding: ...                                       │
      │     SQL fragment: SUM(...) WHERE status IN (...)         │
      │                                                          │
      │ === VOCABULARY / SYNONYMS ===                            │
      │   'bills' → invoices                        ← key!      │
      │   'unpaid' → invoices.status IN (...)        ← key!     │
      │                                                          │
      │ === TEMPORAL EXPRESSIONS (resolved for today) ===       │
      │   last_month: 2026-02-01 → 2026-02-28        ← key!    │
      │                                                          │
      │ === QUESTION ===                                         │
      │ Show me all unpaid bills from last month                 │
      └──────────────────────────────────────────────────────────┘
        │
        ▼
   d) OpenAI GPT-4o reads the prompt and generates:
      ```sql
      SELECT i.invoice_number, v.name AS vendor, i.grand_total, i.status
      FROM invoices i
      JOIN vendors v ON v.id = i.vendor_id
      WHERE i.status IN ('received','validated','approved','on_hold')   ← from synonyms.unpaid
        AND i.invoice_date BETWEEN '2026-02-01' AND '2026-02-28'       ← from temporal.last_month
      ORDER BY i.grand_total DESC
      LIMIT 100
      ```

   e) validate_sql() → EXPLAIN passes ✅

   f) executor.execute() → 4 rows returned

   g) _generate_explanation() → "This query looks at the invoices table..."

   h) cache.store(question, sql, explanation)

4. NLPQueryResult returned to app.py with:
   - sql         = the SELECT above
   - query_result = 4 rows
   - explanation  = plain-English text
   - log          = 10 pipeline log entries
   - tokens_used  = ~1 400

5. Streamlit renders: KPI row, SQL expander, dataframe, explanation, logs
```

---

## Why not just use the raw DDL?

This is the most important question.  Here is a side-by-side comparison of what the LLM sees with and without the semantic layer for the same question:

**Question:** *"What was our revenue last quarter?"*

### Without semantic layer (raw DDL only)

```sql
CREATE TABLE invoices (
    id INTEGER PRIMARY KEY,
    invoice_number TEXT,
    vendor_id INTEGER,
    ...
    grand_total DECIMAL(15,2),
    status TEXT CHECK(status IN ('received','validated','approved',
                                 'rejected','on_hold','paid','partially_paid')),
    invoice_date DATE,
    ...
);
```

The LLM sees `status` is an enum, but has no idea which values mean "paid" in a business sense.  It might generate:

```sql
-- Wrong: includes partially_paid, approved, validated
SELECT SUM(grand_total) FROM invoices
WHERE invoice_date >= DATE('now', '-3 months')
```

### With semantic layer

The LLM also sees:

```
=== BUSINESS METRICS ===
  revenue: Total value of fully paid invoices (INR).
    SQL fragment: SUM(invoices.grand_total) FILTER (WHERE invoices.status = 'paid')
    Synonyms: income, earnings, total_paid, amount_paid

=== TEMPORAL EXPRESSIONS (resolved for today) ===
  last_quarter  : 2025-10-01  →  2025-12-31
```

And generates:

```sql
-- Correct: only status='paid', exact quarter dates
SELECT SUM(grand_total) AS revenue
FROM invoices
WHERE status = 'paid'
  AND invoice_date BETWEEN '2025-10-01' AND '2025-12-31'
```

---

## Summary — what each section prevents

| YAML Section | Problem it prevents |
|---|---|
| `tables.description` | LLM uses the wrong table for a concept |
| `tables.synonyms` | LLM can't map "bills" → `invoices` table |
| `tables.columns.values` | LLM hallucinates enum values that don't exist |
| `relationships.direct` | LLM gets the JOIN ON clause wrong |
| `relationships.multi_hop` | LLM can't navigate a 3-table path (invoice→PO→dept) |
| `metrics` | LLM defines "revenue" wrongly (wrong status filter) |
| `synonyms` (status conditions) | LLM uses wrong WHERE clause for "unpaid", "overdue" |
| `temporal` | LLM can't compute "last quarter" as exact dates |
| `ambiguity_rules` | LLM silently picks a random interpretation of vague terms |

---

## How to extend the semantic layer

The YAML is designed to be edited by domain experts without touching Python.

### Add a new table synonym

```yaml
tables:
  invoices:
    synonyms: [bill, bills, payable, payables, invoice, invoices, tax_invoice, voucher]
    #                                                                          ↑ add here
```

### Add a new business metric

```yaml
metrics:
  gst_liability:
    sql: "SUM(invoices.total_tax) FILTER (WHERE invoices.status IN ('approved','paid'))"
    description: "Total GST amount on approved and paid invoices."
    synonyms: [tax_due, gst_owed]
```

### Add a new ambiguity rule

```yaml
ambiguity_rules:
  top_departments:
    trigger: "top department"
    default_assumption: "Assuming 'top department' means highest total PO spend."
```

### Add a new synonym

```yaml
synonyms:
  rejected_invoices: "invoices.status = 'rejected'"
  high_value:        "invoices.grand_total > 1000000"
```

No code change required — `SemanticLayer.__init__()` picks up all changes on the next application restart.

---

## Key files at a glance

| File | Role |
|---|---|
| `semantic_layer.yaml` | The business dictionary — edited by domain experts |
| `engine/semantic_layer.py` | `SemanticLayer` class — reads YAML, builds prompt context, detects ambiguity; `TemporalResolver` — computes actual date ranges |
| `engine/nlp_to_sql.py` | `NLPtoSQLEngine` — calls `semantic.detect_ambiguity()` at step 1, passes `semantic.build_prompt_context()` into every LLM prompt |
| `engine/config.py` | `Settings.semantic_yaml` — the path to the YAML, configurable via `CASHFLO_YAML` env var |
