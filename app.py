#!/usr/bin/env python3
"""
OpenClaw Token Usage Dashboard
Parses session JSONL files, extracts token usage per LLM call, serves on configured port.

Configuration (all via environment variables):
  SESSIONS_DIR        Path to OpenClaw session JSONL files
                      Default: ~/.openclaw/agents/main/sessions
  DASHBOARD_TITLE     Browser title / header title
                      Default: OpenClaw Token Usage
  DASHBOARD_SUBTITLE  Optional subtitle shown in header (e.g. "My Instance")
                      Default: "" (empty)
  PORT                HTTP port to listen on
                      Default: 8081
  DB_PATH             SQLite database path (relative to script dir or absolute)
                      Default: tokens.db (next to app.py)
  INGEST_INTERVAL     Seconds between background ingest runs
                      Default: 300
  TZ_OFFSET_HOURS     Integer hours offset from UTC for date grouping in charts
                      Default: auto-detected from system timezone, fallback -7 (PDT)
"""

import json, os, re, sqlite3, threading, time, glob
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, render_template_string, request

# ── Configuration ────────────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))

SESSIONS_DIR    = os.path.expanduser(
    os.environ.get("SESSIONS_DIR", "~/.openclaw/agents/main/sessions")
)
DASHBOARD_TITLE    = os.environ.get("DASHBOARD_TITLE",    "OpenClaw Token Usage")
DASHBOARD_SUBTITLE = os.environ.get("DASHBOARD_SUBTITLE", "")
PORT            = int(os.environ.get("PORT",            "8081"))
DB_PATH         = os.environ.get("DB_PATH", os.path.join(_script_dir, "tokens.db"))
INGEST_INTERVAL = int(os.environ.get("INGEST_INTERVAL", "300"))

# Timezone offset: auto-detect from system, fall back to -7 (PDT)
try:
    _tz_auto = int(datetime.now().astimezone().utcoffset().total_seconds() / 3600)
except Exception:
    _tz_auto = -7
TZ_OFFSET_HOURS   = int(os.environ.get("TZ_OFFSET_HOURS", str(_tz_auto)))
TZ_OFFSET_SECONDS = TZ_OFFSET_HOURS * 3600

app = Flask(__name__)
ingest_lock = threading.Lock()

# ── Model pricing (per million tokens) ──────────────────────────────
MODEL_PRICING = {
    # Anthropic
    "claude-sonnet-4-6":  {"input": 5.00,  "output": 25.00},
    "claude-sonnet-4-5":  {"input": 5.00,  "output": 25.00},
    "claude-opus-4-6":    {"input": 15.00, "output": 75.00},
    "claude-opus-4-5":    {"input": 15.00, "output": 75.00},
    "claude-haiku-4-5":   {"input": 0.80,  "output": 4.00},
    # OpenAI
    "gpt-oss-20b":           {"input": 2.00,  "output": 8.00},
    "openai/gpt-oss-20b":    {"input": 2.00,  "output": 8.00},
    "gpt-5.2":               {"input": 2.00,  "output": 8.00},
    "gpt-5.3-codex":         {"input": 5.00,  "output": 20.00},
    # MiniMax (https://minimaxi.com/en/news/minimax-m2)
    "MiniMax-M2.5":              {"input": 0.30, "output": 1.20},
    "MiniMax-M2.5-highspeed":    {"input": 0.60, "output": 2.40},
    "MiniMax-M2.7":              {"input": 0.30, "output": 1.20},
    "MiniMax-M2.7-highspeed":    {"input": 0.60, "output": 2.40},
    "M2-her":                    {"input": 0.30, "output": 1.20},
    # Alibaba / Qwen
    "qwen3.6-plus":             {"input": 0.50, "output": 3.00},
    "qwen/qwen3.6-plus":        {"input": 0.50, "output": 3.00},
}

def cost_usd(model, input_tok, output_tok):
    p = MODEL_PRICING.get(model, {"input": 3.00, "output": 15.00})
    return (input_tok * p["input"] + output_tok * p["output"]) / 1_000_000

# ── DB ───────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ── Session type classification ────────────────────────────────────
def classify_session(session_id, sessions_dir=None):
    """Return a session type string for a session UUID by reading its first user message."""
    if sessions_dir is None:
        sessions_dir = SESSIONS_DIR
    # Try all file variants
    candidates = (
        glob.glob(os.path.join(sessions_dir, f"{session_id}.jsonl")) +
        glob.glob(os.path.join(sessions_dir, f"{session_id}.jsonl.*")) +
        glob.glob(os.path.join(sessions_dir, f"{session_id}-topic-*.jsonl"))
    )
    if not candidates:
        return "unknown"
    first_user_text = ""
    try:
        with open(candidates[0], errors="replace") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    if obj.get("type") != "message":
                        continue
                    msg = obj.get("message", {})
                    if msg.get("role") != "user":
                        continue
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        first_user_text = " ".join(
                            c.get("text", "") for c in content if isinstance(c, dict)
                        )
                    else:
                        first_user_text = str(content)
                    break
                except Exception:
                    continue
    except Exception:
        return "unknown"

    t = first_user_text[:500].lower()

    # Heartbeat sessions
    if "read heartbeat.md" in t or "heartbeat_ok" in t:
        # Try to figure out what heartbeat was doing by scanning further into the session
        try:
            with open(candidates[0], errors="replace") as fh:
                full_text = fh.read(60000).lower()
        except Exception:
            full_text = t
        if "email" in full_text or "inbox" in full_text or "waggle" in full_text or "unread" in full_text:
            return "heartbeat/email"
        if "calendar" in full_text or "cal.py" in full_text:
            return "heartbeat/calendar"
        if "github" in full_text:
            return "heartbeat/github"
        if "blog" in full_text:
            return "heartbeat/blog"
        return "heartbeat/general"

    # Subagent sessions
    if "subagent context" in t or "you are running as a subagent" in t:
        return "subagent"

    # Cron sessions
    cron_match = re.search(r"\[cron:([^\]]+)\]", first_user_text[:300])
    if cron_match:
        cron_name = cron_match.group(1).lower()
        if "email" in cron_name or "inbox" in cron_name:
            return "cron/email"
        if "github" in cron_name or "issue" in cron_name or "pr" in cron_name or "stale" in cron_name:
            return "cron/github"
        if "weather" in cron_name:
            return "cron/weather"
        if "market" in cron_name or "stock" in cron_name or "retirement" in cron_name:
            return "cron/market"
        if "cam" in cron_name or "sun" in cron_name or "timelapse" in cron_name or "sky" in cron_name or "sunset" in cron_name or "sunrise" in cron_name:
            return "cron/camera"
        if "briefing" in cron_name or "journal" in cron_name or "dream" in cron_name:
            return "cron/briefing"
        if "security" in cron_name:
            return "cron/security"
        if "usage" in cron_name or "report" in cron_name:
            return "cron/report"
        return "cron/other"

    # Thread history / Slack context
    if "thread history" in t or "slack" in t[:100]:
        return "chat/slack"

    # Direct chat messages
    if "telegram" in t[:200] or "message_id" in t[:200] or "sender_id" in t[:200]:
        return "chat/main"
    # Generic System messages (old format)
    if "system:" in t[:50] or "cron job" in t[:100]:
        return "cron/other"

    # Fallback: if it looks like a direct conversation
    if len(first_user_text) > 5:
        return "chat/main"
    return "unknown"


# Cache so we don't re-classify the same session repeatedly
_session_type_cache = {}

def get_session_type(session_id):
    if session_id not in _session_type_cache:
        _session_type_cache[session_id] = classify_session(session_id)
    return _session_type_cache[session_id]


def init_db():
    conn = get_db()
    # Create tables (without session_type for compatibility)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS token_runs (
            run_id       TEXT PRIMARY KEY,
            job_id       TEXT,
            session_id   TEXT,
            model        TEXT,
            provider     TEXT,
            ts           INTEGER,
            duration_ms  INTEGER,
            status       TEXT,
            input_tokens       INTEGER DEFAULT 0,
            output_tokens      INTEGER DEFAULT 0,
            total_tokens       INTEGER DEFAULT 0,
            cost_usd           REAL DEFAULT 0,
            cache_read_tokens  INTEGER DEFAULT 0,
            cache_write_tokens INTEGER DEFAULT 0,
            cost_input         REAL DEFAULT 0,
            cost_output        REAL DEFAULT 0,
            cost_cache_read    REAL DEFAULT 0,
            cost_cache_write   REAL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS ingest_state (
            log_file TEXT PRIMARY KEY,
            last_pos INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_ts ON token_runs(ts);
        CREATE INDEX IF NOT EXISTS idx_job ON token_runs(job_id);
        CREATE INDEX IF NOT EXISTS idx_model ON token_runs(model);
    """)
    # Add session_type column (migration-safe — ignored if already exists)
    try:
        conn.execute("ALTER TABLE token_runs ADD COLUMN session_type TEXT DEFAULT 'unknown'")
        conn.commit()
    except Exception:
        pass  # column already exists
    # Add index for session_type (separate from executescript so it's also migration-safe)
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session_type ON token_runs(session_type)")
        conn.commit()
    except Exception:
        pass
    # Add error_message column for 429/error tracking
    try:
        conn.execute("ALTER TABLE token_runs ADD COLUMN error_message TEXT DEFAULT ''")
        conn.commit()
    except Exception:
        pass
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON token_runs(status)")
        conn.commit()
    except Exception:
        pass
    conn.close()

# Models whose vLLM/local connector returns zeros — estimate from content length
LOCAL_ZERO_TOKEN_MODELS = {
    "Intel/Qwen3.5-397B-A17B-int4-AutoRound",
}
CHARS_PER_TOKEN = 4  # rough estimate for English/code

def estimate_tokens_from_content(content):
    """Estimate token count from message content when usage is zero.
    Includes thinking/reasoning blocks which are often the bulk of output.
    """
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                parts.append(c.get("text", ""))
                parts.append(c.get("thinking", ""))
                parts.append(c.get("reasoning", ""))
            else:
                parts.append(str(c))
        text = " ".join(p for p in parts if p)
    else:
        text = str(content) if content else ""
    chars = len(text)
    if chars == 0:
        return 0
    return max(1, chars // CHARS_PER_TOKEN)

# ── SESSION PARSING ──────────────────────────────────────────────────
def session_files():
    """Yield all session JSONL files: plain, .deleted.*, .reset.*, -topic-*.jsonl"""
    seen = set()
    for pattern in ["*.jsonl", "*.jsonl.deleted.*", "*.jsonl.reset.*"]:
        for f in sorted(glob.glob(os.path.join(SESSIONS_DIR, pattern))):
            if f not in seen and not f.endswith(".lock"):
                seen.add(f)
                yield f

def session_id_from_path(path):
    """Extract session UUID from any session file variant."""
    name = os.path.basename(path)
    name = re.sub(r'\.jsonl(\.deleted\.[^/]+|\.reset\.[^/]+)?$', '', name)
    name = re.sub(r'-topic-[\d.]+$', '', name)
    return name

def ingest_logs():
    """Ingest all session JSONL files from the openclaw sessions directory."""
    with ingest_lock:
        conn = get_db()
        for session_file in session_files():
            row = conn.execute(
                "SELECT last_pos FROM ingest_state WHERE log_file=?", (session_file,)
            ).fetchone()
            last_pos = row["last_pos"] if row else 0

            try:
                with open(session_file, "r", errors="replace") as f:
                    f.seek(last_pos)
                    session_id = session_id_from_path(session_file)
                    for line in f:
                        parse_session_line(conn, line.strip(), session_id)
                    new_pos = f.tell()
                conn.execute(
                    "INSERT OR REPLACE INTO ingest_state VALUES (?,?)",
                    (session_file, new_pos)
                )
            except Exception as e:
                print(f"[ingest] Error reading {session_file}: {e}")

        conn.commit()
        conn.close()

def parse_session_line(conn, line, session_id):
    """Parse one line from a session JSONL file. Each assistant message with usage = one LLM call."""
    if not line:
        return
    try:
        obj = json.loads(line)
    except Exception:
        return

    if obj.get("type") != "message":
        return

    msg = obj.get("message", {})
    usage = msg.get("usage")
    if not usage:
        return

    msg_id = obj.get("id", "")
    run_id = f"{session_id}:{msg_id}"
    model = msg.get("model", "unknown")
    # Infer provider from model name
    if "claude" in model:
        provider = "anthropic"
    elif "gpt" in model or "codex" in model:
        provider = "openai"
    else:
        provider = "unknown"

    # Parse timestamp
    ts_str = obj.get("timestamp", "")
    try:
        ts = int(datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp())
    except Exception:
        ts = 0

    input_raw  = usage.get("input", 0)
    output_tok = usage.get("output", 0)
    cache_read = usage.get("cacheRead", 0)
    cache_write= usage.get("cacheWrite", 0)
    input_tok  = input_raw
    total_tok  = usage.get("totalTokens", input_raw + output_tok + cache_read + cache_write)

    # Local vLLM models return all-zero usage structs — estimate from content length
    if model in LOCAL_ZERO_TOKEN_MODELS and output_tok == 0:
        content = msg.get("content", "")
        est = estimate_tokens_from_content(content)
        output_tok = est
        total_tok  = est
        input_tok  = 0

    # Use pre-calculated cost breakdown if available
    cost_obj = usage.get("cost", {}) or {}
    cost_input     = cost_obj.get("input", 0) or 0
    cost_output    = cost_obj.get("output", 0) or 0
    cost_cache_read  = cost_obj.get("cacheRead", 0) or 0
    cost_cache_write = cost_obj.get("cacheWrite", 0) or 0
    cost = cost_obj.get("total") if cost_obj else None
    if cost is None:
        cost = cost_usd(model, input_raw, output_tok)

    # Determine status from stopReason field
    stop_reason = msg.get("stopReason", "")
    error_msg = msg.get("errorMessage", "")
    if stop_reason == "error":
        status = "error"
    elif stop_reason == "cancelled":
        status = "cancelled"
    elif stop_reason:
        status = "ok"
    else:
        status = "ok"

    stype = get_session_type(session_id)

    try:
        conn.execute("""
            INSERT OR IGNORE INTO token_runs
              (run_id, job_id, session_id, model, provider, ts,
               duration_ms, status, input_tokens, output_tokens,
               total_tokens, cost_usd, session_type,
               cache_read_tokens, cache_write_tokens,
               cost_input, cost_output, cost_cache_read, cost_cache_write,
               error_message)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (run_id, session_id, session_id, model, provider, ts,
              0, status, input_tok, output_tok, total_tok, cost, stype,
              cache_read, cache_write,
              cost_input, cost_output, cost_cache_read, cost_cache_write,
              error_msg))
    except Exception:
        pass

# ── BACKGROUND INGEST ────────────────────────────────────────────────
def ingest_loop():
    while True:
        try:
            ingest_logs()
        except Exception as e:
            print(f"[ingest_loop] {e}")
        time.sleep(INGEST_INTERVAL)

# ── ROUTES ───────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        title=DASHBOARD_TITLE,
        subtitle=DASHBOARD_SUBTITLE,
    )

def parse_range():
    """Parse from_ts / to_ts from query params. Returns (from_ts, to_ts) as ints or None."""
    now = int(time.time())
    from_ts = request.args.get("from_ts", type=int, default=None)
    to_ts   = request.args.get("to_ts",   type=int, default=None)
    return from_ts, to_ts

def range_clause(from_ts, to_ts, col="ts"):
    """Build SQL WHERE clause fragment and params list for a time range."""
    clauses, params = [], []
    if from_ts is not None:
        clauses.append(f"{col} >= ?"); params.append(from_ts)
    if to_ts is not None:
        clauses.append(f"{col} <= ?"); params.append(to_ts)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params

# Helper: SQL expression for local date from UTC timestamp
_DATE_EXPR = f"strftime('%Y-%m-%d', datetime(ts + {TZ_OFFSET_SECONDS}, 'unixepoch'))"

@app.route("/api/summary")
def api_summary():
    conn = get_db()
    from_ts, to_ts = parse_range()
    where, params = range_clause(from_ts, to_ts)

    scoped = conn.execute(
        f"SELECT SUM(input_tokens) i, SUM(output_tokens) o, "
        f"SUM(total_tokens) t, SUM(cost_usd) c, COUNT(*) n, "
        f"SUM(cache_read_tokens) cr, SUM(cache_write_tokens) cw "
        f"FROM token_runs {where}", params
    ).fetchone()

    total = conn.execute(
        "SELECT SUM(input_tokens) i, SUM(output_tokens) o, "
        "SUM(total_tokens) t, SUM(cost_usd) c, COUNT(*) n, "
        "SUM(cache_read_tokens) cr, SUM(cache_write_tokens) cw FROM token_runs"
    ).fetchone()

    proj_from = from_ts or (int(time.time()) - 30 * 86400)
    proj_days = max(1, (int(time.time()) - proj_from) / 86400)
    proj = conn.execute(
        "SELECT SUM(cost_usd) c, COUNT(*) n FROM token_runs WHERE ts >= ?",
        (proj_from,)
    ).fetchone()
    monthly_est = round((proj["c"] or 0) / proj_days * 30, 2)

    conn.close()

    def row(r):
        return {
            "input": r["i"] or 0, "output": r["o"] or 0,
            "total": r["t"] or 0, "cost": round(r["c"] or 0, 4), "runs": r["n"] or 0,
            "cacheRead": r["cr"] or 0, "cacheWrite": r["cw"] or 0
        }
    return jsonify({
        "scoped": row(scoped),
        "allTime": row(total),
        "monthlyProjection": monthly_est
    })

@app.route("/api/daily")
def api_daily():
    conn = get_db()
    from_ts, to_ts = parse_range()
    where, params = range_clause(from_ts, to_ts)
    rows = conn.execute(f"""
        SELECT {_DATE_EXPR} AS day,
               SUM(input_tokens) AS input_tok,
               SUM(output_tokens) AS output_tok,
               SUM(total_tokens) AS total_tok,
               SUM(cost_usd) AS cost,
               COUNT(*) AS runs
        FROM token_runs {where}
        GROUP BY day ORDER BY day ASC
    """, params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/daily_runs")
def api_daily_runs():
    """Daily run counts by session type."""
    conn = get_db()
    from_ts, to_ts = parse_range()
    where, params = range_clause(from_ts, to_ts)
    rows = conn.execute(f"""
        SELECT {_DATE_EXPR} AS day,
               session_type,
               COUNT(*) AS runs
        FROM token_runs {where}
        GROUP BY day, session_type ORDER BY day ASC
    """, params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/by_model")
def api_by_model():
    conn = get_db()
    from_ts, to_ts = parse_range()
    where, params = range_clause(from_ts, to_ts)
    rows = conn.execute(f"""
        SELECT model, provider,
               SUM(input_tokens) AS input_tok,
               SUM(output_tokens) AS output_tok,
               SUM(total_tokens) AS total_tok,
               SUM(cost_usd) AS cost,
               COUNT(*) AS runs
        FROM token_runs {where}
        GROUP BY model ORDER BY total_tok DESC
    """, params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/daily_by_model")
def api_daily_by_model():
    conn = get_db()
    from_ts, to_ts = parse_range()
    where, params = range_clause(from_ts, to_ts)
    rows = conn.execute(f"""
        SELECT {_DATE_EXPR} AS day,
               model,
               SUM(total_tokens)       AS total_tok,
               SUM(input_tokens)       AS input_tok,
               SUM(output_tokens)      AS output_tok,
               SUM(cache_read_tokens)  AS cache_read_tok,
               SUM(cache_write_tokens) AS cache_write_tok,
               SUM(cost_usd) AS cost
        FROM token_runs {where}
        GROUP BY day, model ORDER BY day ASC
    """, params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/sonnet_cost_detail")
def api_sonnet_cost_detail():
    """Daily Sonnet cost broken down by input / output / cache-read / cache-write."""
    conn = get_db()
    from_ts, to_ts = parse_range()
    conditions = ["model IN ('claude-sonnet-4-6','claude-sonnet-4-5')"]
    params = []
    if from_ts:
        conditions.append("ts >= ?")
        params.append(from_ts)
    if to_ts:
        conditions.append("ts <= ?")
        params.append(to_ts)
    where = "WHERE " + " AND ".join(conditions)
    rows = conn.execute(f"""
        SELECT {_DATE_EXPR} AS day,
               SUM(cost_input)         AS cost_input,
               SUM(cost_output)        AS cost_output,
               SUM(cost_cache_read)    AS cost_cache_read,
               SUM(cost_cache_write)   AS cost_cache_write,
               SUM(cost_usd)           AS cost_total,
               SUM(input_tokens)       AS input_tok,
               SUM(output_tokens)      AS output_tok,
               SUM(cache_read_tokens)  AS cache_read_tok,
               SUM(cache_write_tokens) AS cache_write_tok
        FROM token_runs {where}
        GROUP BY day ORDER BY day ASC
    """, params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/daily_by_session_type")
def api_daily_by_session_type():
    conn = get_db()
    from_ts, to_ts = parse_range()
    where, params = range_clause(from_ts, to_ts)
    rows = conn.execute(f"""
        SELECT {_DATE_EXPR} AS day,
               session_type,
               SUM(total_tokens) AS total_tok,
               SUM(cost_usd) AS cost
        FROM token_runs {where}
        GROUP BY day, session_type ORDER BY day ASC
    """, params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/reclassify", methods=["POST"])
def api_reclassify():
    """Backfill session_type for all rows that still say 'unknown'."""
    conn = get_db()
    rows = conn.execute(
        "SELECT DISTINCT session_id FROM token_runs WHERE session_type = 'unknown' OR session_type IS NULL"
    ).fetchall()
    updated = 0
    for row in rows:
        sid = row["session_id"]
        stype = classify_session(sid)
        if stype != "unknown":
            conn.execute(
                "UPDATE token_runs SET session_type=? WHERE session_id=?",
                (stype, sid)
            )
            updated += 1
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "updated": updated})

@app.route("/api/daily_cost")
def api_daily_cost():
    conn = get_db()
    from_ts, to_ts = parse_range()
    where, params = range_clause(from_ts, to_ts)
    rows = conn.execute(f"""
        SELECT {_DATE_EXPR} AS day,
               model,
               SUM(cost_usd) AS cost
        FROM token_runs {where}
        GROUP BY day, model ORDER BY day ASC
    """, params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/by_job")
def api_by_job():
    conn = get_db()
    from_ts, to_ts = parse_range()
    where, params = range_clause(from_ts, to_ts)
    rows = conn.execute(f"""
        SELECT t.job_id,
               SUM(t.input_tokens) AS input_tok,
               SUM(t.output_tokens) AS output_tok,
               SUM(t.total_tokens) AS total_tok,
               SUM(t.cost_usd) AS cost,
               COUNT(*) AS runs,
               AVG(t.input_tokens) AS avg_in,
               AVG(t.output_tokens) AS avg_out,
               SUM(CASE WHEN t.status != 'ok' THEN 1 ELSE 0 END) AS errors,
               ROUND(CAST(SUM(t.output_tokens) AS REAL) /
                     NULLIF(SUM(t.input_tokens), 0), 1) AS efficiency,
               (SELECT model FROM token_runs r2
                WHERE r2.job_id = t.job_id
                GROUP BY model ORDER BY COUNT(*) DESC LIMIT 1) AS primary_model,
               COUNT(DISTINCT t.model) AS model_count
        FROM token_runs t
        {where}
        GROUP BY t.job_id ORDER BY total_tok DESC LIMIT 20
    """, params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/recent")
def api_recent():
    conn = get_db()
    from_ts, to_ts = parse_range()
    where, params = range_clause(from_ts, to_ts)
    rows = conn.execute(f"""
        SELECT run_id, job_id, model, provider, ts,
               input_tokens, output_tokens, total_tokens, cost_usd, status
        FROM token_runs {where} ORDER BY ts DESC LIMIT 100
    """, params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/ingest", methods=["POST"])
def force_ingest():
    ingest_logs()
    return jsonify({"ok": True})

@app.route("/api/rate_limits")
def rate_limits():
    """Rate limit (429) errors per model over time."""
    from_ts, to_ts = parse_range()
    params = []
    where = "WHERE status = 'error'"
    if from_ts:
        where += " AND ts >= ?"
        params.append(from_ts)
    if to_ts:
        where += " AND ts <= ?"
        params.append(to_ts)

    conn = get_db()
    rows = conn.execute(f"""
        SELECT model, provider,
               COUNT(*) as errors,
               SUM(CASE WHEN error_message LIKE '%429%' THEN 1 ELSE 0 END) as rate_limits,
               MIN(ts) as first_error,
               MAX(ts) as last_error
        FROM token_runs {where}
        GROUP BY model
        ORDER BY errors DESC
    """, params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/rate_limits_daily")
def rate_limits_daily():
    """Daily rate limit errors per model."""
    from_ts, to_ts = parse_range()
    params = []
    where = "WHERE status = 'error'"
    if from_ts:
        where += " AND ts >= ?"
        params.append(from_ts)
    if to_ts:
        where += " AND ts <= ?"
        params.append(to_ts)

    conn = get_db()
    rows = conn.execute(f"""
        SELECT model, provider,
               {_DATE_EXPR} as day,
               COUNT(*) as errors,
               SUM(CASE WHEN error_message LIKE '%429%' THEN 1 ELSE 0 END) as rate_limits,
               SUM(CASE WHEN error_message LIKE '%timeout%' OR error_message LIKE '%Timeout%' OR error_message LIKE '%timed out%' OR error_message LIKE '%ETIMEDOUT%' THEN 1 ELSE 0 END) as timeouts
        FROM token_runs {where}
        GROUP BY model, day
        ORDER BY day, model
    """, params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

# ── HTML TEMPLATE ─────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 14px; }
  header { background: #161b22; border-bottom: 1px solid #30363d; padding: 12px 24px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 18px; font-weight: 600; color: #58a6ff; }
  header span { color: #8b949e; font-size: 13px; }
  .refresh-btn { margin-left: auto; background: #21262d; border: 1px solid #30363d; color: #e6edf3; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; }
  .refresh-btn:hover { background: #30363d; }
  .range-bar { display: flex; align-items: center; gap: 8px; margin-left: 20px; }
  .preset { background: #21262d; border: 1px solid #30363d; color: #8b949e; padding: 5px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; }
  .preset:hover { color: #e6edf3; }
  .preset.active { background: #1f6feb; border-color: #1f6feb; color: #fff; }
  .sep { color: #30363d; }
  input[type=date] { background: #21262d; border: 1px solid #30363d; color: #e6edf3; padding: 5px 8px; border-radius: 6px; font-size: 12px; color-scheme: dark; }
  main { padding: 20px 24px; max-width: 1400px; margin: 0 auto; }
  .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .stat-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .stat-label { color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
  .stat-value { font-size: 24px; font-weight: 700; color: #e6edf3; }
  .stat-sub { color: #8b949e; font-size: 12px; margin-top: 4px; }
  .stat-card.blue .stat-value { color: #58a6ff; }
  .stat-card.green .stat-value { color: #3fb950; }
  .stat-card.orange .stat-value { color: #d29922; }
  .stat-card.purple .stat-value { color: #bc8cff; }
  .section { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .section h2 { font-size: 14px; font-weight: 600; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 16px; }
  .charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .panel-stack { display: flex; flex-direction: column; gap: 16px; margin-bottom: 0; }
  canvas { max-height: 260px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; color: #8b949e; font-weight: 500; padding: 8px 12px; border-bottom: 1px solid #30363d; font-size: 11px; text-transform: uppercase; }
  td { padding: 8px 12px; border-bottom: 1px solid #21262d; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #1c2128; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
  .badge-ok { background: #1a3a2a; color: #3fb950; }
  .badge-error { background: #3a1a1a; color: #f85149; }
  .mono { font-family: monospace; font-size: 12px; color: #8b949e; }
  .period-tabs { display: flex; gap: 8px; margin-bottom: 16px; }
  .period-tab { background: #21262d; border: 1px solid #30363d; color: #8b949e; padding: 5px 14px; border-radius: 6px; cursor: pointer; font-size: 12px; }
  .period-tab.active { background: #1f6feb; border-color: #1f6feb; color: #fff; }
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  .cost { color: #3fb950; }
</style>
</head>
<body>
<header>
  <h1>⚡ {{ title }}</h1>
  {% if subtitle %}<span>{{ subtitle }}</span>{% endif %}
  <div class="range-bar">
    <button class="preset active" data-preset="all" onclick="setPreset(this,'all')">All Time</button>
    <button class="preset" data-preset="30d" onclick="setPreset(this,'30d')">30D</button>
    <button class="preset" data-preset="7d" onclick="setPreset(this,'7d')">7D</button>
    <button class="preset" data-preset="today" onclick="setPreset(this,'today')">Today</button>
    <span class="sep">|</span>
    <input type="date" id="dateFrom" onchange="setCustomRange()">
    <span style="color:#8b949e">→</span>
    <input type="date" id="dateTo" onchange="setCustomRange()">
  </div>
  <button class="refresh-btn" onclick="loadAll()">↻ Refresh</button>
</header>
<main>
  <!-- Range label -->
  <div id="range-label" style="color:#8b949e;font-size:12px;margin-bottom:12px;"></div>

  <!-- Summary cards -->
  <div id="summary-grid" class="stat-grid" style="margin-bottom:16px;"></div>

  <div class="panel-stack">
  <!-- Daily chart + By Model chart -->
  <div class="charts-row">
    <div class="section">
      <h2>Daily Runs by Session Type</h2>
      <canvas id="dailyChart"></canvas>
    </div>
    <div class="section">
      <h2>By Model (Cost)</h2>
      <canvas id="modelChart"></canvas>
    </div>
  </div>

  <!-- Daily tokens by model -->
  <div class="section">
    <h2>Daily Token Usage by Model <span style="color:#8b949e;font-size:11px;font-weight:400;text-transform:none;">— ⚠ local model output estimated from content length (÷4 chars/token)</span></h2>
    <canvas id="dailyModelChart" style="max-height:260px;"></canvas>
  </div>

  <!-- Daily tokens by session type -->
  <div class="section">
    <h2>Daily Token Usage by Session Type <span style="color:#8b949e;font-size:11px;font-weight:400;text-transform:none;">— stacked by type, heartbeat broken down by sub-task</span></h2>
    <canvas id="sessionTypeChart" style="max-height:260px;"></canvas>
  </div>

  <!-- Sonnet Token Breakdown -->
  <div class="section">
    <h2>Sonnet Daily Token Breakdown <span style="color:#8b949e;font-size:11px;font-weight:400;text-transform:none;">— cache write $3.75/M &nbsp;|&nbsp; cache read $0.30/M &nbsp;|&nbsp; input $5/M &nbsp;|&nbsp; output $25/M</span></h2>
    <canvas id="dailyInputModelChart" style="max-height:280px;"></canvas>
  </div>

  <!-- Daily Output Tokens by Model (all models) -->
  <div class="section">
    <h2>Daily Output Token Usage by Model <span style="color:#8b949e;font-size:11px;font-weight:400;text-transform:none;">— output tokens by model</span></h2>
    <canvas id="dailyOutputModelChart" style="max-height:260px;"></canvas>
  </div>

  <!-- Sonnet Daily Cost -->
  <div class="section">
    <h2>Sonnet Daily Cost — Actual Breakdown <span style="color:#8b949e;font-size:11px;font-weight:400;text-transform:none;">— input $3/M &nbsp;|&nbsp; output $15/M &nbsp;|&nbsp; cache write $3.75/M &nbsp;|&nbsp; cache read $0.30/M</span></h2>
    <canvas id="costChart" style="max-height:260px;"></canvas>
  </div>

  <!-- Rate Limit / 429 panel -->
  <div class="section">
    <h2>Errors per Model — Rate Limits &amp; Timeouts (Daily)</h2>
    <canvas id="rateLimitChart"></canvas>
  </div>

  <!-- By Job table -->
  <div class="section">
    <h2>Top Token Consumers (by cron job)</h2>
    <table id="jobTable">
      <thead><tr>
        <th>Job ID</th>
        <th>Model</th>
        <th class="num">Runs</th>
        <th class="num">Errors</th>
        <th class="num">Input</th>
        <th class="num">Output</th>
        <th class="num">Context</th>
        <th class="num">Out/In ✦</th>
        <th class="num">Est. Cost</th>
      </tr></thead>
      <tbody></tbody>
    </table>
    <p style="color:#8b949e;font-size:11px;margin-top:8px;">✦ Output/Input ratio — higher means more tokens generated per prompt token (efficiency proxy)</p>
  </div>

  <!-- Recent runs -->
  <div class="section">
    <h2>Recent Runs</h2>

    <table id="recentTable">
      <thead><tr>
        <th>Time</th>
        <th>Job</th>
        <th>Model</th>
        <th class="num">Input</th>
        <th class="num">Output</th>
        <th class="num">Context</th>
        <th class="num">Cost</th>
        <th>Status</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>
  </div><!-- end panel-stack -->
<footer style="text-align:center;padding:16px;color:#8b949e;font-size:12px;border-top:1px solid #21262d;margin-top:24px;">
  <span id="footer-text">© 2026 Sam Cox — OpenClaw Token Usage</span> · <a href="https://github.com/jasonacox-sam/openclaw-token-usage" style="color:#58a6ff;text-decoration:none;">GitHub</a>
</footer>
</main>

<script>
let dailyChart, modelChart, costChart, dailyModelChart, sessionTypeChart, rateLimitChart;

// ── Date range state ─────────────────────────────────────────────────
let rangeFromTs = null;
let rangeToTs = null;

function rangeParams() {
  const p = new URLSearchParams();
  if (rangeFromTs) p.set('from_ts', rangeFromTs);
  if (rangeToTs)   p.set('to_ts',   rangeToTs);
  return p.toString() ? '?' + p.toString() : '';
}

// Midnight local timestamp helper (uses browser timezone)
function midnightLocalTs() {
  const now = new Date();
  now.setHours(0, 0, 0, 0);
  return Math.floor(now.getTime() / 1000);
}

function setPreset(btn, preset) {
  document.querySelectorAll('.preset').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('dateFrom').value = '';
  document.getElementById('dateTo').value = '';
  const now = Math.floor(Date.now() / 1000);
  if      (preset === 'today') { rangeFromTs = midnightLocalTs(); rangeToTs = null; }
  else if (preset === '7d')    { rangeFromTs = midnightLocalTs() - 6*86400; rangeToTs = null; }
  else if (preset === '30d')   { rangeFromTs = midnightLocalTs() - 29*86400; rangeToTs = null; }
  else                          { rangeFromTs = null; rangeToTs = null; }
  updateRangeLabel();
  loadAll();
}

function setCustomRange() {
  const from = document.getElementById('dateFrom').value;
  const to   = document.getElementById('dateTo').value;
  if (!from && !to) return;
  document.querySelectorAll('.preset').forEach(b => b.classList.remove('active'));
  rangeFromTs = from ? Math.floor(new Date(from).getTime()/1000) : null;
  rangeToTs   = to   ? Math.floor(new Date(to).getTime()/1000) + 86399 : null;
  updateRangeLabel();
  loadAll();
}

function updateRangeLabel() {
  const el = document.getElementById('range-label');
  if (!rangeFromTs && !rangeToTs) { el.textContent = 'Showing: All Time'; return; }
  const fmt2 = ts => ts ? new Date(ts*1000).toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric'}) : '…';
  el.textContent = `Showing: ${fmt2(rangeFromTs)} → ${fmt2(rangeToTs || Math.floor(Date.now()/1000))}`;
}

// ── Formatters ───────────────────────────────────────────────────────
function fmt(n) {
  if (!n) return '0';
  if (n >= 1000000) return (n/1000000).toFixed(2) + 'M';
  if (n >= 1000) return (n/1000).toFixed(1) + 'K';
  return n.toString();
}
function fmtCost(c) { return '$' + (c||0).toFixed(4); }
function fmtTime(ts) {
  return new Date(ts * 1000).toLocaleString('en-US', {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'});
}
function shortJob(id) {
  if (!id) return '?';
  return id.length > 20 ? id.substring(0,8) + '…' : id;
}

// ── Data loaders ─────────────────────────────────────────────────────
async function loadSummary() {
  const d = await fetch('/api/summary' + rangeParams()).then(r=>r.json());
  const grid = document.getElementById('summary-grid');
  const s = d.scoped, all = d.allTime;
  const isFiltered = rangeFromTs || rangeToTs;
  grid.innerHTML = `
    <div class="stat-card blue">
      <div class="stat-label">${isFiltered ? 'Period' : 'All-Time'} Input</div>
      <div class="stat-value">${fmt(s.input)}</div>
      <div class="stat-sub">${s.runs} runs</div>
    </div>
    <div class="stat-card purple">
      <div class="stat-label">${isFiltered ? 'Period' : 'All-Time'} Completion</div>
      <div class="stat-value">${fmt(s.output)}</div>
      <div class="stat-sub">generated tokens</div>
    </div>
    <div class="stat-card orange">
      <div class="stat-label">${isFiltered ? 'Period' : 'All-Time'} Cache</div>
      <div class="stat-value">${fmt((s.cacheRead||0) + (s.cacheWrite||0))}</div>
      <div class="stat-sub">read + write</div>
    </div>
    <div class="stat-card green">
      <div class="stat-label">${isFiltered ? 'Period' : 'All-Time'} Cost</div>
      <div class="stat-value">${fmtCost(s.cost)}</div>
      <div class="stat-sub">estimated</div>
    </div>
    <div class="stat-card orange">
      <div class="stat-label">30-Day Projection</div>
      <div class="stat-value">$${d.monthlyProjection}</div>
      <div class="stat-sub">at current rate</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">All-Time Cost</div>
      <div class="stat-value">${fmtCost(all.cost)}</div>
      <div class="stat-sub">${all.runs} runs</div>
    </div>
  `;
}

async function loadDaily() {
  const rows = await fetch('/api/daily_runs' + rangeParams()).then(r=>r.json());

  const daySet  = [...new Set(rows.map(r => r.day))].sort();
  const typeSet = [...new Set(rows.map(r => r.session_type || 'unknown'))].sort();

  const typeConfig = {
    'chat/main':           { label: '💬 Main Chat',           bg: '#58a6ff99', border: '#58a6ff' },
    'chat/slack':          { label: '💬 Slack',               bg: '#4493f899', border: '#4493f8' },
    'heartbeat/email':     { label: '💓 Heartbeat — Email',   bg: '#f0883e99', border: '#f0883e' },
    'heartbeat/calendar':  { label: '💓 Heartbeat — Cal',     bg: '#ffa65799', border: '#ffa657' },
    'heartbeat/github':    { label: '💓 Heartbeat — GitHub',  bg: '#d2992299', border: '#d29922' },
    'heartbeat/blog':      { label: '💓 Heartbeat — Blog',    bg: '#e3b34199', border: '#e3b341' },
    'heartbeat/general':   { label: '💓 Heartbeat',           bg: '#d2992299', border: '#d29922' },
    'cron/email':          { label: '⏰ Cron — Email',         bg: '#3fb95099', border: '#3fb950' },
    'cron/github':         { label: '⏰ Cron — GitHub',        bg: '#2ea04399', border: '#2ea043' },
    'cron/weather':        { label: '⏰ Cron — Weather',       bg: '#79c0ff99', border: '#79c0ff' },
    'cron/market':         { label: '⏰ Cron — Market',        bg: '#bc8cff99', border: '#bc8cff' },
    'cron/camera':         { label: '⏰ Cron — Camera',        bg: '#a371f799', border: '#a371f7' },
    'cron/briefing':       { label: '⏰ Cron — Briefing',      bg: '#56d36499', border: '#56d364' },
    'cron/security':       { label: '⏰ Cron — Security',      bg: '#f8514999', border: '#f85149' },
    'cron/report':         { label: '⏰ Cron — Report',        bg: '#89929b99', border: '#89929b' },
    'cron/other':          { label: '⏰ Cron — Other',         bg: '#8957e599', border: '#bc8cff' },
    'subagent':            { label: '🤖 Subagent',             bg: '#ff7b7299', border: '#ff7b72' },
    'unknown':             { label: '❓ Unknown',               bg: '#30363d99', border: '#8b949e' },
  };
  const fallback = (i) => ([
    { bg: '#a371f799', border: '#a371f7' },
    { bg: '#ffa65799', border: '#ffa657' },
    { bg: '#f8514999', border: '#f85149' },
  ])[i % 3];

  const datasets = typeSet.map((t, i) => {
    const cfg = typeConfig[t] || fallback(i);
    return {
      label: cfg.label || t,
      data: daySet.map(day => {
        const match = rows.find(r => r.day === day && (r.session_type || 'unknown') === t);
        return match ? match.runs : 0;
      }),
      backgroundColor: cfg.bg,
      borderColor: cfg.border,
      borderWidth: 1,
    };
  });

  if (dailyChart) dailyChart.destroy();
  const ctx = document.getElementById('dailyChart').getContext('2d');
  dailyChart = new Chart(ctx, {
    type: 'bar',
    data: { labels: daySet.map(d => d.slice(5)), datasets },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: {
        legend: { labels: { color: '#8b949e', boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          mode: 'index',
          callbacks: {
            afterBody: (items) => {
              const total = items.reduce((s, i) => s + i.raw, 0);
              return [`────────`, `Total: ${total} runs`];
            },
            label: ctx => ` ${ctx.dataset.label}: ${ctx.raw} runs`
          }
        }
      },
      scales: {
        x: { stacked: true, ticks: { color: '#8b949e', maxTicksLimit: 16 }, grid: { color: '#21262d' } },
        y: { stacked: true, ticks: { color: '#8b949e' }, grid: { color: '#21262d' } }
      }
    }
  });
}

async function loadModelChart() {
  const rows = await fetch('/api/by_model' + rangeParams()).then(r=>r.json());
  const sorted = rows.filter(r => (r.cost||0) > 0).sort((a,b) => (b.cost||0) - (a.cost||0));
  const labels = sorted.map(r => r.model || 'unknown');
  const data   = sorted.map(r => parseFloat((r.cost||0).toFixed(4)));

  const modelColorMap = {
    'claude-sonnet-4-6': '#58a6ff',
    'claude-sonnet-4-5': '#79c0ff',
    'claude-opus-4-6':   '#bc8cff',
    'claude-opus-4-5':   '#a371f7',
    'claude-haiku-4-5':  '#4493f8',
    'MiniMax-M2.5':      '#3fb950',
    'gpt-oss-20b':       '#e3b341',
    'openai/gpt-oss-20b':'#d29922',
    'gpt-5.2':           '#ffa657',
    'gpt-5.3-codex':     '#f0883e',
    'qwen':              '#ff7b72',
  };
  const fallback = ['#f85149','#56d364','#e3b341','#79c0ff','#a371f7','#ffa657'];
  const bgColors = labels.map((l,i) => modelColorMap[l] || fallback[i % fallback.length]);

  if (modelChart) modelChart.destroy();
  const ctx = document.getElementById('modelChart').getContext('2d');
  modelChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{ data, backgroundColor: bgColors, borderWidth: 2, borderColor: '#161b22' }]
    },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: {
        legend: { position: 'right', labels: { color: '#8b949e', boxWidth: 12, font: { size: 12 } } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.label}: $${ctx.raw.toFixed(4)}` } }
      }
    }
  });
}

// Per-model pricing for client-side display
const MODEL_PRICING = {
  'claude-sonnet-4-6':  { input: 5.00,  output: 25.00 },
  'claude-sonnet-4-5':  { input: 3.00,  output: 15.00 },
  'claude-opus-4-6':    { input: 15.00, output: 75.00 },
  'MiniMax-M2.7':       { input: 0.30,  output: 1.20  },
};
function modelPricing(model) {
  return MODEL_PRICING[model] || { input: 3.00, output: 15.00 };
}

async function loadCostChart() {
  const rows = await fetch('/api/sonnet_cost_detail' + rangeParams()).then(r=>r.json());
  const daySet = rows.map(r => r.day);

  function fmtTok(v) { return v >= 1e6 ? (v/1e6).toFixed(1)+'M' : v >= 1e3 ? (v/1e3).toFixed(0)+'K' : String(v); }

  const datasets = [
    {
      label: 'Cache write ($3.75/M)',
      data: rows.map(r => parseFloat((r.cost_cache_write||0).toFixed(4))),
      backgroundColor: '#f0883e',
      borderColor: '#ffa657',
      borderWidth: 1,
    },
    {
      label: 'Input ($3/M)',
      data: rows.map(r => parseFloat((r.cost_input||0).toFixed(4))),
      backgroundColor: '#1f4e8599',
      borderColor: '#58a6ff',
      borderWidth: 1,
    },
    {
      label: 'Output ($15/M)',
      data: rows.map(r => parseFloat((r.cost_output||0).toFixed(4))),
      backgroundColor: '#58a6ff',
      borderColor: '#79c0ff',
      borderWidth: 1,
    },
    {
      label: 'Cache read ($0.30/M)',
      data: rows.map(r => parseFloat((r.cost_cache_read||0).toFixed(4))),
      backgroundColor: '#2ea04399',
      borderColor: '#3fb950',
      borderWidth: 1,
    },
  ];

  if (costChart) costChart.destroy();
  const ctx = document.getElementById('costChart').getContext('2d');
  costChart = new Chart(ctx, {
    type: 'bar',
    data: { labels: daySet.map(d => d.slice(5)), datasets },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: {
        legend: { labels: { color: '#8b949e', boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          mode: 'index',
          callbacks: {
            afterBody: (items) => {
              const row = rows[items[0].dataIndex];
              return [
                `─────────────────`,
                `Total: $${(row.cost_total||0).toFixed(4)}`,
                `Cache write: ${fmtTok(row.cache_write_tok||0)} tok`,
                `Input: ${fmtTok((row.input_tok||0)-(row.cache_read_tok||0)-(row.cache_write_tok||0))} tok`,
                `Output: ${fmtTok(row.output_tok||0)} tok`,
                `Cache read: ${fmtTok(row.cache_read_tok||0)} tok`,
              ];
            },
            label: ctx => ` ${ctx.dataset.label}: $${ctx.raw.toFixed(4)}`
          }
        }
      },
      scales: {
        x: { stacked: true, ticks: { color: '#8b949e', maxTicksLimit: 16 }, grid: { color: '#21262d' } },
        y: { stacked: true, ticks: { color: '#8b949e', callback: v => '$' + v.toFixed(2) }, grid: { color: '#21262d' } }
      }
    }
  });
}

async function loadJobs() {
  const rows = await fetch('/api/by_job' + rangeParams()).then(r=>r.json());
  const tbody = document.querySelector('#jobTable tbody');
  tbody.innerHTML = rows.map(r => {
    const modelLabel = r.primary_model
      ? (r.model_count > 1
          ? `${r.primary_model} <span style="color:#8b949e;font-size:11px" title="Used ${r.model_count} models">(+${r.model_count-1})</span>`
          : r.primary_model)
      : '?';
    const errCell = r.errors > 0
      ? `<td class="num" style="color:#f85149">${r.errors}</td>`
      : `<td class="num" style="color:#8b949e">0</td>`;
    const eff = r.efficiency ? r.efficiency.toFixed(1) + '×' : '—';
    return `
    <tr>
      <td class="mono">${shortJob(r.job_id)}</td>
      <td style="font-size:12px">${modelLabel}</td>
      <td class="num">${r.runs}</td>
      ${errCell}
      <td class="num">${fmt(r.input_tok)}</td>
      <td class="num">${fmt(r.output_tok)}</td>
      <td class="num">${fmt(r.total_tok)}</td>
      <td class="num" style="color:#d29922">${eff}</td>
      <td class="num cost">${fmtCost(r.cost)}</td>
    </tr>`;
  }).join('');
}

async function loadRecent() {
  const rows = await fetch('/api/recent' + rangeParams()).then(r=>r.json());
  const tbody = document.querySelector('#recentTable tbody');
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td class="mono">${fmtTime(r.ts)}</td>
      <td class="mono">${shortJob(r.job_id)}</td>
      <td>${r.model || '?'}</td>
      <td class="num">${fmt(r.input_tokens)}</td>
      <td class="num">${fmt(r.output_tokens)}</td>
      <td class="num">${fmt(r.total_tokens)}</td>
      <td class="num cost">${fmtCost(r.cost_usd)}</td>
      <td><span class="badge ${r.status === 'ok' ? 'badge-ok' : 'badge-error'}">${r.status}</span></td>
    </tr>
  `).join('');
}

async function loadDailyByModel() {
  const rows = await fetch('/api/daily_by_model' + rangeParams()).then(r=>r.json());

  const daySet   = [...new Set(rows.map(r => r.day))].sort();
  const modelSet = [...new Set(rows.map(r => r.model || 'unknown'))].sort();

  const modelColors = {
    'claude-sonnet-4-6': { bg: '#58a6ff99', border: '#58a6ff' },
    'claude-sonnet-4-5': { bg: '#79c0ff99', border: '#79c0ff' },
    'claude-opus-4-6':   { bg: '#bc8cff99', border: '#bc8cff' },
    'claude-opus-4-5':   { bg: '#a371f799', border: '#a371f7' },
    'claude-haiku-4-5':  { bg: '#4493f899', border: '#4493f8' },
    'MiniMax-M2.5':      { bg: '#3fb95099', border: '#3fb950' },
    'MiniMax-M2.7':      { bg: '#2ea04399', border: '#2ea043' },
    'gpt-oss-20b':       { bg: '#e3b34199', border: '#e3b341' },
    'openai/gpt-oss-20b':{ bg: '#d2992299', border: '#d29922' },
    'glm-5':             { bg: '#ff7b7299', border: '#ff7b72' },
    'Intel/Qwen3.5-397B-A17B-int4-AutoRound': { bg: '#56d36499', border: '#56d364' },
    'qwen':              { bg: '#ffa65799', border: '#ffa657' },
  };
  const fallbackColors = [
    { bg: '#f8514999', border: '#f85149' },
    { bg: '#a371f799', border: '#a371f7' },
    { bg: '#ffa65799', border: '#ffa657' },
  ];

  const datasets = modelSet.map((model, i) => {
    const colors = modelColors[model] || fallbackColors[i % fallbackColors.length];
    const data = daySet.map(day => {
      const match = rows.find(r => r.day === day && (r.model || 'unknown') === model);
      return match ? (match.total_tok || 0) : 0;
    });
    const label = model.replace('Intel/Qwen3.5-397B-A17B-int4-AutoRound', 'Qwen-397B (local★)');
    return { label, data, backgroundColor: colors.bg, borderColor: colors.border, borderWidth: 1 };
  });

  if (dailyModelChart) dailyModelChart.destroy();
  const ctx = document.getElementById('dailyModelChart').getContext('2d');
  dailyModelChart = new Chart(ctx, {
    type: 'bar',
    data: { labels: daySet.map(d => d.slice(5)), datasets },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: {
        legend: { labels: { color: '#8b949e', boxWidth: 12, font: { size: 11 } } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.dataset.label}: ${fmt(ctx.raw)} tok` } }
      },
      scales: {
        x: { stacked: true, ticks: { color: '#8b949e', maxTicksLimit: 16 }, grid: { color: '#21262d' } },
        y: { stacked: true, ticks: { color: '#8b949e', callback: v => fmt(v) }, grid: { color: '#21262d' } }
      }
    }
  });
}

let dailyInputModelChart, dailyOutputModelChart;

async function loadDailyInputByModel() {
  const rows = await fetch('/api/sonnet_cost_detail' + rangeParams()).then(r=>r.json());
  const daySet = rows.map(r => r.day);

  const datasets = [
    {
      label: 'Cache write (\u00d73.75x vs input)',
      data: rows.map(r => r.cache_write_tok || 0),
      backgroundColor: '#f0883e',
      borderColor: '#ffa657',
      borderWidth: 1,
    },
    {
      label: 'Cache read (\u00d70.06x vs input)',
      data: rows.map(r => r.cache_read_tok || 0),
      backgroundColor: '#2ea04399',
      borderColor: '#3fb950',
      borderWidth: 1,
    },
    {
      label: 'Raw input ($5/M)',
      data: rows.map(r => {
        const raw = (r.input_tok||0) - (r.cache_read_tok||0) - (r.cache_write_tok||0);
        return Math.max(0, raw);
      }),
      backgroundColor: '#1f4e8599',
      borderColor: '#58a6ff',
      borderWidth: 1,
    },
    {
      label: 'Output ($25/M)',
      data: rows.map(r => r.output_tok || 0),
      backgroundColor: '#58a6ff',
      borderColor: '#79c0ff',
      borderWidth: 1,
    },
  ];

  if (dailyInputModelChart) dailyInputModelChart.destroy();
  const ctx = document.getElementById('dailyInputModelChart').getContext('2d');
  dailyInputModelChart = new Chart(ctx, {
    type: 'bar',
    data: { labels: daySet.map(d => d.slice(5)), datasets },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: {
        legend: { labels: { color: '#8b949e', boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          mode: 'index',
          callbacks: {
            afterBody: (items) => {
              const row = rows[items[0].dataIndex];
              const total = (row.input_tok||0) + (row.output_tok||0);
              return [`────────`, `Total: ${fmt(total)} tok`, `Cost: $${(row.cost_total||0).toFixed(2)}`];
            },
            label: ctx => ` ${ctx.dataset.label}: ${fmt(ctx.raw)}`
          }
        }
      },
      scales: {
        x: { stacked: true, ticks: { color: '#8b949e', maxTicksLimit: 16 }, grid: { color: '#21262d' } },
        y: { stacked: true, ticks: { color: '#8b949e', callback: v => fmt(v) }, grid: { color: '#21262d' } }
      }
    }
  });
}

async function loadDailyOutputByModel() {
  const rows = await fetch('/api/daily_by_model' + rangeParams()).then(r=>r.json());

  const daySet   = [...new Set(rows.map(r => r.day))].sort();
  const modelSet = [...new Set(rows.map(r => r.model || 'unknown'))].sort();

  const modelColors = {
    'claude-sonnet-4-6': { bg: '#58a6ff99', border: '#58a6ff' },
    'claude-sonnet-4-5': { bg: '#79c0ff99', border: '#79c0ff' },
    'claude-opus-4-6':   { bg: '#bc8cff99', border: '#bc8cff' },
    'MiniMax-M2.7':      { bg: '#2ea04399', border: '#2ea043' },
    'glm-5':             { bg: '#ff7b7299', border: '#ff7b72' },
    'Intel/Qwen3.5-397B-A17B-int4-AutoRound': { bg: '#56d36499', border: '#56d364' },
  };
  const fallbackColors = [
    { bg: '#f8514999', border: '#f85149' },
    { bg: '#a371f799', border: '#a371f7' },
    { bg: '#89929b99', border: '#89929b' },
  ];

  const datasets = modelSet.map((model, i) => {
    const colors = modelColors[model] || fallbackColors[i % fallbackColors.length];
    const data = daySet.map(day => {
      const match = rows.find(r => r.day === day && (r.model || 'unknown') === model);
      return match ? (match.output_tok || 0) : 0;
    });
    const label = model.replace('Intel/Qwen3.5-397B-A17B-int4-AutoRound', 'Qwen-397B (local★)');
    return { label, data, backgroundColor: colors.bg, borderColor: colors.border, borderWidth: 1 };
  });

  if (dailyOutputModelChart) dailyOutputModelChart.destroy();
  const ctx = document.getElementById('dailyOutputModelChart').getContext('2d');
  dailyOutputModelChart = new Chart(ctx, {
    type: 'bar',
    data: { labels: daySet.map(d => d.slice(5)), datasets },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: {
        legend: { labels: { color: '#8b949e', boxWidth: 12, font: { size: 11 } } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.dataset.label}: ${fmt(ctx.raw)} out-tok` } }
      },
      scales: {
        x: { stacked: true, ticks: { color: '#8b949e', maxTicksLimit: 16 }, grid: { color: '#21262d' } },
        y: { stacked: true, ticks: { color: '#8b949e', callback: v => fmt(v) }, grid: { color: '#21262d' } }
      }
    }
  });
}

async function loadSessionTypeChart() {
  const rows = await fetch('/api/daily_by_session_type' + rangeParams()).then(r=>r.json());

  const daySet  = [...new Set(rows.map(r => r.day))].sort();
  const typeSet = [...new Set(rows.map(r => r.session_type || 'unknown'))].sort();

  const typeConfig = {
    'chat/main':          { label: '💬 Main Chat',          bg: '#58a6ff99', border: '#58a6ff' },
    'chat/slack':         { label: '💬 Slack Chat',         bg: '#4493f899', border: '#4493f8' },
    'heartbeat/email':    { label: '💓 Heartbeat — Email',  bg: '#f0883e99', border: '#f0883e' },
    'heartbeat/calendar': { label: '💓 Heartbeat — Calendar',bg: '#ffa65799', border: '#ffa657' },
    'heartbeat/github':   { label: '💓 Heartbeat — GitHub', bg: '#d2992299', border: '#d29922' },
    'heartbeat/blog':     { label: '💓 Heartbeat — Blog',   bg: '#e3b34199', border: '#e3b341' },
    'heartbeat/general':  { label: '💓 Heartbeat — General',bg: '#ff7b7299', border: '#ff7b72' },
    'cron/email':         { label: '⏰ Cron — Email',        bg: '#3fb95099', border: '#3fb950' },
    'cron/github':        { label: '⏰ Cron — GitHub',       bg: '#2ea04399', border: '#2ea043' },
    'cron/weather':       { label: '⏰ Cron — Weather',      bg: '#79c0ff99', border: '#79c0ff' },
    'cron/market':        { label: '⏰ Cron — Market',       bg: '#bc8cff99', border: '#bc8cff' },
    'cron/camera':        { label: '⏰ Cron — Camera',       bg: '#a371f799', border: '#a371f7' },
    'cron/briefing':      { label: '⏰ Cron — Briefing',     bg: '#56d36499', border: '#56d364' },
    'cron/security':      { label: '⏰ Cron — Security',     bg: '#f8514999', border: '#f85149' },
    'cron/report':        { label: '⏰ Cron — Report',       bg: '#89929b99', border: '#89929b' },
    'cron/other':         { label: '⏰ Cron — Other',        bg: '#30363d99', border: '#484f58' },
    'subagent':           { label: '🤖 Subagent',            bg: '#8957e599', border: '#8957e5' },
    'unknown':            { label: '❓ Unknown',             bg: '#21262d99', border: '#30363d' },
  };
  const fallback = { bg: '#6e768199', border: '#6e7681' };

  const datasets = typeSet.map(stype => {
    const cfg = typeConfig[stype] || { ...fallback, label: stype };
    const data = daySet.map(day => {
      const match = rows.find(r => r.day === day && (r.session_type || 'unknown') === stype);
      return match ? (match.total_tok || 0) : 0;
    });
    return {
      label: cfg.label,
      data,
      backgroundColor: cfg.bg,
      borderColor: cfg.border,
      borderWidth: 1,
    };
  });

  if (sessionTypeChart) sessionTypeChart.destroy();
  const ctx = document.getElementById('sessionTypeChart').getContext('2d');
  sessionTypeChart = new Chart(ctx, {
    type: 'bar',
    data: { labels: daySet.map(d => d.slice(5)), datasets },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: {
        legend: { labels: { color: '#8b949e', boxWidth: 12, font: { size: 11 } } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.dataset.label}: ${fmt(ctx.raw)} tok` } }
      },
      scales: {
        x: { stacked: true, ticks: { color: '#8b949e', maxTicksLimit: 16 }, grid: { color: '#21262d' } },
        y: { stacked: true, ticks: { color: '#8b949e', callback: v => fmt(v) }, grid: { color: '#21262d' } }
      }
    }
  });
}

async function loadRateLimits() {
  const data = await fetch('/api/rate_limits_daily' + rangeParams()).then(r => r.json());

  const MODEL_MAP = [
    { key: 'glm-5',               label: 'GLM-5 (Z.ai)',   color: '#f78166' },
    { key: 'claude-sonnet-4-6',   label: 'Sonnet 4.6',     color: '#58a6ff' },
    { key: 'claude-sonnet-4-5',   label: 'Sonnet 4.5',     color: '#79c0ff' },
    { key: 'MiniMax-M2.7',        label: 'MiniMax M2.7',   color: '#d2a8ff' },
    { key: 'MiniMax-M2.5',        label: 'MiniMax M2.5',   color: '#bc8cff' },
    { key: 'Intel/Qwen3.5-397B-A17B-int4-AutoRound', label: 'Local Qwen 397B', color: '#3fb950' },
  ];

  const daysSet = new Set(data.map(d => d.day));
  const days = Array.from(daysSet).sort();

  const datasets = [];
  for (const m of MODEL_MAP) {
    const byDay = {};
    data.filter(d => d.model === m.key).forEach(d => { byDay[d.day] = d.errors; });
    const vals = days.map(day => byDay[day] || 0);
    if (vals.some(v => v > 0)) {
      datasets.push({
        label: m.label,
        data: vals,
        borderColor: m.color,
        backgroundColor: m.color + '22',
        tension: 0.3,
        pointRadius: 3,
        borderWidth: 2,
        fill: false,
      });
    }
  }

  // Catch any model with 429s not in the static map
  const mappedKeys = new Set(MODEL_MAP.map(m => m.key));
  const otherModels = [...new Set(data.filter(d => d.errors > 0 && !mappedKeys.has(d.model)).map(d => d.model))];
  const extraColors = ['#e3b341','#ffa657','#76e3ea'];
  otherModels.forEach((model, i) => {
    const byDay = {};
    data.filter(d => d.model === model && d.errors > 0).forEach(d => { byDay[d.day] = d.errors; });
    const vals = days.map(day => byDay[day] || 0);
    if (vals.some(v => v > 0)) {
      datasets.push({
        label: model.length > 24 ? model.substring(0, 22) + '…' : model,
        data: vals,
        borderColor: extraColors[i % extraColors.length],
        backgroundColor: extraColors[i % extraColors.length] + '22',
        tension: 0.3,
        pointRadius: 3,
        borderWidth: 2,
        fill: false,
      });
    }
  });

  const ctx = document.getElementById('rateLimitChart').getContext('2d');
  if (rateLimitChart) rateLimitChart.destroy();
  rateLimitChart = new Chart(ctx, {
    type: 'line',
    data: { labels: days, datasets },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#8b949e', boxWidth: 12 } },
        tooltip: {
          callbacks: {
            title: items => items[0].label,
            label: item => ` ${item.dataset.label}: ${item.parsed.y} errors`,
          }
        }
      },
      scales: {
        x: { ticks: { color: '#8b949e', maxRotation: 45 }, grid: { color: '#21262d' } },
        y: {
          ticks: { color: '#8b949e', stepSize: 1 },
          grid: { color: '#21262d' },
          title: { display: true, text: 'Errors (429s, timeouts, etc.)', color: '#8b949e' },
          beginAtZero: true,
        }
      }
    }
  });
}

async function loadAll() {
  await Promise.all([loadSummary(), loadDaily(), loadModelChart(), loadDailyByModel(), loadDailyInputByModel(), loadDailyOutputByModel(), loadSessionTypeChart(), loadCostChart(), loadJobs(), loadRecent(), loadRateLimits()]);
}

updateRangeLabel();
loadAll();
setInterval(loadAll, 60000);
</script>
</body>
</html>
"""

# ── MAIN ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    ingest_logs()  # initial ingest
    t = threading.Thread(target=ingest_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=PORT, debug=False)
