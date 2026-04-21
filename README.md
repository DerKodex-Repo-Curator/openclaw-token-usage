# OpenClaw Token Usage Dashboard

A self-hosted dashboard that tracks LLM token usage, costs, and rate limits from [OpenClaw](https://github.com/openclaw/openclaw) session logs. Dark-themed, zero-config, single-file Flask app.

![Dashboard screenshot showing token usage charts, cost breakdowns, and rate limit monitoring with a dark GitHub-style theme](https://your-screenshot-url-here)

## Features

- **Cost tracking** — per-model cost estimation with configurable pricing
- **Daily charts** — runs, tokens, and cost over time by session type and model
- **Rate limit monitoring** — 429 error tracking per model with time series chart
- **Session classification** — auto-detects cron jobs, heartbeats, and conversation types
- **Cache-aware** — tracks cache read/write tokens separately (Anthropic-style breakdown)
- **Real-time** — background ingest parses session logs every 5 minutes
- **Dark theme** — GitHub-dark styled UI with Chart.js visualizations
- **Date range filter** — Today / 7D / 30D / All Time / Custom range with timezone awareness

## Quick Start

```bash
pip install flask
python3 app.py
```

Open http://localhost:8081 — that's it.

The dashboard auto-discovers your OpenClaw session files and starts ingesting token usage data immediately.

## Configuration

All configuration via environment variables (sensible defaults, no config file needed):

| Variable | Default | Description |
|---|---|---|
| `SESSIONS_DIR` | `~/.openclaw/agents/main/sessions` | Path to OpenClaw session JSONL files |
| `DASHBOARD_TITLE` | `OpenClaw Token Usage` | Browser tab and header title |
| `DASHBOARD_SUBTITLE` | *(empty)* | Optional subtitle in header (e.g. `"my-instance"`) |
| `PORT` | `8081` | HTTP listen port |
| `DB_PATH` | `tokens.db` | SQLite database (relative to script dir or absolute) |
| `INGEST_INTERVAL` | `300` | Seconds between background session log scans |
| `TZ_OFFSET_HOURS` | *auto-detect* | UTC offset for date grouping in charts (e.g. `-7` for PDT) |

### Example

```bash
export DASHBOARD_SUBTITLE="production"
export TZ_OFFSET_HOURS=-5
python3 app.py
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/summary` | Cost/token/run totals + monthly projection |
| `GET /api/daily` | Daily aggregates (tokens, cost, runs) |
| `GET /api/daily_runs` | Daily run counts by session type |
| `GET /api/by_model` | Aggregates per model (tokens, cost, runs) |
| `GET /api/daily_by_model` | Daily tokens per model |
| `GET /api/daily_cost` | Daily cost breakdown |
| `GET /api/by_job` | Aggregates per cron job/session |
| `GET /api/recent` | Last 100 runs with status |
| `GET /api/rate_limits` | Error/429 counts per model |
| `GET /api/rate_limits_daily` | Daily rate limit errors per model |
| `POST /ingest` | Force immediate re-ingest |

All `GET` endpoints support `?from_ts=UNIX&to_ts=UNIX` for date range filtering.

## How It Works

1. **Ingest** — A background thread scans OpenClaw session JSONL files, extracts every assistant message with a `usage` block, and stores it in SQLite
2. **Classify** — Sessions are auto-classified by type (heartbeat, cron job, conversation) based on message content patterns
3. **Serve** — Flask serves the dashboard HTML and JSON APIs on the configured port
4. **Track errors** — Failed API calls (429 rate limits, etc.) are captured from `stopReason: "error"` in session logs

## Model Pricing

Built-in pricing for popular models (per million tokens). Edit the `MODEL_PRICING` dict in `app.py` to add or adjust:

- Anthropic: Claude Sonnet 4.5/4.6, Opus 4.5/4.6, Haiku 4.5
- OpenAI: GPT-5.2, GPT-5.3 Codex
- Z.ai: GLM-5
- Alibaba: Qwen 3.6 Plus
- MiniMax: M2.5, M2.7
- Local models (Qwen 397B, etc.) — output tokens estimated from content length

## License

MIT — see [LICENSE](LICENSE).
