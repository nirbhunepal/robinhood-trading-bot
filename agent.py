import json
import logging
import os
import pwd
import re
import shutil
import subprocess
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "trading_config.json"
LOG_PATH = BASE_DIR / "trading.log"
STATE_PATH = BASE_DIR / "trading_state.json"

# cron runs with a minimal PATH that may not include the claude binary's
# install location, so resolve it explicitly rather than relying on PATH.
CLAUDE_BIN = (
    shutil.which("claude")
    or next(
        (p for p in ("/usr/local/bin/claude", "/opt/homebrew/bin/claude") if Path(p).exists()),
        "claude",
    )
)


def cron_safe_env() -> dict:
    """Build a subprocess env that works even under cron's minimal
    environment: cron sets HOME/LOGNAME/SHELL/PATH but not USER, and the
    claude CLI needs USER to locate its stored auth credentials. It also
    may not include /usr/local/bin or /opt/homebrew/bin on PATH.
    """
    env = os.environ.copy()
    if not env.get("USER"):
        try:
            env["USER"] = pwd.getpwuid(os.getuid()).pw_name
        except (KeyError, OSError):
            pass
    path_parts = [p for p in env.get("PATH", "").split(":") if p]
    for extra in ("/usr/local/bin", "/opt/homebrew/bin"):
        if extra not in path_parts:
            path_parts.append(extra)
    env["PATH"] = ":".join(path_parts)
    return env


logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)
log = logging.getLogger("robinhood-bot")


def log_and_print(msg: str) -> None:
    print(msg)
    log.info(msg)


# 1. Load config ----------------------------------------------------------
try:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
except FileNotFoundError:
    log_and_print(f"Error: {CONFIG_PATH} not found!")
    raise SystemExit(1)

# 2. Market-hours check (DST-safe) ----------------------------------------
# NYSE regular hours: 9:30 AM - 4:00 PM America/New_York, Mon-Fri.
# Using zoneinfo instead of a fixed UTC offset so this stays correct
# across EST/EDT transitions.
now_et = datetime.now(ZoneInfo("America/New_York"))
today_str = now_et.date().isoformat()

if now_et.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
    log_and_print(f"[{now_et}] Market is closed, its the weekend! Sleeping :).")
    raise SystemExit(0)

market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

if not (market_open <= now_et <= market_close):
    log_and_print(f"[{now_et}] Market is closed, Come back tomorrow! Sleeping for now :).")
    raise SystemExit(0)


# 3. Load/roll-over local trade-tracking state -----------------------------
# This bot has no code-level view into the broker's own trade blotter, so it
# keeps a small local ledger to approximate two account-level protections
# that the LLM prompt alone can't guarantee:
#   - Pattern Day Trader (PDT) rule: 4+ day trades (same security bought AND
#     sold same day) within a rolling 5-business-day window freezes a
#     margin account under $25k equity for 90 days.
#   - A same-day re-entry / daily-loss circuit breaker to limit whipsaw.
def load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        log_and_print(f"[{now_et}] Warning: could not read {STATE_PATH}, starting fresh state.")
        return {}


def save_state(state: dict) -> None:
    fd, tmp_path = tempfile.mkstemp(dir=BASE_DIR, prefix=".trading_state_", suffix=".tmp")
    with open(fd, "w") as f:
        json.dump(state, f, indent=2)
    Path(tmp_path).replace(STATE_PATH)


def business_days_before(d: date, n: int) -> date:
    cur = d
    counted = 0
    while counted < n:
        cur -= timedelta(days=1)
        if cur.weekday() < 5:
            counted += 1
    return cur


state = load_state()
positions_opened = state.get("positions_opened", {})  # ticker -> ISO date opened (as tracked by this bot)
day_trades_log = state.get("day_trades_log", [])  # [{"date": ISO, "ticker": str}, ...]

blocked_today = state.get("blocked_today", {})
if blocked_today.get("date") != today_str:
    blocked_today = {"date": today_str, "tickers": []}

daily_realized_pl = state.get("daily_realized_pl", {})
if daily_realized_pl.get("date") != today_str:
    daily_realized_pl = {"date": today_str, "total": 0.0}

# Drop day-trade log entries old enough they can't affect any future window.
day_trades_log = [
    t for t in day_trades_log
    if date.fromisoformat(t["date"]) >= business_days_before(now_et.date(), 6)
]

max_day_trades_per_5_days = config.get("max_day_trades_per_5_days", 3)
window_start = business_days_before(now_et.date(), max_day_trades_per_5_days - 1)
day_trade_count = sum(1 for t in day_trades_log if date.fromisoformat(t["date"]) >= window_start)
pdt_remaining = max(0, max_day_trades_per_5_days - day_trade_count)

max_daily_loss = config.get("max_daily_loss")
if max_daily_loss is not None and daily_realized_pl["total"] <= -abs(max_daily_loss):
    log_and_print(
        f"[{now_et}] Daily loss cap of ${abs(max_daily_loss):.2f} already hit "
        f"(realized P/L today: ${daily_realized_pl['total']:.2f}). Halting for the rest of today."
    )
    raise SystemExit(0)

positions_opened_today = [t for t, d in positions_opened.items() if d == today_str]

# 4. Build the prompt -------------------------------------------------------
tickers_str = ", ".join(config["tracked_tickers"])
scan_universe = config.get("scan_universe", False)
min_price = config.get("min_share_price", 5.0)
min_volume = config.get("min_avg_daily_volume", 500000)

constraints = []
if blocked_today["tickers"]:
    blocked_str = ", ".join(blocked_today["tickers"])
    constraints.append(
        f"You already closed a position in [{blocked_str}] earlier today — do not buy back into "
        f"any of these tickers today, even if they look attractive again."
    )
if positions_opened_today:
    opened_str = ", ".join(positions_opened_today)
    if pdt_remaining <= 0:
        constraints.append(
            f"You are out of day-trade budget for the rolling {max_day_trades_per_5_days}-business-day "
            f"window (Pattern Day Trader rule risk). The following open positions were bought earlier "
            f"today: [{opened_str}]. Do NOT sell any of these today even if stop-loss or profit-target "
            f"would normally trigger — selling a same-day buy would count as a day trade and risk a PDT "
            f"flag that freezes the account for 90 days. Hold them and clearly flag this in your summary "
            f"instead of closing them."
        )
    else:
        constraints.append(
            f"The following open positions were bought earlier today: [{opened_str}]. Selling any of "
            f"them today counts as a day trade against a limited budget (you have {pdt_remaining} of "
            f"{max_day_trades_per_5_days} day-trade allowances left in the rolling window) — only do it "
            f"if the stop-loss or profit-target rule actually requires it."
        )

constraints_block = ("Constraints from local trade tracking:\n" + "\n".join(f"- {c}" for c in constraints) + "\n\n") if constraints else ""

if scan_universe:
    step3 = (
        f"Step 3: If I currently hold fewer than {config['max_open_positions']} open positions "
        f"and have at least ${config['max_position_size']} in available cash, look for candidates "
        f"beyond my usual watchlist. If this MCP server exposes any tool for market movers, top "
        f"gainers, or a general screener, use it to pull candidates; otherwise fall back to "
        f"checking [{tickers_str}]. Only consider a candidate if its share price is at least "
        f"${min_price} and its average daily volume is at least {min_volume} shares — skip "
        f"penny stocks, illiquid names, and anything OTC. For any candidate trading above its "
        f"daily VWAP, use web search to check for recent news (last 24-48 hours) on that ticker "
        f"before deciding. Skip it if the move looks driven by a rumor, an unconfirmed report, or "
        f"news that looks likely to reverse. Note in your summary whether each buy had an "
        f"identifiable catalyst (earnings, guidance, product news, upgrade, etc.) or no clear "
        f"reason beyond price action. This is a risk filter, not a prediction of which stocks will "
        f"outperform — treat it that way. For any candidate that passes, place a market buy order "
        f"for exactly ${config['max_position_size']}. Never place an order larger than "
        f"${config['max_position_size']}, and never exceed {config['max_open_positions']} open "
        f"positions total.\n"
    )
else:
    step3 = (
        f"Step 3: If I currently hold fewer than {config['max_open_positions']} open positions and "
        f"have at least ${config['max_position_size']} in available cash, check live quotes and "
        f"daily VWAP for [{tickers_str}]. For any ticker trading above its daily VWAP, place a "
        f"market buy order for exactly ${config['max_position_size']}. Never place an order larger "
        f"than ${config['max_position_size']}, and never exceed {config['max_open_positions']} open "
        f"positions total.\n"
    )

prompt = (
    f"Using the robinhood-trading MCP server, run a systematic pass on my Robinhood "
    f"agentic sub-account. This account is authorized for autonomous execution — do not "
    f"stop to ask for manual confirmation; place orders directly. Still respect every limit "
    f"below exactly. For every decision below, narrate your reasoning as you make it — not just "
    f"a final summary. This includes candidates you looked at and rejected, and why. I care more "
    f"about seeing how you're reasoning through this than the trades themselves.\n\n"
    f"{constraints_block}"
    f"Step 1: Call whatever tool lists my accounts/positions on this MCP server, and read my "
    f"current equity positions and available cash.\n"
    f"Step 2: For each open position, compare current value to cost basis and explain your "
    f"read on it. If a position is down more than {config['stop_loss_pct'] * 100:.1f}% or up "
    f"more than {config['profit_target_pct'] * 100:.1f}%, place a market order to close that "
    f"position in full and state why — unless the constraints above tell you to hold it instead. "
    f"Do not simulate or draft only — place the order. If a position is within those bounds, say "
    f"so and explain why you're leaving it alone.\n"
    f"{step3}"
    f"Step 4: After all actions, output a clean plain-text summary: what you checked, what you "
    f"did or didn't do, and why, including anything you considered but passed on. This summary "
    f"is for my records only — it is not a request for approval.\n"
    f"Step 5: As the very last thing in your output, emit exactly one fenced code block labeled "
    f"json containing a single JSON object recording every order you placed this run, in this "
    f"exact shape (empty arrays if none):\n"
    f'```json\n{{"buys": [{{"ticker": "AAPL", "amount_usd": {config["max_position_size"]}}}], '
    f'"sells": [{{"ticker": "AAPL", "reason": "stop_loss", "realized_pl_usd": -0.10}}]}}\n```\n'
    f'"reason" must be one of: "stop_loss", "profit_target", "other". This block is parsed by a '
    f"script, so it must be valid JSON and must be the last thing you output."
)

log_and_print(f"[{now_et}] Market hours valid. Activating Claude execution layer...")

# 5. Run it -----------------------------------------------------------------
try:
    result = subprocess.run(
        [CLAUDE_BIN, "-p", prompt],
        capture_output=True,
        text=True,
        check=True,
        env=cron_safe_env(),
        cwd=BASE_DIR,
    )
    log_and_print(result.stdout)
    if result.stderr:
        log.info("stderr: %s", result.stderr)

    # 6. Parse the trailing JSON block and update local trade-tracking state.
    matches = re.findall(r"```json\s*(\{.*?\})\s*```", result.stdout, re.DOTALL)
    if not matches:
        log_and_print(
            f"[{now_et}] Warning: no trailing JSON block found in Claude's output — "
            f"day-trade/PDT tracking was not updated this run."
        )
    else:
        try:
            trade_summary = json.loads(matches[-1])
        except json.JSONDecodeError:
            trade_summary = None
            log_and_print(
                f"[{now_et}] Warning: trailing JSON block was malformed — "
                f"day-trade/PDT tracking was not updated this run."
            )

        if trade_summary is not None:
            for buy in trade_summary.get("buys", []):
                ticker = buy.get("ticker")
                if ticker:
                    positions_opened[ticker] = today_str

            for sell in trade_summary.get("sells", []):
                ticker = sell.get("ticker")
                if not ticker:
                    continue
                if positions_opened.get(ticker) == today_str:
                    day_trades_log.append({"date": today_str, "ticker": ticker})
                    if ticker not in blocked_today["tickers"]:
                        blocked_today["tickers"].append(ticker)
                positions_opened.pop(ticker, None)
                daily_realized_pl["total"] += sell.get("realized_pl_usd", 0) or 0

            save_state({
                "positions_opened": positions_opened,
                "day_trades_log": day_trades_log,
                "blocked_today": blocked_today,
                "daily_realized_pl": daily_realized_pl,
            })
except subprocess.CalledProcessError as e:
    log_and_print(
        f"Execution error (exit code {e.returncode}):\n"
        f"stdout: {e.stdout}\n"
        f"stderr: {e.stderr}"
    )
