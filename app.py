# app.py (FULL UPDATED)
from __future__ import annotations

import os
import re
import json
import sqlite3
import base64
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from flask import Flask, render_template, jsonify, request, session, url_for, redirect
from werkzeug.security import check_password_hash, generate_password_hash

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ============================================================
# App config
# ============================================================
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-in-real-app")
# Session cookie defaults that work well in local dev
app.config.setdefault("SESSION_COOKIE_SAMESITE", "Lax")
app.config.setdefault("SESSION_COOKIE_HTTPONLY", True)
DB_PATH = os.getenv("HORIZON_DB_PATH", "horizon.db")

# Emails that can moderate the public library (remove any scenario)
MODERATOR_EMAILS = set([
    e.strip().lower() for e in os.getenv("HORIZON_MODERATOR_EMAILS", "scott@test.com").split(",")
    if e.strip()
])

def is_moderator_session() -> bool:
    email = (session.get("user_email") or "").strip().lower()
    return bool(email) and (email in MODERATOR_EMAILS)

GENERATED_DIR = os.path.join(app.root_path, "static", "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

DEBUG_SCENARIO = os.getenv("HORIZON_DEBUG_SCENARIO", "1") == "1"


def _dbg(msg: str):
    if DEBUG_SCENARIO:
        print(f"[HORIZON_DEBUG] {msg}", flush=True)


# ============================================================
# DB helpers
# ============================================================
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_column(conn, table: str, column: str, coldef_sql: str):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r["name"] for r in cur.fetchall()]
    if column not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coldef_sql}")
        conn.commit()


def init_db():
    with get_conn() as conn:
        cur = conn.cursor()

        # USERS
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT,
            role TEXT NOT NULL,
            region TEXT,
            interests TEXT,
            values_json TEXT,
            risk_level INTEGER,
            complexity TEXT
        )
        """)
        conn.commit()
        ensure_column(conn, "users", "password_hash", "TEXT")

        # SCENARIOS (generated content)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS scenarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            user_id INTEGER,
            input_json TEXT NOT NULL,
            scenario_json TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """)
        conn.commit()

        # cover + sharing for Library
        ensure_column(conn, "scenarios", "cover_image_url", "TEXT")
        ensure_column(conn, "scenarios", "is_public", "INTEGER NOT NULL DEFAULT 1")

        # RUNS (progress per scenario)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            user_id INTEGER,
            scenario_id INTEGER NOT NULL,
            current_step INTEGER NOT NULL DEFAULT 1,
            choices_json TEXT NOT NULL DEFAULT '[]',
            final_json TEXT,
            images_enabled INTEGER NOT NULL DEFAULT 0,
            images_json TEXT NOT NULL DEFAULT '{}',
            mbti_type TEXT,
            mbti_scores_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(scenario_id) REFERENCES scenarios(id)
        )
        """)
        conn.commit()

        # safe migrations
        ensure_column(conn, "runs", "images_enabled", "INTEGER NOT NULL DEFAULT 0")
        ensure_column(conn, "runs", "images_json", "TEXT NOT NULL DEFAULT '{}'")
        ensure_column(conn, "runs", "mbti_type", "TEXT")
        ensure_column(conn, "runs", "mbti_scores_json", "TEXT NOT NULL DEFAULT '{}'")

        # Step-level decision signals (Horizon MBTI)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS run_step_signals (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at TEXT NOT NULL,
          user_id INTEGER,
          run_id INTEGER NOT NULL,
          scenario_id INTEGER NOT NULL,
          step INTEGER NOT NULL,
          option_id TEXT NOT NULL,
          confidence INTEGER NOT NULL,
          primary_value TEXT NOT NULL,
          risk_tolerance INTEGER NOT NULL,
          oversight_preference INTEGER NOT NULL,
          governance_preference INTEGER NOT NULL,
          rationale TEXT,
          decision_ms INTEGER,
          axis_json TEXT NOT NULL DEFAULT '{}',
          UNIQUE(run_id, step)
        )
        """)
        conn.commit()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# Scenario schema + validation (includes views bullets)
# ============================================================
STK_ALLOWED = ["Government", "Company", "Workers", "Customers", "Experts", "Community"]

SCENARIO_JSON_SCHEMA: Dict[str, Any] = {
    "name": "horizon_5step_scenario",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "steps": {
                "type": "array",
                "minItems": 5,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "headline": {"type": "string", "minLength": 1},
                        "scenario": {"type": "string", "minLength": 1},
                        "views": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 6,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "stakeholder": {"type": "string", "enum": STK_ALLOWED},
                                    "bullets": {
                                        "type": "array",
                                        "minItems": 2,
                                        "maxItems": 4,
                                        "items": {"type": "string", "minLength": 1},
                                    },
                                },
                                "required": ["stakeholder", "bullets"],
                            },
                        },
                        "question": {"type": "string", "minLength": 1},
                        "options": {
                            "type": "array",
                            "minItems": 4,
                            "maxItems": 4,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "id": {"type": "string", "enum": ["A", "B", "C", "D"]},
                                    "label": {"type": "string", "minLength": 1},
                                    "summary": {"type": "string", "minLength": 1},
                                },
                                "required": ["id", "label", "summary"],
                            },
                        },
                        "final_prompt": {"type": "string", "minLength": 1},
                    },
                    "required": ["headline"],
                    "oneOf": [
                        {"required": ["headline", "scenario", "views", "question", "options"]},
                        {"required": ["headline", "final_prompt"]},
                    ],
                },
            },
        },
        "required": ["title", "steps"],
    },
}

FINAL_JSON_SCHEMA: Dict[str, Any] = {
    "name": "horizon_final_analysis",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verdict_title": {"type": "string", "minLength": 1},
            "dimension_impacts": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ethics_psych": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"score_1to10": {"type": "integer", "minimum": 1, "maximum": 10}},
                        "required": ["score_1to10"],
                    },
                    "economic": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"score_1to10": {"type": "integer", "minimum": 1, "maximum": 10}},
                        "required": ["score_1to10"],
                    },
                    "social": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"score_1to10": {"type": "integer", "minimum": 1, "maximum": 10}},
                        "required": ["score_1to10"],
                    },
                    "political": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"score_1to10": {"type": "integer", "minimum": 1, "maximum": 10}},
                        "required": ["score_1to10"],
                    },
                },
                "required": ["ethics_psych", "economic", "social", "political"],
            },
            "recommended_safeguards": {
                "type": "array",
                "minItems": 3,
                "maxItems": 8,
                "items": {"type": "string", "minLength": 1},
            },
        },
        "required": ["verdict_title", "dimension_impacts", "recommended_safeguards"],
    },
}


# ============================================================
# Normalization
# ============================================================
OPT_IDS = ["A", "B", "C", "D"]


def _sentences_count(s: str) -> int:
    if not isinstance(s, str):
        return 0
    s = s.strip()
    if not s:
        return 0
    parts = re.split(r"(?<=[.!?])\s+", s)
    return len([p for p in parts if p.strip()])


def _ensure_4_option_list(opts_any) -> Optional[List[dict]]:
    if isinstance(opts_any, dict):
        if all(k in opts_any for k in OPT_IDS):
            fixed = []
            for k in OPT_IDS:
                text = str(opts_any.get(k, "")).strip() or f"Option {k}"
                fixed.append({"id": k, "label": text[:60], "summary": text})
            return fixed
        return None

    if isinstance(opts_any, list) and opts_any and all(isinstance(x, str) for x in opts_any):
        fixed = []
        for i, k in enumerate(OPT_IDS):
            text = (opts_any[i] if i < len(opts_any) else f"Option {k}").strip()
            fixed.append({"id": k, "label": text[:60], "summary": text})
        return fixed

    if isinstance(opts_any, list) and all(isinstance(x, dict) for x in opts_any):
        by_id: Dict[str, dict] = {}
        for o in opts_any:
            oid = str(o.get("id", "")).strip().upper()
            if oid in OPT_IDS and oid not in by_id:
                by_id[oid] = o

        fixed = []
        for k in OPT_IDS:
            o = by_id.get(k) or {}
            label = str(o.get("label") or "").strip()
            summary = str(o.get("summary") or "").strip()
            if not summary and isinstance(o.get("text"), str):
                summary = o["text"].strip()
            if not label and summary:
                label = summary[:60]
            if not label:
                label = f"Option {k}"
            if not summary:
                summary = label
            fixed.append({"id": k, "label": label, "summary": summary})
        return fixed

    return None


def _normalize_views(views_any) -> List[dict]:
    if isinstance(views_any, dict):
        out_list = []
        for st in STK_ALLOWED:
            if st in views_any:
                bullets = views_any.get(st)
                if isinstance(bullets, str):
                    bullets = [bullets]
                if not isinstance(bullets, list):
                    bullets = [str(bullets)]
                bullets = [str(b).strip() for b in bullets if str(b).strip()]
                if len(bullets) >= 2:
                    out_list.append({"stakeholder": st, "bullets": bullets[:4]})
        views_any = out_list

    if isinstance(views_any, list):
        cleaned = []
        seen = set()
        for v in views_any:
            if not isinstance(v, dict):
                continue
            st = str(v.get("stakeholder") or "").strip()
            if st not in STK_ALLOWED or st in seen:
                continue
            bullets = v.get("bullets")
            if isinstance(bullets, str):
                bullets = [bullets]
            if not isinstance(bullets, list):
                bullets = []
            bullets = [str(b).strip() for b in bullets if str(b).strip()]
            if len(bullets) < 2:
                continue
            cleaned.append({"stakeholder": st, "bullets": bullets[:4]})
            seen.add(st)
        if len(cleaned) >= 3:
            return cleaned[:6]

    return [
        {"stakeholder": "Government", "bullets": ["Wants compliance and measurable safety outcomes.", "Concerned about public trust and incident response capacity."]},
        {"stakeholder": "Company", "bullets": ["Targets cost reduction and service quality improvements.", "Worries about integration cost, liability, and downtime."]},
        {"stakeholder": "Workers", "bullets": ["Concerned about income stability and job redesign.", "Asks for training, fair evaluation, and grievance channels."]},
    ]


def normalize_scenario_output(out: Any, payload: Optional[dict] = None) -> Any:
    if not isinstance(out, dict):
        return out

    payload = payload or {}

    if not isinstance(out.get("title"), str) or not out["title"].strip():
        steps = out.get("steps") if isinstance(out.get("steps"), list) else []
        first_head = ""
        if steps and isinstance(steps[0], dict):
            first_head = (steps[0].get("headline") or "").strip()

        job = (payload.get("job_title") or "a job").strip() or "a job"
        field = (payload.get("field") or "general").strip().lower()
        level = (payload.get("replacement_level") or "assist").strip().lower()
        level_txt = "Assisting" if level == "assist" else ("Partially Replacing" if level == "partial" else "Replacing")

        out["title"] = first_head or f"{field.title()} Scenario: AI {level_txt} {job}"

    steps = out.get("steps")
    if not isinstance(steps, list):
        return out

    steps = steps[:5] + [{}] * max(0, 5 - len(steps))
    out["steps"] = steps

    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            steps[i] = {"headline": f"Step {i+1}"}
            s = steps[i]

        if i == 4:
            s["headline"] = str(s.get("headline") or "Final Analysis").strip() or "Final Analysis"
            if not isinstance(s.get("final_prompt"), str) or not s["final_prompt"].strip():
                s["final_prompt"] = (
                    "Generate the final 4-dimension impact analysis based on the selected options (A/B/C/D) "
                    "for steps 1–4. Provide scores (1–10) for ethics/psychological, economic, social, political, "
                    "and list 3–8 recommended safeguards tailored to the field."
                )
            for k in list(s.keys()):
                if k not in ("headline", "final_prompt"):
                    s.pop(k, None)
            continue

        s["headline"] = str(s.get("headline") or f"Step {i+1}").strip() or f"Step {i+1}"
        s["scenario"] = str(s.get("scenario") or "").strip()
        s["question"] = str(s.get("question") or "").strip()

        s["views"] = _normalize_views(s.get("views"))

        fixed_opts = _ensure_4_option_list(s.get("options"))
        if fixed_opts is None:
            fixed_opts = [{"id": k, "label": f"Option {k}", "summary": f"Choose option {k}."} for k in OPT_IDS]
        s["options"] = fixed_opts

        if not s["scenario"]:
            region = (payload.get("region") or "the region").strip()
            job = (payload.get("job_title") or "this job").strip()
            field = (payload.get("field") or "general").strip().lower()
            s["scenario"] = (
                f"In {region}, organizations introduce AI to reshape how {job} work in the {field} domain. "
                "A pilot across multiple sites reports a measurable improvement in turnaround time, but also shows reliability gaps at peak hours. "
                "A survey indicates a sizable share of users prefer the new workflow, while a smaller (but vocal) group reports frustration with transparency. "
                "Projected costs shift: some operational expenses drop, yet spending increases for integration, security, and training. "
                "Workers raise concerns about income volatility and how performance will be evaluated under AI-assisted workflows. "
                "Companies emphasize competitiveness and service consistency, arguing that delay could reduce market share. "
                "Regulators focus on accountability, liability, and incident reporting standards before broader rollout. "
                "Community groups ask whether benefits are evenly distributed across neighborhoods and vulnerable populations."
            )

        scount = _sentences_count(s["scenario"])
        if scount < 7:
            pads = [
                "Independent experts request clearer metrics, audits, and transparency reports.",
                "Local government asks for a compliance plan and a public incident-response protocol.",
                "Budget planning highlights trade-offs between speed of rollout and strength of safeguards.",
                "Stakeholders debate whether the system should be optional, phased, or mandatory by policy.",
            ]
            need = 7 - scount
            if not s["scenario"].endswith((".", "!", "?")):
                s["scenario"] += "."
            s["scenario"] += " " + " ".join(pads[:need])
        elif scount > 9:
            parts = re.split(r"(?<=[.!?])\s+", s["scenario"].strip())
            s["scenario"] = " ".join(parts[:8]).strip()

        if not s["question"]:
            s["question"] = "Which policy action should be taken next, given the trade-offs and stakeholder concerns?"

        allowed = {"headline", "scenario", "views", "question", "options"}
        for k in list(s.keys()):
            if k not in allowed:
                s.pop(k, None)

    return out


# ============================================================
# Validators
# ============================================================
def _validate_5step_scenario_reason(out: Any) -> Tuple[bool, str]:
    if not isinstance(out, dict):
        return False, f"out is not dict: {type(out)}"

    title = out.get("title")
    if not isinstance(title, str) or not title.strip():
        return False, "missing/empty title"

    steps = out.get("steps")
    if not isinstance(steps, list):
        return False, f"steps is not list: {type(steps)}"
    if len(steps) != 5:
        return False, f"steps length != 5 (got {len(steps)})"

    for i in range(4):
        s = steps[i]
        if not isinstance(s, dict):
            return False, f"step {i+1} is not dict: {type(s)}"

        for k in ("headline", "scenario", "question"):
            v = s.get(k)
            if not isinstance(v, str) or not v.strip():
                return False, f"step {i+1} missing/empty '{k}'"

        views = s.get("views")
        if not isinstance(views, list) or not (3 <= len(views) <= 6):
            return False, f"step {i+1} views must be list len 3..6"
        seen_stk = set()
        for vi, v in enumerate(views):
            if not isinstance(v, dict):
                return False, f"step {i+1} views[{vi}] not dict"
            st = v.get("stakeholder")
            if st not in STK_ALLOWED:
                return False, f"step {i+1} views[{vi}] invalid stakeholder: {st}"
            if st in seen_stk:
                return False, f"step {i+1} duplicate stakeholder in views: {st}"
            seen_stk.add(st)
            bullets = v.get("bullets")
            if not isinstance(bullets, list) or not (2 <= len(bullets) <= 4):
                return False, f"step {i+1} views[{vi}] bullets must be list len 2..4"
            if not all(isinstance(b, str) and b.strip() for b in bullets):
                return False, f"step {i+1} views[{vi}] bullets must be non-empty strings"

        opts = s.get("options")
        if not isinstance(opts, list):
            return False, f"step {i+1} options is not list: {type(opts)}"
        if len(opts) != 4:
            return False, f"step {i+1} options length != 4 (got {len(opts)})"

        seen = set()
        for j, o in enumerate(opts):
            if not isinstance(o, dict):
                return False, f"step {i+1} option[{j}] not dict: {type(o)}"
            oid = o.get("id")
            if oid not in ("A", "B", "C", "D"):
                return False, f"step {i+1} option[{j}] id invalid: {oid}"
            if oid in seen:
                return False, f"step {i+1} duplicate option id: {oid}"
            seen.add(oid)

            lab = o.get("label")
            summ = o.get("summary")
            if not isinstance(lab, str) or not lab.strip():
                return False, f"step {i+1} option {oid} missing/empty label"
            if not isinstance(summ, str) or not summ.strip():
                return False, f"step {i+1} option {oid} missing/empty summary"

        if seen != {"A", "B", "C", "D"}:
            return False, f"step {i+1} option ids not exactly A/B/C/D: {sorted(seen)}"

    s5 = steps[4]
    if not isinstance(s5, dict):
        return False, f"step 5 is not dict: {type(s5)}"
    if not isinstance(s5.get("headline"), str) or not s5["headline"].strip():
        return False, "step 5 missing/empty headline"
    if not isinstance(s5.get("final_prompt"), str) or not s5["final_prompt"].strip():
        return False, "step 5 missing/empty final_prompt"

    return True, "ok"


def _validate_final(out: Any) -> bool:
    if not isinstance(out, dict):
        return False
    if not isinstance(out.get("verdict_title"), str) or not out["verdict_title"].strip():
        return False
    di = out.get("dimension_impacts")
    if not isinstance(di, dict):
        return False
    for dim in ("ethics_psych", "economic", "social", "political"):
        if dim not in di or not isinstance(di[dim], dict):
            return False
        score = di[dim].get("score_1to10")
        if not isinstance(score, int) or not (1 <= score <= 10):
            return False
    rs = out.get("recommended_safeguards")
    if not isinstance(rs, list) or not (3 <= len(rs) <= 8):
        return False
    if not all(isinstance(x, str) and x.strip() for x in rs):
        return False
    return True


# ============================================================
# Scenario generation
# ============================================================
def _get_openai_client() -> Optional["OpenAI"]:
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def _responses_create_json(
    client: "OpenAI",
    *,
    model: str,
    system_text: str,
    user_obj: dict,
    temperature: float = 0.7,
) -> Tuple[Optional[dict], Optional[str]]:
    user_text = json.dumps(user_obj, ensure_ascii=False)

    def _extract_text_from_chat(resp) -> str:
        try:
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            temperature=temperature,
        )
        text = _extract_text_from_chat(resp)
        _dbg(f"CHAT_PLAIN raw output_text (first 1200 chars):\n{text[:1200]}")

        try:
            return json.loads(text), None
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = text[start : end + 1]
                _dbg(f"CHAT_PLAIN salvage candidate (first 1200 chars):\n{candidate[:1200]}")
                return json.loads(candidate), None
            return None, "CHAT_PLAIN: output was not valid JSON"
    except Exception as e:
        return None, str(e)


def _stub_scenario(payload: dict) -> dict:
    job = (payload.get("job_title") or "a job").strip() or "a job"
    region = (payload.get("region") or "your region").strip() or "your region"
    level = (payload.get("replacement_level") or "assist").strip().lower()
    field = (payload.get("field") or "general").strip().lower()

    level_txt = "assist" if level == "assist" else ("partly replace" if level == "partial" else "replace")
    title = f"{field.title()} Scenario: AI {level_txt.capitalize()}ing {job}"

    def opts(a, b, c, d):
        return [
            {"id": "A", "label": a[0], "summary": a[1]},
            {"id": "B", "label": b[0], "summary": b[1]},
            {"id": "C", "label": c[0], "summary": c[1]},
            {"id": "D", "label": d[0], "summary": d[1]},
        ]

    base_views = [
        {"stakeholder": "Government", "bullets": ["Requests compliance and reporting rules.", "Wants a clear liability and incident-response plan."]},
        {"stakeholder": "Company", "bullets": ["Focuses on efficiency and competitiveness.", "Worried about integration cost and service downtime."]},
        {"stakeholder": "Workers", "bullets": ["Wants job redesign and income stability.", "Requests training and fair evaluation standards."]},
    ]

    steps = [
        {
            "headline": "A new program is announced",
            "scenario": (
                f"In {region}, officials announce a plan where AI will {level_txt} {job}. "
                "A pilot across several sites reports a 12% improvement in turnaround time and a 7% decrease in operational delays. "
                "However, incident logs still show occasional failures during peak hours, and experts warn these edge cases matter. "
                "A survey suggests 58% of customers support the change, while 29% are concerned about transparency and accountability. "
                "Companies highlight potential productivity gains, but unions raise concerns about pay variability and job redesign. "
                "Regulators emphasize liability rules and demand a public incident-response process. "
                "Community groups ask whether service quality improvements will be evenly distributed across neighborhoods. "
                "You must decide how to start without locking in avoidable harms."
            ),
            "views": base_views,
            "question": "What is the first policy move you take to begin this transition?",
            "options": opts(
                ("Phased pilot + audit", "Launch a phased pilot with independent audits and rollback triggers; slower benefits but stronger safety learning."),
                ("Fast rollout + incentives", "Scale quickly with incentives to adopt; faster gains but higher risk of uneven impacts and backlash."),
                ("Strict limits + human oversight", "Set strict scope limits and require humans-in-the-loop for critical decisions; safer but costlier."),
                ("Targeted deployment", "Deploy only where readiness thresholds are met and fund equity supports; slower scale but reduces inequality risk."),
            ),
        },
        {
            "headline": "Performance data creates pressure",
            "scenario": (
                "After the first phase, performance dashboards show a 15% improvement in throughput in high-demand areas, but only 4% elsewhere. "
                "Maintenance costs rise by an estimated 9% due to monitoring, model updates, and security hardening. "
                "Customers report shorter waits, yet complaint volume increases by 18% around opaque automated decisions. "
                "Workers say scheduling has become more unpredictable, and some report higher stress under new evaluation metrics. "
                "Companies want to expand to protect market share, arguing competitors are moving faster. "
                "Regulators ask for clearer data retention rules and periodic fairness reporting. "
                "Experts recommend stress-testing the system on worst-case conditions before expansion. "
                "You must choose how to handle the mismatch between benefits and complaints."
            ),
            "views": [
                {"stakeholder": "Customers", "bullets": ["Like faster service but want explanations when things go wrong.", "Concerned about privacy and opaque decisions."]},
                {"stakeholder": "Company", "bullets": ["Wants to scale to capture efficiency gains.", "Warns that delays could reduce competitiveness."]},
                {"stakeholder": "Government", "bullets": ["Requests privacy rules and transparency reporting.", "Concerned about public complaints and legitimacy."]},
            ],
            "question": "How do you respond before scaling further?",
            "options": opts(
                ("Transparency package", "Mandate public metrics, audits, and clear user explanations; improves trust but slows rollout."),
                ("Worker protections", "Add workload limits, training, and grievance channels; raises costs but stabilizes adoption."),
                ("Technical hardening first", "Delay scaling to stress-test and harden reliability; safer but politically harder."),
                ("Selective expansion", "Expand only in high-performing contexts while fixing weak areas; risks perceived inequality."),
            ),
        },
        {
            "headline": "Equity and access become central",
            "scenario": (
                "Local analysis suggests low-income areas receive fewer quality improvements, and service gaps persist. "
                "A projection estimates uneven deployment could raise dissatisfaction by 10–20% if unaddressed. "
                "Companies propose premium tiers to fund improvements, but advocates warn that may widen inequality. "
                "Workers argue that staffing cuts in some areas could reduce local employment resilience. "
                "Regulators consider minimum service standards and penalties for neglecting underserved regions. "
                "Experts propose readiness scoring and equity-weighted performance metrics. "
                "Customers want consistent quality regardless of district. "
                "You must decide how equity will be enforced in the policy."
            ),
            "views": [
                {"stakeholder": "Community", "bullets": ["Demands fairness across neighborhoods.", "Worried about a two-tier system."]},
                {"stakeholder": "Government", "bullets": ["Considers minimum service standards.", "Wants enforceable oversight and penalties."]},
                {"stakeholder": "Company", "bullets": ["Seeks flexible pricing to fund upgrades.", "Warns strict rules may reduce innovation speed."]},
            ],
            "question": "What equity rule do you adopt?",
            "options": opts(
                ("Minimum service standards", "Set enforceable minimum standards and penalties; increases compliance burden but improves fairness."),
                ("Equity subsidies", "Use subsidies and targeted investment to lift underserved areas; costs public money but reduces gaps."),
                ("Readiness scoring", "Deploy by readiness thresholds plus equity-weighted metrics; slower but more defensible."),
                ("Market-led approach", "Allow pricing tiers with consumer protections; faster funding but higher inequality risk."),
            ),
        },
        {
            "headline": "Governance and legitimacy test",
            "scenario": (
                "A high-profile incident triggers media scrutiny and a spike in public concern. "
                "A follow-up poll indicates trust drops by 12 points unless stronger oversight is implemented. "
                "Companies worry that heavy regulation could reduce innovation and raise costs by an estimated 5–8%. "
                "Workers demand a clear accountability chain and an appeals process for AI-driven evaluations. "
                "Regulators consider whether enforcement should be centralized or shared with independent boards. "
                "Experts recommend transparent incident postmortems and publishable audit results. "
                "Customers want quick fixes and clearer explanations, not bureaucratic delays. "
                "You must choose the governance structure that will be seen as legitimate."
            ),
            "views": [
                {"stakeholder": "Experts", "bullets": ["Recommend publishable audits and incident postmortems.", "Warn against black-box deployment."]},
                {"stakeholder": "Workers", "bullets": ["Want appeals and accountability for evaluation metrics.", "Concerned about surveillance and stress."]},
                {"stakeholder": "Government", "bullets": ["Needs enforceable governance model.", "Wants to prevent polarization and restore trust."]},
            ],
            "question": "Which governance model do you implement?",
            "options": opts(
                ("Independent oversight board", "Create an independent board with audit powers; boosts legitimacy but can slow decisions."),
                ("Public consultation model", "Use structured public deliberation to set rules; improves buy-in but takes time."),
                ("Centralized authority", "Centralize decisions for speed with internal review; faster but risks distrust."),
                ("Shared governance", "Split authority across agencies and stakeholders; balances power but can become complex."),
            ),
        },
        {
            "headline": "Final Analysis",
            "final_prompt": (
                "Generate the final 4-dimension impact analysis based on the selected options (A/B/C/D) "
                "for steps 1–4. Provide scores (1–10) for ethics/psychological, economic, social, political, "
                "and list 3–8 recommended safeguards tailored to the field."
            ),
        },
    ]

    return {"title": title, "steps": steps}


def _make_placeholder_step(step_num: int, payload: dict) -> dict:
    """A valid step object that won't be shown until the user reaches it."""
    region = (payload.get("region") or "the region").strip()
    job = (payload.get("job_title") or "this job").strip()
    field = (payload.get("field") or "general").strip().lower()

    scenario = (
        f"This step will be generated after the previous choice, so the story stays connected. "
        f"In {region}, stakeholders continue debating how AI reshapes {job} in the {field} domain. "
        "Early monitoring reports a 10%–15% performance spread across sites, and complaint volume shifts by about 5% week-to-week. "
        "Budget planning remains uncertain, with integration costs projected to rise by roughly 6% this quarter. "
        "Regulators request clearer documentation and incident-response readiness. "
        "Workers and community groups ask for predictable rules and transparent appeals. "
        "Companies push for a timeline that maintains competitiveness without triggering preventable harms. "
        "You will decide the next policy action once this step is generated."
    )

    return {
        "headline": f"Upcoming decision (Step {step_num})",
        "scenario": scenario,
        "views": _normalize_views(None),
        "question": "This decision will appear once the step is generated.",
        "options": [{"id": k, "label": f"Option {k}", "summary": f"Placeholder option {k}."} for k in OPT_IDS],
    }


def _is_placeholder_step(step_obj: dict) -> bool:
    if not isinstance(step_obj, dict):
        return True
    sc = str(step_obj.get("scenario") or "")
    return sc.strip().startswith("This step will be generated after the previous choice")


def _normalize_single_step(step_obj: Any, payload: dict, step_index_1based: int) -> dict:
    """Normalize one step (1..4) to match the front-end expectations."""
    if not isinstance(step_obj, dict):
        step_obj = {}
    s = {
        "headline": str(step_obj.get("headline") or f"Step {step_index_1based}").strip() or f"Step {step_index_1based}",
        "scenario": str(step_obj.get("scenario") or "").strip(),
        "question": str(step_obj.get("question") or "").strip(),
        "views": _normalize_views(step_obj.get("views")),
        "options": _ensure_4_option_list(step_obj.get("options")) or [
            {"id": k, "label": f"Option {k}", "summary": f"Choose option {k}."} for k in OPT_IDS
        ],
    }

    # Ensure scenario length 7–8 sentences
    if not s["scenario"]:
        region = (payload.get("region") or "the region").strip()
        job = (payload.get("job_title") or "this job").strip()
        field = (payload.get("field") or "general").strip().lower()
        s["scenario"] = (
            f"In {region}, the AI transition continues to reshape how {job} work in the {field} domain. "
            "Monitoring shows a 12% improvement in one segment but only 3% elsewhere, raising questions about uneven benefits. "
            "A short survey finds about 55% approval and 25% concern, mainly about transparency and accountability. "
            "Costs shift again: some operating expenses drop, but security and auditing costs rise by roughly 7%. "
            "Workers report stress increases when metrics change quickly, and request clearer evaluation standards. "
            "Companies argue for faster expansion to protect competitiveness and avoid losing market share. "
            "Regulators ask for clearer documentation, retention rules, and an incident-response protocol. "
            "Community groups push for equity safeguards so underserved areas do not fall behind."
        )

    scount = _sentences_count(s["scenario"])
    if scount < 7:
        pads = [
            "Independent experts ask for stress-tests and publishable audit summaries.",
            "Local officials request an enforcement plan with measurable milestones.",
            "Budget planners debate the trade-off between speed and strength of safeguards.",
            "Stakeholders argue over whether the system should be optional, phased, or mandatory.",
        ]
        need = 7 - scount
        if not s["scenario"].endswith((".", "!", "?")):
            s["scenario"] += "."
        s["scenario"] += " " + " ".join(pads[:need])
    elif scount > 9:
        parts = re.split(r"(?<=[.!?])\s+", s["scenario"].strip())
        s["scenario"] = " ".join(parts[:8]).strip()

    if not s["question"]:
        s["question"] = "Which policy action should be taken next, given the trade-offs and stakeholder concerns?"

    return s


def generate_step(payload: dict, scenario_title: str, history: List[dict], step_num: int) -> dict:
    """Generate ONE connected step (1..4) using prior choices for continuity."""
    client = _get_openai_client()
    if client is None:
        # fallback: use stub's corresponding step
        stub = _stub_scenario(payload)
        step_obj = stub["steps"][step_num - 1]
        return _normalize_single_step(step_obj, payload, step_num)

    model = os.getenv("HORIZON_SCENARIO_MODEL", "gpt-4.1-mini")

    system = (
        "You generate one step of a 5-step classroom-safe policy simulation for students.\n"
        "Return ONLY valid JSON. No markdown. No extra keys.\n"
        "Output must be a single JSON object with keys: headline, scenario, views, question, options.\n"
        "\n"
        "REQUIREMENTS:\n"
        "- scenario must be EXACTLY 7–8 sentences and include at least 2 quantitative data points.\n"
        f"- views: array of 3–6 stakeholders; stakeholder must be one of: {', '.join(STK_ALLOWED)}.\n"
        "- each views item: {stakeholder, bullets} with 2–4 bullets.\n"
        "- options: EXACTLY 4 objects with ids A/B/C/D, each {id,label,summary}.\n"
        "- Make the step clearly connected to the previous step and the user's previous choice(s).\n"
        "Tone: neutral, educational, multi-perspective, non-persuasive.\n"
        "Safety: no hateful/harassing/sexual/violent/extremist content.\n"
    )

    user = {
        "task": f"Generate step {step_num} (of steps 1–4) for the policy simulation.",
        "scenario_title": scenario_title,
        "topic": {
            "job_title": (payload.get("job_title") or "").strip(),
            "replacement_level": (payload.get("replacement_level") or "assist").strip().lower(),
            "region": (payload.get("region") or "a specified region").strip(),
            "field": (payload.get("field") or "general").strip().lower(),
        },
        "continuity": {
            "history": history,
            "instruction": (
                "Use the history to carry forward consequences, pressures, and stakeholder reactions. "
                "Do not contradict prior information. Introduce a new decision point that follows from the last choice."
            ),
        },
        "format_reminder": {
            "keys": ["headline", "scenario", "views", "question", "options"],
            "options_ids": ["A", "B", "C", "D"],
        },
    }

    out, err = _responses_create_json(
        client,
        model=model,
        system_text=system,
        user_obj=user,
        temperature=0.7,
    )

    if out is None:
        _dbg(f"Step gen failed -> stub. step={step_num} err={err}")
        stub = _stub_scenario(payload)
        return _normalize_single_step(stub["steps"][step_num - 1], payload, step_num)

    # Normalize this single step defensively
    step_norm = _normalize_single_step(out, payload, step_num)

    # Lightweight validation by embedding into a 5-step shell
    ok, reason = _validate_5step_scenario_reason({"title": scenario_title or "Scenario", "steps": [
        step_norm,
        _make_placeholder_step(2, payload),
        _make_placeholder_step(3, payload),
        _make_placeholder_step(4, payload),
        {"headline": "Final Analysis", "final_prompt": "x"},
    ]})
    if not ok:
        _dbg(f"Step normalization/validation failed -> stub. step={step_num} reason={reason}")
        stub = _stub_scenario(payload)
        return _normalize_single_step(stub["steps"][step_num - 1], payload, step_num)

    return step_norm


def _build_history_for_stepgen(scenario: dict, choices: List[dict]) -> List[dict]:
    steps = scenario.get("steps") or []
    hist = []
    for c in choices:
        try:
            step_i = int(c.get("step") or 0)
        except Exception:
            continue
        if not (1 <= step_i <= 4):
            continue
        if step_i - 1 >= len(steps):
            continue
        s = steps[step_i - 1] if isinstance(steps[step_i - 1], dict) else {}
        opt_id = str(c.get("option") or "").strip().upper()
        opt_obj = None
        for o in (s.get("options") or []):
            if isinstance(o, dict) and str(o.get("id") or "").strip().upper() == opt_id:
                opt_obj = o
                break

        scen_text = str(s.get("scenario") or "").strip()
        excerpt = scen_text
        if scen_text:
            parts = re.split(r"(?<=[.!?])\s+", scen_text)
            excerpt = " ".join([p for p in parts[:2] if p.strip()]).strip() or scen_text[:240]

        hist.append({
            "step": step_i,
            "headline": s.get("headline") or f"Step {step_i}",
            "scenario_excerpt": excerpt,
            "chosen": {
                "id": opt_id,
                "label": (opt_obj or {}).get("label") if opt_obj else None,
                "summary": (opt_obj or {}).get("summary") if opt_obj else None,
            }
        })
    return hist



def _ensure_step_generated_for_run(run_id: int, step_to_generate: int) -> None:
    """
    Lazy-generate and persist step 2/3/4 after the user makes a choice.

    IMPORTANT:
    - This function should NOT depend on any global "current_step" variable.
    - It should only generate `step_to_generate` if that step is still a placeholder.
    """
    if step_to_generate not in (2, 3, 4):
        return

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT r.id AS run_id, r.choices_json, s.id AS scenario_id, s.input_json, s.scenario_json
            FROM runs r
            JOIN scenarios s ON s.id = r.scenario_id
            WHERE r.id = ?
            """,
            (run_id,),
        )
        row = cur.fetchone()
        if not row:
            return

        payload = json.loads(row["input_json"] or "{}")
        scenario = json.loads(row["scenario_json"] or "{}")
        choices = json.loads(row["choices_json"] or "[]")

        steps = scenario.get("steps") if isinstance(scenario.get("steps"), list) else []
        if len(steps) < 5:
            return

        idx = step_to_generate - 1
        existing = steps[idx] if idx < len(steps) else None
        if isinstance(existing, dict) and not _is_placeholder_step(existing):
            return  # already generated

        title = scenario.get("title") or "Scenario"
        history = _build_history_for_stepgen(scenario, choices)

        try:
            new_step = generate_step(payload, title, history, step_to_generate)
        except Exception as e:
            _dbg(f"Step gen exception -> stub. step={step_to_generate} err={e}")
            stub = _stub_scenario(payload)
            new_step = stub["steps"][idx]

        steps[idx] = new_step
        scenario["steps"] = steps

        cur.execute(
            "UPDATE scenarios SET scenario_json = ? WHERE id = ?",
            (json.dumps(scenario, ensure_ascii=False), row["scenario_id"]),
        )
        conn.commit()


def generate_scenario(payload: dict) -> dict:
    """Step-wise scenario: generate Step 1 now, Steps 2–4 later after choices."""
    # Step 1 now
    step1 = generate_step(payload, scenario_title="", history=[], step_num=1)

    # Title (same logic as normalize_scenario_output)
    job = (payload.get("job_title") or "a job").strip() or "a job"
    field = (payload.get("field") or "general").strip().lower()
    level = (payload.get("replacement_level") or "assist").strip().lower()
    level_txt = "Assisting" if level == "assist" else ("Partially Replacing" if level == "partial" else "Replacing")
    title = f"{field.title()} Scenario: AI {level_txt} {job}"

    steps = [
        step1,
        _make_placeholder_step(2, payload),
        _make_placeholder_step(3, payload),
        _make_placeholder_step(4, payload),
        {"headline": "Final Analysis", "final_prompt": (
            "Generate the final 4-dimension impact analysis based on the selected options (A/B/C/D) "
            "for steps 1–4. Provide scores (1–10) for ethics/psychological, economic, social, political, "
            "and list 3–8 recommended safeguards tailored to the field."
        )},
    ]

    out = {"title": title, "steps": steps}
    # Keep strict keys/shape and sentence length (placeholders will be padded to match)
    out = normalize_scenario_output(out, payload)
    return out

# ============================================================
# Final analysis generation
# ============================================================
def _stub_final(choices: List[dict]) -> dict:
    picked = "".join([c.get("option", "") for c in choices])[:4]
    base = 6
    ethics = max(1, min(10, base + (1 if "C" in picked else 0) - (1 if "B" in picked else 0)))
    econ = max(1, min(10, base + (1 if "B" in picked else 0) - (1 if "C" in picked else 0)))
    social = max(1, min(10, base + (1 if "A" in picked else 0) - (1 if "B" in picked else 0)))
    pol = max(1, min(10, base + (1 if "A" in picked else 0) - (1 if "C" in picked else 0)))

    verdict = "Balanced transition with safeguards" if ethics >= 6 and econ >= 6 else "High-risk transition; strengthen protections"

    return {
        "verdict_title": verdict,
        "dimension_impacts": {
            "ethics_psych": {"score_1to10": ethics},
            "economic": {"score_1to10": econ},
            "social": {"score_1to10": social},
            "political": {"score_1to10": pol},
        },
        "recommended_safeguards": [
            "Phased rollout with public reporting",
            "Worker transition support (re-skilling + income bridge)",
            "Independent audits for bias and safety",
            "Clear complaint and appeal process for affected people",
        ],
    }


def generate_final_analysis(scenario: dict, choices: List[dict]) -> dict:
    client = _get_openai_client()
    if client is None:
        return _stub_final(choices)

    model = os.getenv("HORIZON_FINAL_MODEL", "gpt-4.1-mini")

    system = (
        "You are an analyst that summarizes policy simulation outcomes.\n"
        "Return ONLY valid JSON that matches the provided schema. No markdown. No extra keys.\n"
        "Scores are 1-10. Use neutral, educational framing. Do not give political persuasion.\n"
        "Tailor safeguards to the domain implied by the scenario."
    )

    user = {
        "task": "Given the scenario title + step headlines and the user's 4 choices (A/B/C/D), generate a final impact analysis.",
        "scenario": {
            "title": scenario.get("title", ""),
            "step_headlines": [s.get("headline", "") for s in (scenario.get("steps") or [])[:4]],
        },
        "choices": choices,
        "instructions": {
            "dimension_meaning": {
                "ethics_psych": "fairness, dignity, psychological safety, trust",
                "economic": "productivity, employment transition, costs, growth",
                "social": "inequality, access, cohesion, community stability",
                "political": "legitimacy, governance capacity, polarization risk, regulatory clarity",
            },
            "safeguards": "Provide 3-8 concise safeguards tailored to the choices and domain",
        },
    }

    out, _err = _responses_create_json(
        client,
        model=model,
        system_text=system,
        user_obj=user,
        temperature=0.6,
    )

    if out is None or not _validate_final(out):
        return _stub_final(choices)

    return out


# ============================================================
# Step image generation (on-demand)
# ============================================================
def _svg_data_uri(title: str, subtitle: str) -> str:
    title = (title or "").strip()[:80]
    subtitle = (subtitle or "").strip()[:140]

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="768">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#0b0f17"/>
          <stop offset="1" stop-color="#111a2a"/>
        </linearGradient>
      </defs>
      <rect width="100%" height="100%" fill="url(#g)"/>
      <rect x="56" y="56" width="912" height="656" rx="28"
            fill="rgba(255,255,255,0.05)" stroke="rgba(13,202,240,0.35)"/>
      <text x="96" y="180" fill="rgba(13,202,240,0.95)" font-family="monospace" font-size="34" letter-spacing="6">
        PROJECT HORIZON
      </text>
      <text x="96" y="265" fill="rgba(255,255,255,0.92)" font-family="system-ui, -apple-system, Segoe UI, Roboto" font-size="54" font-weight="700">
        {title}
      </text>
      <text x="96" y="340" fill="rgba(255,255,255,0.70)" font-family="system-ui, -apple-system, Segoe UI, Roboto" font-size="28">
        {subtitle}
      </text>
      <text x="96" y="670" fill="rgba(255,255,255,0.38)" font-family="monospace" font-size="18" letter-spacing="2">
        (Placeholder image — enable OpenAI to generate real art)
      </text>
    </svg>"""
    return "data:image/svg+xml;charset=utf-8," + quote(svg)


def _openai_step_image(client: "OpenAI", prompt: str, out_path: str) -> None:
    resp = client.images.generate(
        model=os.getenv("HORIZON_IMAGE_MODEL", "gpt-image-1"),
        prompt=prompt,
        size=os.getenv("HORIZON_IMAGE_SIZE", "1024x1024"),
    )

    b64 = None
    try:
        b64 = resp.data[0].b64_json
    except Exception:
        b64 = None

    if not b64:
        raise RuntimeError("Image generation returned no b64 data.")

    img_bytes = base64.b64decode(b64)
    with open(out_path, "wb") as f:
        f.write(img_bytes)

def _is_data_uri(url: str) -> bool:
    return isinstance(url, str) and url.startswith("data:image/")


def ensure_scenario_cover(scenario_id: int, prefer_openai: bool = False) -> str:
    """
    Ensures scenarios.cover_image_url is populated.

    - prefer_openai=False: store a fast SVG placeholder (or keep existing)
    - prefer_openai=True: if OpenAI is configured, generate a real cover image and replace placeholders
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, cover_image_url, scenario_json FROM scenarios WHERE id = ?", (scenario_id,))
        row = cur.fetchone()

    if not row:
        raise ValueError("Scenario not found.")

    existing = row["cover_image_url"] or ""
    try:
        scenario = json.loads(row["scenario_json"] or "{}")
    except Exception:
        scenario = {}

    title = (scenario.get("title") or f"Scenario #{scenario_id}").strip()
    steps = scenario.get("steps") or []
    step1 = steps[0] if isinstance(steps, list) and steps else {}
    headline = (step1.get("headline") or "Step 1").strip()
    context = (step1.get("scenario") or "")[:180].strip()

    # If we already have a real URL and we aren't forcing OpenAI, keep it
    if existing and (not prefer_openai):
        return existing

    # If prefer_openai and we already have a real URL (not a data-uri placeholder), keep it
    if prefer_openai and existing and (not _is_data_uri(existing)):
        return existing

    client = _get_openai_client()

    # If OpenAI is not available or we don't want a real image: store SVG placeholder (fast)
    if (client is None):
        placeholder = _svg_data_uri(title, headline)
        with get_conn() as conn:
            cur = conn.cursor()
            # only fill if empty, so we don't overwrite a real cover
            cur.execute(
                "UPDATE scenarios SET cover_image_url = COALESCE(NULLIF(cover_image_url,''), ?) WHERE id = ?",
                (placeholder, scenario_id),
            )
            conn.commit()
        return placeholder if not existing else existing

    # Generate a real cover image (no text)
    prompt = (
        "Create a classroom-safe, modern, cinematic cover illustration for a policy simulation library.\n"
        "No text in the image. No logos. No brand marks.\n"
        f"Scenario title: {title}\n"
        f"Opening headline: {headline}\n"
        f"Opening context: {context}\n"
        "Style: clean, modern, slightly futuristic, soft lighting, high detail, wide composition."
    )
    fname = f"scenario_{scenario_id}_cover.png"
    out_path = os.path.join(GENERATED_DIR, fname)

    if not os.path.exists(out_path):
        _openai_step_image(client, prompt, out_path)

    url = url_for("static", filename=f"generated/{fname}")
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE scenarios SET cover_image_url = ? WHERE id = ?", (url, scenario_id))
        conn.commit()
    return url



def generate_step_image_for_run(run_id: int, step: int) -> str:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT r.images_enabled, r.images_json, s.scenario_json
            FROM runs r
            JOIN scenarios s ON s.id = r.scenario_id
            WHERE r.id = ?
        """, (run_id,))
        row = cur.fetchone()

    if not row:
        raise ValueError("Run not found.")

    if int(row["images_enabled"] or 0) != 1:
        raise PermissionError("Images are disabled for this run.")

    images = json.loads(row["images_json"] or "{}")
    key = str(step)
    if key in images and images[key]:
        return images[key]

    scenario = json.loads(row["scenario_json"])
    title = scenario.get("title", "Scenario")
    steps = scenario.get("steps") or []
    step_obj = steps[step - 1] if 1 <= step <= len(steps) else {}
    headline = step_obj.get("headline", f"Step {step}")
    scenario_text = step_obj.get("scenario") or step_obj.get("final_prompt") or ""

    client = _get_openai_client()
    if client is None:
        url = _svg_data_uri(f"{headline}", (scenario_text[:120] + ("…" if len(scenario_text) > 120 else "")))
    else:
        prompt = (
            "Create a classroom-safe, modern, cinematic illustration for a policy simulation app.\n"
            "No text in the image. No logos. No brand marks.\n"
            f"Scene title: {title}\n"
            f"Step {step} headline: {headline}\n"
            f"Context: {scenario_text}\n"
            "Style: clean, modern, slightly futuristic, soft lighting, high detail, wide composition."
        )

        fname = f"run_{run_id}_step_{step}.png"
        out_path = os.path.join(GENERATED_DIR, fname)
        _openai_step_image(client, prompt, out_path)
        url = url_for("static", filename=f"generated/{fname}")

    images[key] = url
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE runs SET images_json = ? WHERE id = ?",
                    (json.dumps(images, ensure_ascii=False), run_id))
        conn.commit()

    return url


# ============================================================
# Horizon MBTI (Decision Style) — scoring + storage
# ============================================================
PRIMARY_VALUES = ["safety", "fairness", "cost", "speed", "privacy", "legitimacy", "jobs", "innovation"]

def _clamp_int(v: Any, lo: int, hi: int, default: int) -> int:
    try:
        x = int(v)
    except Exception:
        return default
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

def _clean_short_text(s: Any, max_len: int = 240) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s[:max_len]

def compute_axes_from_signal(sig: dict) -> Dict[str, float]:
    """
    Axes:
      SU: + => Safety-first (S), - => Utility-first (U)
      FG: + => Fairness-first (F), - => Growth/efficiency-first (G)
      HA: + => Human-led (H), - => Automation-led (A)
      PC: + => Participatory governance (P), - => Centralized governance (C)

    Signal inputs are 1..5 for sliders.
    Confidence weights the magnitude slightly (0.7..1.1).
    """
    conf = _clamp_int(sig.get("confidence"), 1, 5, 3)
    pv = (sig.get("primary_value") or "").strip().lower()

    risk = _clamp_int(sig.get("risk_tolerance"), 1, 5, 3)
    over = _clamp_int(sig.get("oversight_preference"), 1, 5, 3)
    gov = _clamp_int(sig.get("governance_preference"), 1, 5, 3)

    # base contributions
    su = float(over - risk)               # -4..+4
    ha = float(over - 3)                  # -2..+2
    pc = float(gov - 3)                   # -2..+2

    # fairness vs growth: use primary_value
    if pv in ("fairness", "jobs", "privacy", "legitimacy", "safety"):
        fg = 2.0
    elif pv in ("cost", "speed", "innovation"):
        fg = -2.0
    else:
        fg = 0.0

    w = 0.6 + 0.1 * conf                  # 0.7..1.1
    return {"SU": su * w, "FG": fg * w, "HA": ha * w, "PC": pc * w}

def axes_to_type(scores: Dict[str, float]) -> str:
    su = scores.get("SU", 0.0)
    fg = scores.get("FG", 0.0)
    ha = scores.get("HA", 0.0)
    pc = scores.get("PC", 0.0)
    return "".join([
        "S" if su >= 0 else "U",
        "F" if fg >= 0 else "G",
        "H" if ha >= 0 else "A",
        "P" if pc >= 0 else "C",
    ])

def recompute_run_mbti(conn: sqlite3.Connection, run_id: int) -> Tuple[str, Dict[str, float]]:
    cur = conn.cursor()
    cur.execute("""
        SELECT axis_json FROM run_step_signals
        WHERE run_id = ?
        ORDER BY step ASC
    """, (run_id,))
    rows = cur.fetchall()

    totals = {"SU": 0.0, "FG": 0.0, "HA": 0.0, "PC": 0.0}
    for r in rows:
        try:
            ax = json.loads(r["axis_json"] or "{}")
        except Exception:
            ax = {}
        for k in totals.keys():
            try:
                totals[k] += float(ax.get(k, 0.0))
            except Exception:
                pass

    mbti = axes_to_type(totals)
    return mbti, totals


def recompute_and_store_mbti(conn: sqlite3.Connection, run_id: int) -> Dict[str, Any]:
    """
    Back-compat helper: recompute the run-level MBTI snapshot and store it in `runs`.

    Note: `upsert_step_signal()` already recomputes + stores MBTI, but some callers
    still invoke this function. Keeping it prevents NameError in logs.
    """
    mbti, totals = recompute_run_mbti(conn, run_id)
    cur = conn.cursor()
    cur.execute(
        "UPDATE runs SET mbti_type = ?, mbti_scores_json = ? WHERE id = ?",
        (mbti, json.dumps(totals, ensure_ascii=False), run_id),
    )
    conn.commit()
    return {"mbti_type": mbti, "mbti_scores": totals}


def upsert_step_signal(conn: sqlite3.Connection, *, user_id: Optional[int], run_id: int, scenario_id: int,
                       step: int, option_id: str, signal: dict) -> Dict[str, Any]:
    """
    Stores step signal (upsert), computes axis_json for that step,
    then recomputes run-level mbti_type + mbti_scores_json.
    Returns a payload containing updated mbti info.
    """
    confidence = _clamp_int(signal.get("confidence"), 1, 5, 3)
    primary_value = (signal.get("primary_value") or "").strip().lower()
    if primary_value not in PRIMARY_VALUES:
        primary_value = "legitimacy"

    risk_tolerance = _clamp_int(signal.get("risk_tolerance"), 1, 5, 3)
    oversight_preference = _clamp_int(signal.get("oversight_preference"), 1, 5, 3)
    governance_preference = _clamp_int(signal.get("governance_preference"), 1, 5, 3)
    rationale = _clean_short_text(signal.get("rationale") or "", 240)

    decision_ms = signal.get("decision_ms")
    try:
        decision_ms = int(decision_ms) if decision_ms is not None else None
        if decision_ms is not None and decision_ms < 0:
            decision_ms = None
    except Exception:
        decision_ms = None

    # compute axis contributions for this step
    ax = compute_axes_from_signal({
        "confidence": confidence,
        "primary_value": primary_value,
        "risk_tolerance": risk_tolerance,
        "oversight_preference": oversight_preference,
        "governance_preference": governance_preference,
    })
    axis_json = json.dumps(ax, ensure_ascii=False)

    cur = conn.cursor()
    cur.execute("""
        INSERT INTO run_step_signals
        (created_at, user_id, run_id, scenario_id, step, option_id,
         confidence, primary_value, risk_tolerance, oversight_preference, governance_preference,
         rationale, decision_ms, axis_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, step) DO UPDATE SET
          option_id=excluded.option_id,
          confidence=excluded.confidence,
          primary_value=excluded.primary_value,
          risk_tolerance=excluded.risk_tolerance,
          oversight_preference=excluded.oversight_preference,
          governance_preference=excluded.governance_preference,
          rationale=excluded.rationale,
          decision_ms=excluded.decision_ms,
          axis_json=excluded.axis_json
    """, (
        now_iso(), user_id, run_id, scenario_id, step, option_id,
        confidence, primary_value, risk_tolerance, oversight_preference, governance_preference,
        rationale, decision_ms, axis_json
    ))
    conn.commit()

    mbti, totals = recompute_run_mbti(conn, run_id)
    cur.execute("""
        UPDATE runs
        SET mbti_type = ?, mbti_scores_json = ?
        WHERE id = ?
    """, (mbti, json.dumps(totals, ensure_ascii=False), run_id))
    conn.commit()

    return {"mbti_type": mbti, "mbti_scores": totals}


# ============================================================
# Pages
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scenario", methods=["GET"])
def scenario_page():
    return render_template("scenario_input.html")


@app.route("/sim", methods=["GET"])
def sim_page():
    return render_template("sim.html")

@app.route("/library", methods=["GET"])
def library_page():
    return render_template("library.html")


@app.route("/scenario/<int:scenario_id>/start", methods=["GET"])
def start_from_library(scenario_id: int):
    """Create a new run from an existing scenario, then redirect to /sim."""
    enable_images = 1 if str(request.args.get("images", "0")).strip().lower() in ("1", "true", "yes", "on") else 0
    user_id = session.get("user_id")

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM scenarios WHERE id = ? AND COALESCE(is_public,1)=1", (scenario_id,))
        s = cur.fetchone()
        if not s:
            return redirect("/library")

        cur.execute("""
            INSERT INTO runs
              (created_at, user_id, scenario_id, current_step, choices_json, final_json, images_enabled, images_json, mbti_type, mbti_scores_json)
            VALUES (?, ?, ?, 1, '[]', NULL, ?, ?, NULL, '{}')
        """, (now_iso(), user_id, scenario_id, enable_images, json.dumps({}, ensure_ascii=False)))
        run_id = cur.lastrowid
        conn.commit()

    session["run_id"] = run_id
    return redirect("/sim")


@app.route("/profile", methods=["GET"])
def profile_page():
    return render_template("profile.html")


# ============================================================
# Auth
# ============================================================
@app.route("/login", methods=["GET"])
def login_page():
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login_submit():
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"status": "error", "message": "Email and password are required."}), 400

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name, email, role, password_hash FROM users WHERE email = ?", (email,))
        row = cur.fetchone()

    if not row or not row["password_hash"] or not check_password_hash(row["password_hash"], password):
        return jsonify({"status": "error", "message": "Invalid email or password."}), 401

    session["user_id"] = row["id"]
    session["user_name"] = row["name"]
    session["username"] = row["name"]  # back-compat for templates
    session["user_email"] = row["email"]
    session["role"] = row["role"]
    session["is_moderator"] = 1 if (row["email"] or "").strip().lower() in MODERATOR_EMAILS else 0
    return jsonify({"status": "success", "redirect": "/scenario"})


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"status": "success", "redirect": "/"})


@app.route("/logout", methods=["GET"])
def logout_get():
    session.clear()
    # keep it simple for navbar link
    return "", 204


@app.route("/register", methods=["GET"])
def register_page():
    return render_template("register.html")


@app.route("/register", methods=["POST"])
def register_submit():
    data = request.get_json(force=True)

    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    role = (data.get("role") or "student").strip()
    region = (data.get("region") or "").strip()
    interests = data.get("interests") or []
    values = data.get("values") or {}
    risk_level = int(data.get("risk_level") or 3)
    complexity = (data.get("complexity") or "balanced").strip()

    if not name or not email or not password:
        return jsonify({"status": "error", "message": "Name, email, and password are required."}), 400
    if len(password) < 8:
        return jsonify({"status": "error", "message": "Password must be at least 8 characters."}), 400

    interests_str = ",".join([str(x).strip() for x in interests if str(x).strip()])
    values_str = json.dumps(values, ensure_ascii=False)
    password_hash = generate_password_hash(password)

    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO users
                (created_at, name, email, password_hash, role, region, interests, values_json, risk_level, complexity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now_iso(), name, email, password_hash,
                role, region, interests_str, values_str, risk_level, complexity
            ))
            conn.commit()
    except sqlite3.IntegrityError:
        return jsonify({"status": "error", "message": "This email is already registered."}), 409

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name, email, role FROM users WHERE email = ?", (email,))
        u = cur.fetchone()
    if u:
        session["user_id"] = u["id"]
        session["user_name"] = u["name"]
        session["username"] = u["name"]  # back-compat for templates
        session["user_email"] = u["email"]
        session["role"] = u["role"]
        session["is_moderator"] = 1 if (u["email"] or "").strip().lower() in MODERATOR_EMAILS else 0

    return jsonify({"status": "success", "redirect": "/"})


# ============================================================
# Scenario APIs
# ============================================================
@app.route("/api/generate_scenario", methods=["POST"])
def api_generate_scenario():
    payload = request.get_json(force=True)

    job_title = (payload.get("job_title") or "").strip()
    if not job_title:
        return jsonify({"status": "error", "message": "job_title is required."}), 400

    enable_images = bool(payload.get("enable_images", False))

    scenario = generate_scenario(payload)

    user_id = session.get("user_id")  # optional
    input_json = json.dumps(payload, ensure_ascii=False)
    scenario_json = json.dumps(scenario, ensure_ascii=False)

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO scenarios (created_at, user_id, input_json, scenario_json)
            VALUES (?, ?, ?, ?)
        """, (now_iso(), user_id, input_json, scenario_json))
        scenario_id = cur.lastrowid

        cur.execute("""
            INSERT INTO runs (created_at, user_id, scenario_id, current_step, choices_json, final_json, images_enabled, images_json, mbti_type, mbti_scores_json)
            VALUES (?, ?, ?, 1, '[]', NULL, ?, ?, NULL, '{}')
        """, (now_iso(), user_id, scenario_id, 1 if enable_images else 0, json.dumps({}, ensure_ascii=False)))
        run_id = cur.lastrowid

        conn.commit()

    session["run_id"] = run_id
    return jsonify({"status": "success", "redirect": "/sim"})


@app.route("/api/state", methods=["GET"])
def api_state():
    run_id = session.get("run_id")
    if not run_id:
        return jsonify({"status": "error", "message": "No active run."}), 404

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT r.id AS run_id,
                   r.user_id,
                   r.current_step,
                   r.choices_json,
                   r.final_json,
                   r.images_enabled,
                   r.images_json,
                   r.mbti_type,
                   r.mbti_scores_json,
                   s.scenario_json,
                   s.id AS scenario_id
            FROM runs r
            JOIN scenarios s ON s.id = r.scenario_id
            WHERE r.id = ?
        """, (run_id,))
        row = cur.fetchone()

    if not row:
        return jsonify({"status": "error", "message": "Run not found."}), 404

    scenario = json.loads(row["scenario_json"])
    choices = json.loads(row["choices_json"] or "[]")
    final = json.loads(row["final_json"]) if row["final_json"] else None

    current_step = int(row["current_step"] or 1)
    if len(choices) >= 4 and current_step < 5:
        current_step = 5

    images_enabled = int(row["images_enabled"] or 0) == 1
    images = json.loads(row["images_json"] or "{}")
    if not isinstance(images, dict):
        images = {}

    mbti_scores = {}
    try:
        mbti_scores = json.loads(row["mbti_scores_json"] or "{}")
        if not isinstance(mbti_scores, dict):
            mbti_scores = {}
    except Exception:
        mbti_scores = {}

    progress = {
        "run_id": row["run_id"],
        "scenario_id": row["scenario_id"],
        "current_step": current_step,
        "choices": choices,
        "final": final,
        "images_enabled": images_enabled,
        "images": images,
        "mbti_type": row["mbti_type"],
        "mbti_scores": mbti_scores,
    }

    return jsonify({"status": "success", "scenario": scenario, "progress": progress})


@app.route("/api/choose", methods=["POST"])
def api_choose():
    """
    FULL decision capture endpoint.
    Frontend sends:
      { option: "A", signal: {confidence, primary_value, ...} }

    We store:
      - run_step_signals (upsert) for steps 1–4
      - runs.choices_json (append) and step advance
      - runs.mbti_type/scores (recompute)
    """
    run_id = session.get("run_id")
    if not run_id:
        return jsonify({"status": "error", "message": "No active run."}), 404

    data = request.get_json(force=True)
    option = (data.get("option") or "").strip().upper()
    if option not in ("A", "B", "C", "D"):
        return jsonify({"status": "error", "message": "option must be A, B, C, or D."}), 400

    signal = data.get("signal") or {}
    if not isinstance(signal, dict):
        signal = {}

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT r.current_step, r.choices_json, r.user_id, r.scenario_id
            FROM runs r
            WHERE r.id = ?
        """, (run_id,))
        row = cur.fetchone()

        if not row:
            return jsonify({"status": "error", "message": "Run not found."}), 404

        choices = json.loads(row["choices_json"] or "[]")
        if len(choices) >= 4:
            return jsonify({"status": "error", "message": "Choices already complete. Finalize to see results."}), 400

        step = len(choices) + 1

        # Store step signal only for steps 1..4
        if step <= 4:
            # validate required signal fields (tight enough to be meaningful)
            conf = signal.get("confidence")
            pv = (signal.get("primary_value") or "").strip().lower()
            if conf is None or pv not in PRIMARY_VALUES:
                return jsonify({"status": "error", "message": "Missing or invalid signal fields."}), 400

            upsert_step_signal(
                conn,
                user_id=row["user_id"],
                run_id=int(run_id),
                scenario_id=int(row["scenario_id"]),
                step=int(step),
                option_id=option,
                signal=signal,
            )

        # Append choice
        choices.append({"step": step, "option": option, "ts": now_iso()})

        new_step = 5 if len(choices) >= 4 else (step + 1)
        cur.execute("""
            UPDATE runs
            SET choices_json = ?, current_step = ?
            WHERE id = ?
        """, (json.dumps(choices, ensure_ascii=False), new_step, run_id))
        conn.commit()

        # Generate the next step right after a choice so the storyline stays connected
        if step in (1, 2, 3):
            try:
                _ensure_step_generated_for_run(int(run_id), step + 1)
            except Exception as e:
                _dbg(f"ensure_step_generated failed: {e}")

        # Recompute MBTI snapshot for this run
        try:
            recompute_and_store_mbti(conn, int(run_id))
        except Exception as e:
            _dbg(f"MBTI recompute failed: {e}")

        # Return updated run MBTI snapshot
        cur.execute("SELECT mbti_type, mbti_scores_json FROM runs WHERE id = ?", (run_id,))
        r2 = cur.fetchone()
        mbti_type = r2["mbti_type"] if r2 else None
        try:
            mbti_scores = json.loads(r2["mbti_scores_json"] or "{}") if r2 else {}
        except Exception:
            mbti_scores = {}

    return jsonify({"status": "success", "mbti_type": mbti_type, "mbti_scores": mbti_scores})
@app.route("/api/library", methods=["GET"])
def api_library():
    """Public list for the Library page (stores fast placeholder covers if missing)."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT s.id, s.created_at, s.user_id, s.cover_image_url, s.scenario_json, u.name AS author
            FROM scenarios s
            LEFT JOIN users u ON u.id = s.user_id
            WHERE COALESCE(s.is_public,1)=1
            ORDER BY s.id DESC
            LIMIT 200
        """)
        rows = cur.fetchall()

    items = []
    for r in rows:
        try:
            scen = json.loads(r["scenario_json"] or "{}")
        except Exception:
            scen = {}
        title = (scen.get("title") or f"Scenario #{r['id']}").strip()

        cover = r["cover_image_url"] or ""
        if not cover:
            # fast placeholder; real art gets generated later when someone plays with visuals
            try:
                cover = ensure_scenario_cover(int(r["id"]), prefer_openai=False)
            except Exception:
                cover = _svg_data_uri(title, "Cover")

        items.append({
            "id": int(r["id"]),
            "created_at": r["created_at"],
            "title": title,
            "author": r["author"] or "Anonymous",
            "cover_url": cover,
            "user_id": int(r["user_id"]) if r["user_id"] is not None else None,
            "can_remove": (
                (session.get("role") == "admin") or
                (session.get("is_moderator") == 1) or
                (
                    session.get("user_id") is not None and r["user_id"] is not None and
                    int(session.get("user_id")) == int(r["user_id"])
                )
            )
        })

    return jsonify({"status": "success", "items": items})


@app.route("/api/ensure_cover", methods=["POST"])
def api_ensure_cover():
    """Ensure the current run's scenario has a cover image saved in DB."""
    run_id = session.get("run_id")
    if not run_id:
        return jsonify({"status": "error", "message": "No active run."}), 404

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT r.images_enabled, r.scenario_id, s.cover_image_url
            FROM runs r
            JOIN scenarios s ON s.id = r.scenario_id
            WHERE r.id = ?
        """, (run_id,))
        row = cur.fetchone()

    if not row:
        return jsonify({"status": "error", "message": "Run not found."}), 404

    scenario_id = int(row["scenario_id"])
    images_enabled = int(row["images_enabled"] or 0) == 1

    try:
        url = ensure_scenario_cover(scenario_id, prefer_openai=bool(images_enabled))
        return jsonify({"status": "success", "cover_url": url})
    except Exception:
        return jsonify({"status": "error", "message": "Failed to ensure cover image."}), 500





@app.route("/api/finalize", methods=["POST"])
def api_finalize():
    run_id = session.get("run_id")
    if not run_id:
        return jsonify({"status": "error", "message": "No active run."}), 404

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT r.choices_json, r.final_json, s.scenario_json
            FROM runs r
            JOIN scenarios s ON s.id = r.scenario_id
            WHERE r.id = ?
        """, (run_id,))
        row = cur.fetchone()

    if not row:
        return jsonify({"status": "error", "message": "Run not found."}), 404

    choices = json.loads(row["choices_json"] or "[]")
    if len(choices) < 4:
        return jsonify({"status": "error", "message": "Complete steps 1–4 before finalizing."}), 400

    if row["final_json"]:
        return jsonify({"status": "success", "final": json.loads(row["final_json"])})

    scenario = json.loads(row["scenario_json"])
    final = generate_final_analysis(scenario, choices)

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE runs
            SET final_json = ?, current_step = 5
            WHERE id = ?
        """, (json.dumps(final, ensure_ascii=False), run_id))
        conn.commit()

    return jsonify({"status": "success", "final": final})


@app.route("/api/step_image", methods=["POST"])
def api_step_image():
    run_id = session.get("run_id")
    if not run_id:
        return jsonify({"status": "error", "message": "No active run."}), 404

    data = request.get_json(force=True)
    step = int(data.get("step") or 0)
    if step not in (1, 2, 3, 4, 5):
        return jsonify({"status": "error", "message": "step must be 1..5."}), 400

    try:
        url = generate_step_image_for_run(int(run_id), step)
        return jsonify({"status": "success", "url": url})
    except PermissionError as e:
        return jsonify({"status": "error", "message": str(e)}), 403
    except Exception:
        return jsonify({"status": "error", "message": "Failed to generate image."}), 500


# ============================================================
# Profile APIs (Horizon MBTI in profile)
# ============================================================
@app.route("/api/profile_mbti", methods=["GET"])
def api_profile_mbti():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"status": "error", "message": "Not logged in."}), 401

    with get_conn() as conn:
        cur = conn.cursor()

        # recent runs
        cur.execute("""
            SELECT r.id AS run_id, r.created_at, r.mbti_type, r.mbti_scores_json, s.scenario_json
            FROM runs r
            JOIN scenarios s ON s.id = r.scenario_id
            WHERE r.user_id = ?
            ORDER BY r.id DESC
            LIMIT 20
        """, (user_id,))
        rows = cur.fetchall()

        runs = []
        totals = {"SU": 0.0, "FG": 0.0, "HA": 0.0, "PC": 0.0}
        n = 0

        for r in rows:
            try:
                scen = json.loads(r["scenario_json"] or "{}")
            except Exception:
                scen = {}
            title = (scen.get("title") or "Scenario").strip()

            try:
                scores = json.loads(r["mbti_scores_json"] or "{}")
                if not isinstance(scores, dict):
                    scores = {}
            except Exception:
                scores = {}

            mbti_type = r["mbti_type"] or None

            # Only aggregate runs that have mbti_scores (i.e., at least one step signal)
            if scores:
                for k in totals.keys():
                    try:
                        totals[k] += float(scores.get(k, 0.0))
                    except Exception:
                        pass
                n += 1

            runs.append({
                "run_id": r["run_id"],
                "created_at": r["created_at"],
                "scenario_title": title,
                "mbti_type": mbti_type,
                "mbti_scores": scores,
            })

        overall_scores = {}
        if n > 0:
            overall_scores = {k: totals[k] / float(n) for k in totals.keys()}

        overall_type = axes_to_type(overall_scores) if overall_scores else None

    return jsonify({
        "status": "success",
        "user": {
            "id": user_id,
            "name": session.get("user_name") or "Student"
        },
        "overall": {
            "mbti_type": overall_type,
            "mbti_scores": overall_scores,
            "runs_count": n
        },
        "runs": runs
    })

@app.route("/api/profile_stats", methods=["GET"])
def api_profile_stats():
    """
    Aggregated, user-facing analytics for meaningful profile charts.
    Built from run_step_signals so it reflects real decision inputs.
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"status": "error", "message": "Not logged in."}), 401

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT created_at, step, primary_value, confidence,
                   risk_tolerance, oversight_preference, governance_preference,
                   decision_ms, axis_json
            FROM run_step_signals
            WHERE user_id = ?
            ORDER BY created_at ASC
        """, (user_id,))
        rows = cur.fetchall()

    # No data yet
    if not rows:
        return jsonify({"status": "success", "has_data": False, "summary": {}, "distributions": {}, "series": {}})

    # Summary metrics
    n = 0
    sums = {
        "confidence": 0.0,
        "risk_tolerance": 0.0,
        "oversight_preference": 0.0,
        "governance_preference": 0.0,
    }
    decision_times = []
    axes_totals = {"SU": 0.0, "FG": 0.0, "HA": 0.0, "PC": 0.0}

    # Distributions
    pv_counts = {k: 0 for k in PRIMARY_VALUES}
    step_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    type_counts = {}  # step-level derived type counts

    # Time series (monthly bucket)
    monthly = {}  # "YYYY-MM" -> aggregate axes and counts

    for r in rows:
        n += 1
        pv = (r["primary_value"] or "").strip().lower()
        if pv in pv_counts:
            pv_counts[pv] += 1

        try:
            st = int(r["step"] or 0)
            if st in step_counts:
                step_counts[st] += 1
        except Exception:
            pass

        # sliders
        for k in ("confidence", "risk_tolerance", "oversight_preference", "governance_preference"):
            try:
                sums[k] += float(r[k] or 0)
            except Exception:
                pass

        # decision time
        try:
            ms = r["decision_ms"]
            if ms is not None:
                ms = int(ms)
                if ms >= 0:
                    decision_times.append(ms)
        except Exception:
            pass

        # axes
        try:
            ax = json.loads(r["axis_json"] or "{}")
            for k in axes_totals.keys():
                axes_totals[k] += float(ax.get(k, 0.0))
        except Exception:
            ax = {}

        # type per step contribution snapshot
        try:
            t = axes_to_type({
                "SU": float(ax.get("SU", 0.0)),
                "FG": float(ax.get("FG", 0.0)),
                "HA": float(ax.get("HA", 0.0)),
                "PC": float(ax.get("PC", 0.0)),
            })
            type_counts[t] = type_counts.get(t, 0) + 1
        except Exception:
            pass

        # month bucket
        try:
            dt = (r["created_at"] or "")[:7]  # YYYY-MM
            if re.match(r"^\d{4}-\d{2}$", dt):
                if dt not in monthly:
                    monthly[dt] = {"n": 0, "SU": 0.0, "FG": 0.0, "HA": 0.0, "PC": 0.0}
                monthly[dt]["n"] += 1
                for k in ("SU", "FG", "HA", "PC"):
                    monthly[dt][k] += float(ax.get(k, 0.0))
        except Exception:
            pass

    avg = {k: round(sums[k] / float(n), 3) for k in sums.keys()}
    avg_axes = {k: round(axes_totals[k] / float(n), 4) for k in axes_totals.keys()}
    overall_type = axes_to_type(avg_axes)

    # decision time stats
    dt_stats = {}
    if decision_times:
        decision_times_sorted = sorted(decision_times)
        dt_stats = {
            "avg_ms": round(sum(decision_times_sorted) / float(len(decision_times_sorted)), 1),
            "p50_ms": float(decision_times_sorted[len(decision_times_sorted)//2]),
            "p90_ms": float(decision_times_sorted[max(0, int(len(decision_times_sorted)*0.9)-1)]),
        }

    # Monthly series -> avg axes per month
    months = sorted(monthly.keys())
    series = []
    for m in months:
        item = monthly[m]
        denom = float(item["n"] or 1)
        series.append({
            "month": m,
            "SU": round(item["SU"]/denom, 4),
            "FG": round(item["FG"]/denom, 4),
            "HA": round(item["HA"]/denom, 4),
            "PC": round(item["PC"]/denom, 4),
            "n": item["n"],
        })

    return jsonify({
        "status": "success",
        "has_data": True,
        "summary": {
            "overall_type": overall_type,
            "avg_sliders": avg,
            "avg_axes": avg_axes,
            "decision_time": dt_stats,
            "signals_count": n,
        },
        "distributions": {
            "primary_value": pv_counts,
            "steps": step_counts,
            "type_counts": type_counts,
        },
        "series": {
            "monthly_axes": series
        }
    })


# ============================================================
# Legacy placeholder (optional)
# ============================================================
@app.route("/simulate", methods=["POST"])
def simulate():
    data = request.get_json(force=True)
    policy_choice = data.get("policy", "Default")
    return jsonify({
        "status": "success",
        "future_impact": f"Analyzing '{policy_choice}'... Projected outcome: 25% increase in sustainable energy by 2040.",
        "ai_confidence": 0.92
    })
@app.route("/admin/scenario/<int:scenario_id>/remove", methods=["POST"])
def admin_remove_scenario(scenario_id):
    """
    Remove a scenario from the public Library (sets is_public=0).

    Allowed:
      - admin users
      - moderator emails (e.g., scott@test.com)
      - the scenario owner (creator)
    """
    user_id = session.get("user_id")
    is_admin = (session.get("role") == "admin")
    is_mod = (session.get("is_moderator") == 1)
    if not user_id and not is_admin and not is_mod:
        return jsonify({"ok": False, "error": "Unauthorized"}), 403

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM scenarios WHERE id = ?", (scenario_id,))
        row = cur.fetchone()
        if not row:
            return jsonify({"ok": False, "error": "Scenario not found."}), 404

        owner_id = row["user_id"]
        if (not is_admin) and (not is_mod):
            if owner_id is None or int(owner_id) != int(user_id):
                return jsonify({"ok": False, "error": "Unauthorized"}), 403

        cur.execute("UPDATE scenarios SET is_public = 0 WHERE id = ?", (scenario_id,))
        conn.commit()

    return jsonify({"ok": True})


@app.route("/admin/scenario/remove_all", methods=["POST"])
def admin_remove_all_scenarios():
    """Remove ALL scenarios from the public Library (sets is_public=0 for all). Admin or moderator only."""
    is_admin = (session.get("role") == "admin")
    is_mod = (session.get("is_moderator") == 1)
    if not (is_admin or is_mod):
        return jsonify({"ok": False, "error": "Unauthorized"}), 403
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE scenarios SET is_public = 0 WHERE COALESCE(is_public,1)=1")
        removed = cur.rowcount if cur.rowcount is not None else 0
        conn.commit()
    return jsonify({"ok": True, "removed": removed})


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
