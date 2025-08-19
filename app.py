# app.py
"""
GenAI Architect MVP — Free / Open Source / Streamlit Community-ready

Features:
- RAG search (lightweight keyword-based simulation)
- Agentic simulation (planner + tool execution)
- Guardrails (safety checks & PII redaction)
- Observability (logs, simple charts)
- Compliance dashboard (stub)
- Cloud connector stubs (Azure, AWS, GCP) — simulated (no real API calls)
- CI/CD pipeline simulation checklist (shows you understand deploy steps)
- Feedback loop (thumbs + notes) saved to logs/feedback.jsonl

This app uses only very small dependencies so it runs reliably on Streamlit Community.
"""

import os
import re
import time
import json
import uuid
import random
from datetime import datetime
from typing import List

import streamlit as st
import pandas as pd

# ----------------- Configuration -----------------
APP_TITLE = "GenAI Architect MVP • Free OSS (Streamlit)"
DATA_DIR = "data"
LOG_DIR = "logs"
EVENT_LOG = os.path.join(LOG_DIR, f"{datetime.utcnow().date()}.jsonl")
FEEDBACK_LOG = os.path.join(LOG_DIR, "feedback.jsonl")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ----------------- Demo data ensure -----------------
def ensure_demo_files():
    doc_path = os.path.join(DATA_DIR, "demo_doc.txt")
    policy_path = os.path.join(DATA_DIR, "demo_policy.txt")
    if not os.path.exists(doc_path):
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(
                "Generative AI is the capability of AI systems to generate new content.\n"
                "RAG stands for Retrieval-Augmented Generation; it combines search+generation.\n"
                "Agentic AI enables autonomous agents with planning and memory.\n"
                "Guardrails ensure compliance, safety, and trust in AI outputs.\n"
                "Monitoring ensures observability and post-deployment tracking."
            )
    if not os.path.exists(policy_path):
        with open(policy_path, "w", encoding="utf-8") as f:
            f.write(
                "Company Policy:\n"
                "1. AI systems must not produce unsafe or biased outputs.\n"
                "2. PII must be redacted before any data export.\n"
                "3. All production actions must be logged and auditable.\n"
            )

ensure_demo_files()

# ----------------- Helper functions -----------------
def log_event(kind: str, payload: dict):
    p = dict(payload)
    p["kind"] = kind
    p["ts"] = datetime.utcnow().isoformat()
    with open(EVENT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

def append_feedback(rec: dict):
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ----------------- Guardrails -----------------
PROFANITY = re.compile(r"\b(fuck|shit|bitch|asshole|bastard)\b", re.I)
PII = re.compile(r"\b(\d{3}-\d{2}-\d{4}|\d{12}|\d{16}|[0-9]{4}\s?[0-9]{4}\s?[0-9]{4}\s?[0-9]{4})\b")
JAIL = re.compile(r"(ignore (all|previous) instructions|bypass|disable safety|jailbreak|prompt injection)", re.I)

def validate_query(q: str) -> List[str]:
    errs = []
    if not q or len(q.strip()) < 3:
        errs.append("Prompt too short.")
    if PROFANITY.search(q):
        errs.append("Contains profanity.")
    if PII.search(q):
        errs.append("Possible PII detected.")
    if JAIL.search(q):
        errs.append("Potential jailbreak attempt.")
    return errs

def moderate_text(t: str) -> str:
    t = PROFANITY.sub("[REDACTED]", t)
    t = PII.sub("[REDACTED_ID]", t)
    return t

# ----------------- Lightweight "RAG" retrieval (keyword match) -----------------
def load_kb_texts():
    texts = []
    for fname in os.listdir(DATA_DIR):
        if fname.lower().endswith((".txt", ".md")):
            with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8") as fh:
                texts.append({"source": fname, "text": fh.read()})
    return texts

def simple_retrieve(query: str, kb_texts, top_k: int = 3):
    query_terms = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
    scored = []
    for doc in kb_texts:
        score = 0
        for term in query_terms:
            score += doc["text"].lower().count(term)
        scored.append({"score": score, "source": doc["source"], "text": doc["text"]})
    scored = sorted(scored, key=lambda x: -x["score"])
    # return top_k with score>0 else return a best-effort summary
    hits = [s for s in scored if s["score"] > 0][:top_k]
    if not hits:
        # fallback: first doc snippets
        hits = [{"score": 0, "source": d["source"], "text": d["text"][:400]} for d in kb_texts[:top_k]]
    return hits

# ----------------- Agentic planner & executor (simulation) -----------------
def plan_steps(query: str):
    steps = []
    steps.append({"tool": "retriever", "arg": query})
    if re.search(r"\bcalculate\b|\bcompute\b|\d+\s*[\+\-\*/]\s*\d+", query.lower()):
        steps.append({"tool": "calculator", "arg": query})
    if re.search(r"https?://", query.lower()) or query.lower().startswith("read "):
        steps.append({"tool": "url_reader", "arg": query})
    steps.append({"tool": "synthesize", "arg": query})
    return steps

def execute_plan(plan, kb_texts, top_k=3):
    trace = {"steps": [], "retrieved": [], "calc": None, "synth": None}
    for step in plan:
        tool = step["tool"]
        arg = step["arg"]
        trace["steps"].append(tool)
        if tool == "retriever":
            hits = simple_retrieve(arg, kb_texts, top_k=top_k)
            trace["retrieved"] = hits
        elif tool == "calculator":
            # safe simple calc: extract first expression like "12 * 24"
            m = re.search(r"(\d+\s*[\+\-\*/]\s*\d+)", arg)
            if m:
                try:
                    expr = m.group(1)
                    # very restricted eval
                    calc_res = eval(expr, {"__builtins__": {}}, {})
                    trace["calc"] = str(calc_res)
                except Exception as e:
                    trace["calc"] = f"calc_error: {e}"
            else:
                trace["calc"] = "no-expression-found"
        elif tool == "url_reader":
            trace.setdefault("url_reader", []).append({"url": arg, "note": "url reader disabled in free demo"})
        elif tool == "synthesize":
            # template-based synthesis (no LLM)
            if trace.get("retrieved"):
                first = trace["retrieved"][0]["text"]
                first_sent = first.split(".")[0].strip()
                synth = f"Based on sources: {first_sent}. (See [1])"
                if trace.get("calc"):
                    synth += f" Calculator: {trace['calc']}"
                trace["synth"] = synth
            else:
                synth = "No relevant context found in KB."
                if trace.get("calc"):
                    synth += f" Calculator: {trace['calc']}"
                trace["synth"] = synth
    return trace

# ----------------- Cloud connector stubs -----------------
def simulate_cloud_connect(provider: str):
    # Simulate connection latency + result
    time.sleep(0.5)
    if provider not in ["Azure", "AWS", "GCP"]:
        return {"ok": False, "msg": f"{provider} not recognized."}
    # randomized success to demo connection behavior
    if random.random() < 0.95:
        return {"ok": True, "msg": f"Simulated connection to {provider} successful (demo mode)."}
    else:
        return {"ok": False, "msg": f"Simulated {provider} connection failed (demo) — check credentials."}

# ----------------- CI/CD simulation -----------------
CI_STEPS = [
    {"name": "Checkout", "done": True},
    {"name": "Install deps", "done": True},
    {"name": "Run unit tests (smoke)", "done": True},
    {"name": "Build container (optional)", "done": False},
    {"name": "Deploy to staging", "done": False},
    {"name": "Run integration tests", "done": False},
    {"name": "Promote to prod", "done": False}
]

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.markdown("**Free & Open Source demo** that maps to enterprise GenAI Architect responsibilities. "
            "Cloud connectors are simulated — swap in real APIs in production.")

kb_texts = load_kb_texts()

# Top nav
tab = st.radio("Mode", ["RAG Search", "Agentic Agent", "Guardrails", "Compliance", "Cloud Connectors", "CI/CD", "Monitoring & Logs"], index=0)

if tab == "RAG Search":
    st.header("🔎 RAG Search (lightweight simulation)")
    q = st.text_input("Enter query for RAG", value="What is RAG?")
    k = st.slider("Top-K", 1, 5, 3)
    if st.button("Search"):
        errs = validate_query(q)
        if errs:
            st.error("Blocked by guardrails: " + "; ".join(errs))
            log_event("blocked_input", {"prompt": q, "errors": errs})
        else:
            hits = simple_retrieve(q, kb_texts, top_k=k)
            for i, h in enumerate(hits, start=1):
                st.markdown(f"**[{i}] {h['source']}** (score={h['score']})")
                st.write(h['text'][:1000])
            log_event("rag_search", {"q": q, "results": len(hits)})

elif tab == "Agentic Agent":
    st.header("🤖 Agentic Agent (planner → tools → synth)")
    q = st.text_input("Agent question", value="Summarize Policy Alpha and list customer-facing risks.")
    top_k = st.slider("Top-K retrieved", 1, 4, 2)
    if st.button("Run Agent"):
        errs = validate_query(q)
        if errs:
            st.error("Blocked by guardrails: " + "; ".join(errs))
            log_event("blocked_input", {"prompt": q, "errors": errs})
        else:
            plan = plan_steps(q)
            st.markdown("**Plan (steps):**")
            for s in plan:
                st.write(f"- {s['tool']}: {s['arg'][:120]}")
            trace = execute_plan(plan, kb_texts, top_k=top_k)
            st.markdown("**Synthesis (template):**")
            st.success(moderate_text(trace.get("synth", "")))
            with st.expander("Execution Trace & Retrieved Context"):
                st.json(trace)
            log_event("agent_run", {"q": q, "trace_summary": {"retrieved": len(trace.get("retrieved", []))}})

elif tab == "Guardrails":
    st.header("🛡️ Guardrails & Safety")
    sample = st.text_area("Type text to test guardrails", value="This contains no policy violation.")
    if st.button("Check"):
        errs = validate_query(sample)
        st.write("Validation errors:", errs)
        st.write("Moderated text:", moderate_text(sample))
        log_event("guardrail_check", {"input": sample, "errors": errs})

elif tab == "Compliance":
    st.header("📋 Compliance Dashboard (stub)")
    st.markdown("This is a **mock** compliance dashboard showing key enterprise controls.")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("PII Redaction Enabled", "✅")
        st.metric("Audit Logging", "✅")
        st.metric("RBAC Enforcement", "✅")
    with col2:
        st.metric("SLA (99.9%)", "design")
        st.metric("Data Residency", "demo/NA")
        st.metric("Encryption at Rest", "design")
    st.markdown("**Compliance Notes:** Logs are written to `logs/` and feedback to `logs/feedback.jsonl` for evaluation pipelines.")
    if st.button("Export audit snapshot"):
        # produce a tiny JSON that would be exported in prod
        snapshot = {"ts": datetime.utcnow().isoformat(), "pii_redaction": True, "logs": "logs/ (demo)"}
        st.download_button("Download snapshot (JSON)", json.dumps(snapshot, indent=2), file_name="audit_snapshot.json")
        log_event("audit_export", {"snapshot_ok": True})

elif tab == "Cloud Connectors":
    st.header("☁️ Cloud Connector Stubs (simulated)")
    provider = st.selectbox("Choose provider to simulate connect", ["Azure", "AWS", "GCP"])
    if st.button("Simulate connect"):
        res = simulate_cloud_connect(provider)
        if res["ok"]:
            st.success(res["msg"])
        else:
            st.error(res["msg"])
        log_event("cloud_sim", {"provider": provider, "result": res})

elif tab == "CI/CD":
    st.header("🔁 CI / CD Pipeline (simulation)")
    st.markdown("This demo shows the steps you'd have in a production CI/CD using GitHub Actions, tests, and IaC.")
    for step in CI_STEPS:
        st.write(f"- [{'✅' if step['done'] else '⬜'}] {step['name']}")
    if st.button("Run smoke tests (simulate)"):
        st.info("Running smoke tests...")
        time.sleep(1)
        st.success("Smoke tests passed (simulated).")
        # mark steps as done in the in-memory simulation
        for s in CI_STEPS:
            if s["name"] == "Build container (optional)":
                s["done"] = True
        log_event("ci_cd_sim", {"result": "smoke_ok"})

elif tab == "Monitoring & Logs":
    st.header("📈 Monitoring & Logs")
    # simple synthetic metrics
    total_infers = 0
    blocked = 0
    if os.path.exists(EVENT_LOG):
        with open(EVENT_LOG, "r", encoding="utf-8") as f:
            for ln in f:
                try:
                    obj = json.loads(ln)
                    if obj.get("kind") in ("agent_run", "rag_search", "inference"):
                        total_infers += 1
                    if obj.get("kind") == "blocked_input":
                        blocked += 1
                except:
                    pass
    st.metric("Total Inferences (today)", total_infers)
    st.metric("Blocked Inputs (today)", blocked)
    if os.path.exists(FEEDBACK_LOG):
        with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
            feedback_lines = [json.loads(l) for l in f.readlines() if l.strip()]
        if feedback_lines:
            df = pd.DataFrame(feedback_lines)
            st.markdown("Recent feedback")
            st.dataframe(df.tail(10))
    if st.button("Clear demo logs (local)"):
        # Danger: for demo only
        open(EVENT_LOG, "w").close()
        open(FEEDBACK_LOG, "w").close()
        st.success("Logs cleared (demo).")
        log_event("logs_cleared", {"by": "user"})

# ----------------- feedback UI (global) -----------------
st.markdown("---")
st.write("## Feedback / Eval (demo)")
fb_col1, fb_col2 = st.columns(2)
with fb_col1:
    if st.button("👍 This demo was useful"):
        rec = {"id": str(uuid.uuid4()), "score": 1, "note": "", "ts": datetime.utcnow().isoformat()}
        append_feedback(rec)
        st.success("Thanks for the positive feedback!")
        log_event("feedback", rec)
with fb_col2:
    note = st.text_input("If you found issues, leave a quick note here")
    if st.button("Submit feedback note"):
        rec = {"id": str(uuid.uuid4()), "score": 0, "note": note, "ts": datetime.utcnow().isoformat()}
        append_feedback(rec)
        st.success("Thanks — feedback recorded.")
        log_event("feedback", rec)
