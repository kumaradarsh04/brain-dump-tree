"""
Brain Dump -> Tree backend
---------------------------
A tiny Flask server that takes raw brain-dump text and asks the Gemini API
to turn it into a hierarchical task list (categories + sub-tasks).

Payments use a Razorpay Payment Link you create once in the Razorpay
dashboard (no code needed for that part — see DEPLOYMENT.md). This backend
only needs to verify the webhook Razorpay sends after that link gets paid.

Setup
-----
    pip install flask flask-cors requests gunicorn

    export GEMINI_API_KEY="your-gemini-key"
    export RAZORPAY_WEBHOOK_SECRET="..."   # the secret you set when creating the webhook

Run
---
    python backend.py

The server listens on http://localhost:5000 and exposes:

    POST /organize
        body: {"text": "raw brain dump ...", "email": "user@example.com"}
        returns: {"nodes": [...]}  OR  402 + {"error": "free_limit_reached"} once the
        free daily limit (FREE_DAILY_LIMIT) is used up for that email

    POST /webhook
        Razorpay calls this automatically after your shared Payment Link is paid.
        This is what actually marks an email as premium/unlimited — never trust
        the frontend for this.

    GET /health
        simple liveness check

Storage is two flat JSON files (premium_users.json, usage.json) — fine for an early
MVP with a handful of users; swap for a real database once you outgrow it.

Then open index.html in a browser (it calls http://localhost:5000/organize).
See DEPLOYMENT.md for putting this in front of real users and setting up the
Payment Link + webhook.
"""

import hashlib
import hmac
import json
import os
import re
from datetime import date

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow index.html (opened as a local file or served separately) to call this

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# --- Payments (Razorpay Payment Link) -----------------------------------
# You create the Payment Link itself directly in the Razorpay dashboard (no
# code needed for that part — see DEPLOYMENT.md). This backend only needs to
# verify the webhook Razorpay sends after a real payment, so it can unlock
# that customer's account. Nothing here costs money to set up.
RAZORPAY_WEBHOOK_SECRET = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")

FREE_DAILY_LIMIT = 3
PREMIUM_FILE = "premium_users.json"
USAGE_FILE = "usage.json"


def _load(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _save(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def is_premium(email):
    return _load(PREMIUM_FILE).get(email.lower(), False)


def mark_premium(email):
    data = _load(PREMIUM_FILE)
    data[email.lower()] = True
    _save(PREMIUM_FILE, data)


def check_and_increment_usage(email):
    """Returns True if this email still has free organizes left today."""
    today = str(date.today())
    data = _load(USAGE_FILE)
    key = email.lower()
    record = data.get(key, {})
    count = record.get(today, 0)
    if count >= FREE_DAILY_LIMIT:
        return False
    record[today] = count + 1
    data[key] = record
    _save(USAGE_FILE, data)
    return True

# Pick whichever current Gemini model fits your budget/latency needs.
# "gemini-2.5-flash" is a solid, inexpensive default as of 2026.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

PROMPT_TEMPLATE = """Convert this messy brain dump into a well-organized hierarchical task list. Group related tasks under short category headings when it makes sense (2-6 word headings), and break down vague or large tasks into 2-4 concrete sub-steps only when genuinely useful. Keep item text short and action-oriented. Do not invent unrelated tasks.

Respond with ONLY a raw JSON array, no markdown fences, no commentary, in exactly this shape:
[{{"text": "Category or task", "children": [{{"text": "sub task", "children": []}}]}}]

If a top-level item has no natural sub-items, use an empty children array.

Brain dump:
\"\"\"
{dump}
\"\"\"
"""


def strip_code_fences(text: str) -> str:
    """Gemini sometimes wraps JSON in ```json ... ``` fences — strip them if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def normalize_nodes(nodes):
    """Recursively validate/clean the shape returned by the model."""
    clean = []
    if not isinstance(nodes, list):
        return clean
    for item in nodes:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        children = normalize_nodes(item.get("children", []))
        clean.append({"text": text, "children": children})
    return clean


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": GEMINI_MODEL, "key_configured": bool(GEMINI_API_KEY)})


@app.route("/organize", methods=["POST"])
def organize():
    if not GEMINI_API_KEY:
        return jsonify({"error": "GEMINI_API_KEY is not set on the server"}), 500

    payload = request.get_json(silent=True) or {}
    dump_text = (payload.get("text") or "").strip()
    email = (payload.get("email") or "").strip().lower()

    if not dump_text:
        return jsonify({"error": "No text provided"}), 400
    if not email:
        return jsonify({"error": "No email provided"}), 400

    if not is_premium(email):
        if not check_and_increment_usage(email):
            return jsonify({
                "error": "free_limit_reached",
                "message": f"You've used your {FREE_DAILY_LIMIT} free organizes for today.",
            }), 402

    prompt = PROMPT_TEMPLATE.format(dump=dump_text)

    request_body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "responseMimeType": "application/json",
        },
    }

    try:
        response = requests.post(
            GEMINI_URL,
            params={"key": GEMINI_API_KEY},
            json=request_body,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            return jsonify({"error": "Gemini returned no candidates", "raw": data}), 502

        parts = candidates[0].get("content", {}).get("parts", [])
        raw_text = "".join(p.get("text", "") for p in parts)
        raw_text = strip_code_fences(raw_text)

        parsed = json.loads(raw_text)
        nodes = normalize_nodes(parsed)

        return jsonify({"nodes": nodes})

    except requests.exceptions.HTTPError as e:
        return jsonify({"error": f"Gemini API error: {e}", "details": response.text}), 502
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Could not parse Gemini's response as JSON: {e}", "raw": raw_text}), 502
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request to Gemini failed: {e}"}), 502


@app.route("/webhook", methods=["POST"])
def razorpay_webhook():
    """Razorpay calls this automatically after your shared Payment Link is
    paid. This is what actually unlocks unlimited use — never trust the
    frontend alone for this, since anyone could fake a browser event."""
    payload = request.data
    signature = request.headers.get("X-Razorpay-Signature", "")

    expected_signature = hmac.new(
        RAZORPAY_WEBHOOK_SECRET.encode("utf-8"), payload, hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected_signature, signature):
        return jsonify({"error": "Invalid webhook signature"}), 400

    event = json.loads(payload)
    event_type = event.get("event", "")

    # Triggered when your Payment Link is paid. The customer's own email —
    # the one they typed into Razorpay's checkout page — comes back here.
    if event_type == "payment_link.paid":
        payment_entity = (
            event.get("payload", {}).get("payment", {}).get("entity", {})
        )
        email = (payment_entity.get("email") or "").strip().lower()
        if email:
            mark_premium(email)

    return jsonify({"received": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
