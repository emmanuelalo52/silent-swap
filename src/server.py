"""Study server: serves the web interface, hands each participant a session, records every answer.

Standard library only, so a reviewer can run it without installing anything:
    python3 src/server.py            then open http://localhost:8000/?pid=P01&c=silent

Conditions (?c=...):
    control  FP16 build answers all 24 questions (no swap)
    silent   the after build answers from the swap point; nothing tells the participant
    notice   same swap, plus a one-time update notice and a build label beside each answer
Add &demo=1 to try the interface without writing a log; &jump=N&stage=own|ai|feedback&own=X
opens a given screen directly (used to capture the storyboards).

The assistant's answers come from the recorded runs (runs/*.json) via analysis/session_plan.json,
so sessions replay the real model outputs without a GPU. Correct answers stay on the server and
are only returned after the participant commits a final answer.
"""
import json
import re
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
PLAN_PATH = ROOT / "analysis" / "session_plan.json"
LOGS = ROOT / "logs"
CONDITIONS = ("control", "silent", "notice")
PID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
LETTERS = set("ABCD")


def load_plan():
    return json.loads(PLAN_PATH.read_text())


def served(plan, item, condition):
    """Which build answers this item for this condition (the serving router)."""
    swapped = condition != "control" and item["n"] >= plan["swap_at"]
    return ("after" if swapped else "before"), swapped


def session_for(plan, condition):
    """The participant's view of the session: no correct answers, only what the assistant says."""
    items = []
    for it in plan["items"]:
        side, _ = served(plan, it, condition)
        ai = it[side]
        items.append({"n": it["n"], "subject": it["subject"], "question": it["question"],
                      "choices": it["choices"],
                      "ai": {"letter": ai["pred"], "confidence": round(ai["confidence"], 3)},
                      # provenance label: shown only in the notice condition (see web/app.js)
                      "build": plan[side]["name"]})
    return {"condition": condition, "swap_at": plan["swap_at"], "n_items": len(items),
            "subjects": plan["subjects"], "items": items,
            "notice": plan["notice"] if condition == "notice" else None}


def check_answer(plan, body):
    """Validate a submitted answer and build the feedback and the log record."""
    pid, condition = str(body.get("pid", "")), body.get("condition")
    if not PID_RE.match(pid) or condition not in CONDITIONS:
        raise ValueError("bad pid or condition")
    n = int(body.get("n", 0))
    if not 1 <= n <= len(plan["items"]):
        raise ValueError("bad item number")
    initial, final = body.get("initial"), body.get("final")
    if initial not in LETTERS or final not in LETTERS:
        raise ValueError("answers must be A-D")
    it = plan["items"][n - 1]
    side, swapped = served(plan, it, condition)
    ai = it[side]["pred"]
    record = {
        "pid": pid, "condition": condition, "n": n, "id": it["id"], "subject": it["subject"],
        "build": plan[side]["name"], "swapped": swapped, "flipped_item": it["flipped"],
        "answer": it["answer"], "ai": ai, "ai_confidence": round(it[side]["confidence"], 3),
        "ai_correct": ai == it["answer"], "initial": initial, "final": final,
        "final_correct": final == it["answer"],
        "followed_ai": final == ai, "switched_to_ai": initial != ai and final == ai,
        "ms_to_initial": body.get("ms_to_initial"), "ms_to_final": body.get("ms_to_final"),
        "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    feedback = {"answer": it["answer"], "ai": ai, "ai_correct": record["ai_correct"],
                "you_correct": record["final_correct"]}
    return record, feedback


class Handler(SimpleHTTPRequestHandler):
    def _json(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path != "/api/session":
            return super().do_GET()
        q = parse_qs(url.query)
        condition = q.get("c", ["silent"])[0]
        if condition not in CONDITIONS:
            return self._json(400, {"error": f"condition must be one of {CONDITIONS}"})
        self._json(200, session_for(load_plan(), condition))

    def do_POST(self):
        if urlparse(self.path).path != "/api/answer":
            return self._json(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            record, feedback = check_answer(load_plan(), body)
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            return self._json(400, {"error": str(e)})
        if not body.get("demo"):
            LOGS.mkdir(exist_ok=True)
            with open(LOGS / f"{record['pid']}.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
        self._json(200, feedback)

    def log_message(self, fmt, *args):  # keep the console quiet apart from errors
        if not str(args[1] if len(args) > 1 else "").startswith("2"):
            super().log_message(fmt, *args)


def main(port=8000):
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(WEB)))
    print(f"study server on http://localhost:{port}/?pid=P01&c=silent  (conditions: {', '.join(CONDITIONS)})")
    server.serve_forever()


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
