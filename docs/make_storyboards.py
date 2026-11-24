"""Capture the storyboards from the running app (real screens, not mock-ups).

Starts src/server.py, opens demo URLs (?demo=1&jump=N&stage=...) in headless Chromium, saves each
screen, then lays the screens out as three storyboards with short step labels. Demo URLs never write
logs. Panels are chosen from the session plan, so they always match the current data.

Usage: python3 docs/make_storyboards.py [--chrome PATH]
"""
import argparse
import html
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "storyboards"
PORT = 8765
DEFAULT_CHROME = Path.home() / ".cache/ms-playwright/chromium-1140/chrome-linux/chrome"


def shot(chrome, url, path, w=760, h=900):
    subprocess.run([str(chrome), "--headless=old", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                    "--force-device-scale-factor=2", f"--window-size={w},{h}", "--virtual-time-budget=4000",
                    f"--screenshot={path}", url], check=True, capture_output=True)


def pick_items(plan):
    swap = plan["swap_at"]
    before = [it for it in plan["items"] if it["n"] < swap]
    after = [it for it in plan["items"] if it["n"] >= swap]
    # Before the swap: a question where the assistant is right, so following it pays off.
    learn = next(it for it in before[3:] if it["before"]["correct"])
    # After the swap: a right-to-wrong flip, the case the study is about.
    flip = next(it for it in after if it["flipped"] and it["before"]["correct"] and not it["after"]["correct"])
    return learn, flip


def wrong_letter(it, avoid):
    return next(l for l in "ABCD" if l != it["answer"] and l not in avoid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrome", default=str(DEFAULT_CHROME))
    args = ap.parse_args()
    plan = json.loads((ROOT / "analysis" / "session_plan.json").read_text())
    learn, flip = pick_items(plan)
    OUT.mkdir(parents=True, exist_ok=True)

    base = f"http://127.0.0.1:{PORT}/?pid=DEMO&demo=1"
    own_guess = wrong_letter(learn, {learn["before"]["pred"]})
    boards = [
        ("Storyboard 1 · Learning the assistant before the swap (all conditions)", "s1", [
            ("1  Answers alone first", f"c=silent&jump={learn['n']}&stage=own&own={own_guess}"),
            ("2  Sees suggestion, switches to it", f"c=silent&jump={learn['n']}&stage=ai&own={own_guess}&final={learn['before']['pred']}"),
            ("3  Feedback: assistant was right", f"c=silent&jump={learn['n']}&stage=feedback&own={own_guess}&final={learn['before']['pred']}"),
        ]),
        ("Storyboard 2 · Silent swap: a flipped answer is accepted (silent condition)", "s2", [
            ("1  Right alone; assistant now differs", f"c=silent&jump={flip['n']}&stage=ai&own={flip['answer']}&final={flip['after']['pred']}"),
            ("2  Feedback: following it was wrong", f"c=silent&jump={flip['n']}&stage=feedback&own={flip['answer']}&final={flip['after']['pred']}"),
        ]),
        ("Storyboard 3 · Notified swap: the same question with a notice (notice condition)", "s3", [
            ("1  Update notice at the swap", f"c=notice&jump={plan['swap_at']}&stage=notice"),
            ("2  Build label; keeps own answer", f"c=notice&jump={flip['n']}&stage=ai&own={flip['answer']}&final={flip['answer']}"),
            ("3  Feedback: right, assistant wrong", f"c=notice&jump={flip['n']}&stage=feedback&own={flip['answer']}&final={flip['answer']}"),
        ]),
    ]

    server = subprocess.Popen([sys.executable, str(ROOT / "src" / "server.py"), str(PORT)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1.5)
        for _, key, panels in boards:
            for i, (_, query) in enumerate(panels):
                shot(args.chrome, f"{base}&{query}", OUT / f"{key}_{i + 1}.png")
    finally:
        server.terminate()

    fonts = "".join(
        f"@font-face {{ font-family: 'Segoe UI'; font-weight: {w}; src: local('Segoe UI'), url('file:///mnt/c/Windows/Fonts/{f}'); }}"
        for w, f in [(400, "segoeui.ttf"), (600, "seguisb.ttf"), (700, "segoeuib.ttf")])
    css = (fonts + "body{margin:0;background:#fff;font-family:'Segoe UI',Arial,sans-serif;color:#1a1a19}"
           ".b{padding:18px 22px 22px}.t{font-size:22px;font-weight:700;margin:0 0 14px}"
           ".row{display:flex;gap:16px}.p{flex:1}.l{font-size:18px;font-weight:600;margin:0 0 8px;color:#52514e}"
           ".p img{width:100%;border:1.5px solid #d6d4ce;border-radius:10px;display:block}")
    for title, key, panels in boards:
        cells = "".join(f"<div class='p'><div class='l'>{html.escape(label)}</div><img src='{key}_{i + 1}.png'></div>"
                        for i, (label, _) in enumerate(panels))
        page = OUT / f"{key}.html"
        page.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>"
                        f"<div class='b'><div class='t'>{html.escape(title)}</div><div class='row'>{cells}</div></div></body></html>")
        width, height = (1500, 680) if len(panels) == 3 else (1040, 700)
        subprocess.run([args.chrome, "--headless=old", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                        "--allow-file-access-from-files", "--force-device-scale-factor=2", f"--window-size={width},{height}",
                        f"--screenshot={OUT.parent / (key + '.png')}", f"file://{page}"], check=True, capture_output=True)
        print("wrote", OUT.parent / f"{key}.png")


if __name__ == "__main__":
    main()
