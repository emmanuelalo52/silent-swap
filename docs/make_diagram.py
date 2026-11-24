"""Draw the system diagram (docs/system_diagram.svg and .html; render the HTML to PNG with Chromium).

Border style shows status: solid = implemented, dashed = simulated, dotted grey = planned.
Usage: python3 docs/make_diagram.py
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent
W, H = 1500, 860
INK, MUTED, RULE = "#1a1a19", "#5a5955", "#9a9893"
BLUE, BLUE_SOFT = "#2a78d6", "#eef4fc"
AI, AI_SOFT = "#6250d6", "#f1effc"
SIM, PLAN = "#b26a00", "#8c8a85"
FONT = "'Segoe UI', Arial, sans-serif"

parts = []


def text(x, y, s, size=17, weight=400, fill=INK, anchor="start", style="normal"):
    parts.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
                 f'font-style="{style}" fill="{fill}" text-anchor="{anchor}">{s}</text>')


def box(x, y, w, h, title, lines=(), status="impl", fill="#ffffff", ai=False):
    dash = {"impl": "", "sim": 'stroke-dasharray="9 6"', "plan": 'stroke-dasharray="2 5"'}[status]
    stroke = {"impl": INK, "sim": SIM, "plan": PLAN}[status]
    if ai:
        fill, stroke = AI_SOFT, AI
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="2.2" {dash}/>')
    text(x + 16, y + 30, title, 19, 700, PLAN if status == "plan" else INK)
    for i, line in enumerate(lines):
        text(x + 16, y + 56 + i * 23, line, 15.5, 400, MUTED)


def arrow(x1, y1, x2, y2, label=None, status="impl", lx=None, ly=None, anchor="middle"):
    dash = {"impl": "", "sim": 'stroke-dasharray="9 6"', "plan": 'stroke-dasharray="2 5"'}[status]
    color = {"impl": INK, "sim": SIM, "plan": PLAN}[status]
    marker = {"impl": "a", "sim": "as", "plan": "ap"}[status]
    parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2" '
                 f'{dash} marker-end="url(#{marker})"/>')
    if label:
        text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2 - 8,
             label, 14, 400, MUTED, anchor)


def lane(y, h, title):
    parts.append(f'<rect x="20" y="{y}" width="{W - 40}" height="{h}" rx="16" fill="#f6f5f2"/>')
    text(40, y + 30, title, 15, 700, MUTED)


parts.append("""<defs>
<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#1a1a19"/></marker>
<marker id="as" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#b26a00"/></marker>
<marker id="ap" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#8c8a85"/></marker>
</defs>""")

# ---- offline lane
lane(20, 210, "OFFLINE  ·  measure the serving change (run once on a GPU)")
box(40, 70, 210, 130, "MMLU subset", ["1,140 questions", "57 subjects"])
box(300, 70, 290, 130, "Model runner  (AI)", ["Qwen3-0.6B scores A-D", "FP16 build", "INT8 / FP8 / INT4 builds*"], ai=True)
box(640, 70, 230, 130, "Recorded runs", ["answer + confidence", "per question, per build"])
box(920, 70, 230, 130, "Flip analyser", ["accuracy, flips,", "flips by confidence"])
box(1200, 70, 260, 130, "Session planner", ["picks swap build", "notice rule (flips >= 2%)", "24 questions"])
arrow(250, 135, 298, 135)
arrow(590, 135, 638, 135)
arrow(870, 135, 918, 135)
arrow(1150, 135, 1198, 135)

# ---- online lane
lane(250, 400, "ONLINE  ·  study session (no GPU needed)")
box(40, 340, 190, 110, "Participant", ["browser"], fill=BLUE_SOFT)
box(290, 300, 300, 190, "Web interface", ["1  own answer", "2  AI suggestion + confidence", "3  final answer, then feedback",
                                           "notice + build label", "(notice condition only)"])
box(740, 290, 400, 220, "Session server", [])
box(760, 335, 360, 40, "", [])
text(776, 361, "Condition: control / silent / notice", 15.5, 600)
box(760, 385, 360, 40, "", [], ai=True)
text(776, 411, "Serving router: FP16, swap at Q13  (AI)", 15.5, 600)
box(760, 435, 360, 40, "", [])
text(776, 461, "Change notifier  ·  Logger", 15.5, 600)
box(1200, 300, 260, 100, "Session logs", ["one JSONL file", "per participant"])
box(1200, 440, 260, 110, "Reliance analysis", ["agreement before / after", "on flipped vs stable", "questions"])
arrow(230, 380, 288, 380, "answers", lx=259, ly=370)
arrow(288, 420, 232, 420, "screens", lx=259, ly=446)
arrow(590, 370, 738, 370, "get session, answer", lx=664, ly=360)
arrow(738, 430, 592, 430, "AI answer, feedback", lx=665, ly=456)
arrow(1140, 350, 1198, 350)
arrow(1330, 400, 1330, 438)
arrow(1330, 200, 1330, 262, None)
arrow(1330, 262, 1142, 292, "session plan", lx=1268, ly=258)
box(920, 545, 220, 90, "Scripted participants", ["3 policies x 3 conditions", "pipeline test"], status="sim")
arrow(1030, 545, 1030, 512, status="sim")

# ---- planned
lane(670, 170, "PLANNED")
box(40, 710, 230, 110, "Participant study", ["ethics approval", "~150 people online"], status="plan")
box(290, 710, 330, 110, "Per-answer change flag", ["shadow FP16 check", "marks answers that changed"], status="plan")
box(740, 710, 330, 110, "Live inference server", ["serve + hot-swap builds", "instead of replaying runs"], status="plan")
arrow(135, 710, 135, 452, status="plan")
arrow(455, 710, 455, 492, status="plan")
arrow(820, 710, 820, 512, status="plan")

# ---- legend
lx, ly = 1130, 712
text(lx, ly + 4, "Status", 15, 700, MUTED)
for i, (st, label) in enumerate([("impl", "implemented"), ("sim", "simulated"), ("plan", "planned")]):
    dash = {"impl": "", "sim": 'stroke-dasharray="9 6"', "plan": 'stroke-dasharray="2 5"'}[st]
    stroke = {"impl": INK, "sim": SIM, "plan": PLAN}[st]
    parts.append(f'<rect x="{lx}" y="{ly + 18 + i * 30}" width="44" height="20" rx="5" fill="#fff" '
                 f'stroke="{stroke}" stroke-width="2.2" {dash}/>')
    text(lx + 56, ly + 34 + i * 30, label, 15)
parts.append(f'<rect x="{lx + 160}" y="{ly + 18}" width="44" height="20" rx="5" fill="{AI_SOFT}" stroke="{AI}" stroke-width="2.2"/>')
text(lx + 216, ly + 34, "AI behaviour", 15)
text(lx, ly + 122, "* quantized numerics simulated in FP16", 13.5, 400, MUTED)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
       f'<rect width="{W}" height="{H}" fill="#ffffff"/>' + "\n".join(parts) + "</svg>")
(OUT / "system_diagram.svg").write_text(svg)
fonts = "".join(
    f"@font-face {{ font-family: 'Segoe UI'; font-weight: {w}; src: local('Segoe UI'), url('file:///mnt/c/Windows/Fonts/{f}'); }}"
    for w, f in [(400, "segoeui.ttf"), (600, "seguisb.ttf"), (700, "segoeuib.ttf")])
(OUT / "system_diagram.html").write_text(
    f"<!doctype html><html><head><meta charset='utf-8'><style>{fonts} html,body{{margin:0;background:#fff}}</style>"
    f"</head><body>{svg}</body></html>")
print("wrote", OUT / "system_diagram.svg")
