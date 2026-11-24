"""Count README prose words, including table text; skips code blocks, images, markup and URLs.

Usage: python3 docs/readme_wordcount.py
"""
import re
from pathlib import Path

text = (Path(__file__).resolve().parent.parent / "README.md").read_text()
text = re.sub(r"```.*?```", " ", text, flags=re.S)
words = 0
for line in text.splitlines():
    s = line.strip()
    if not s or s.startswith("![") or set(s) <= set("|-: "):
        continue
    s = s.replace("|", " ")
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)   # keep link text, drop URL
    s = re.sub(r"`[^`]*`", "x", s)                    # inline code counts as one token
    s = re.sub(r"[#*>_-]+", " ", s)
    words += len(s.split())
print(words)
