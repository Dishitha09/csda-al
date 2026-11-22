import re

with open(r"C:\Users\Dishitha Ramesh\Desktop\ml project\csda_al\docs\MASTER_PROMPT_EXTRACTED.md", "r", encoding="utf-8") as f:
    text = f.read()

keywords = ["Bengar", "stopping", "transfer", "Khanal", "Flexible", "SUPClust"]
for kw in keywords:
    matches = [m.start() for m in re.finditer(re.escape(kw), text, re.IGNORECASE)]
    print(f"Keyword '{kw}': {len(matches)} occurrences")
    for pos in matches[:3]:
        snippet = text[max(0, pos-100):min(len(text), pos+200)].replace("\n", " ")
        print(f"  ... {snippet} ...\n")
