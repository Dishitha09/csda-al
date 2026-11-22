import zipfile
import os

zip_path = r"C:\Users\Dishitha Ramesh\Desktop\ML FINAL TRIAL\ISIC2019_Antigravity_Starter_Bundle (1).zip"
with zipfile.ZipFile(zip_path) as z:
    names = z.namelist()
    print("Files in starter bundle:", names)
    master_prompt = z.read("ANTIGRAVITY_MASTER_PROMPT_ISIC2019.md").decode("utf-8")
    with open(r"C:\Users\Dishitha Ramesh\Desktop\ml project\csda_al\docs\MASTER_PROMPT_EXTRACTED.md", "w", encoding="utf-8") as f:
        f.write(master_prompt)
    print("Saved MASTER_PROMPT_EXTRACTED.md, length:", len(master_prompt))
