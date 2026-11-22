import zipfile

zip_path = r"C:\Users\Dishitha Ramesh\Desktop\ML FINAL TRIAL\ISIC2019_Antigravity_Starter_Bundle (1).zip"
with zipfile.ZipFile(zip_path) as z:
    for name in ['README_START_HERE.txt', 'INPUT_AUDIT_NOTES.md', 'ANTIGRAVITY_QUICK_LAUNCH.txt']:
        content = z.read(name).decode("utf-8")
        out_path = rf"C:\Users\Dishitha Ramesh\Desktop\ml project\csda_al\docs\{name}"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Extracted {name}")
