"""Package the entire completed CSDA-AL project into a clean distribution ZIP."""
import shutil
import zipfile
from pathlib import Path

def main():
    root = Path(r"C:\Users\Dishitha Ramesh\Desktop\ml project\csda_al")
    out_zip = Path(r"C:\Users\Dishitha Ramesh\Desktop\ml project\CSDA_AL_Final_Completed_Project.zip")
    downloads_zip = Path(r"C:\Users\Dishitha Ramesh\Downloads\ML FINAL TRIAL\CSDA_AL_Final_Completed_Project.zip")

    include_dirs = ["src", "experiments", "docs", "results", "tests", "scripts"]
    include_files = ["pyproject.toml", "README.md", "README_VALIDATION.md"]

    print("Creating final project archive...")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f_name in include_files:
            p = root / f_name
            if p.exists():
                zipf.write(p, arcname=f_name)
                print(f"  Added {f_name}")

        for d_name in include_dirs:
            d_path = root / d_name
            if not d_path.exists():
                continue
            for file_path in d_path.rglob("*"):
                if file_path.is_file():
                    if any(part in file_path.parts for part in ["__pycache__", ".pytest_cache", ".git"]):
                        continue
                    if file_path.suffix in [".pyc", ".pyo"]:
                        continue
                    arcname = file_path.relative_to(root)
                    zipf.write(file_path, arcname=str(arcname))

    print(f"Archive created: {out_zip} ({out_zip.stat().st_size / (1024*1024):.2f} MB)")
    shutil.copyfile(out_zip, downloads_zip)
    print(f"Copied archive to {downloads_zip}")

if __name__ == "__main__":
    main()
