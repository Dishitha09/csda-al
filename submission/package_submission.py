import os
import shutil
import zipfile

repo_root = r"C:\Users\Dishitha Ramesh\Desktop\ml project\csda_al"
target_dir_name = "ML_Project_Group10_DishithaRamesh"
dest_root = os.path.join(r"C:\Users\Dishitha Ramesh\Desktop\ml project", target_dir_name)
zip_output_path = os.path.join(r"C:\Users\Dishitha Ramesh\Desktop\ml project", f"{target_dir_name}.zip")
downloads_zip_path = os.path.join(r"C:\Users\Dishitha Ramesh\Downloads\ML FINAL TRIAL", f"{target_dir_name}.zip")

# 1. Clean existing target dir if present
if os.path.exists(dest_root):
    shutil.rmtree(dest_root)
os.makedirs(dest_root, exist_ok=True)

# 2. Top-level README
shutil.copy2(os.path.join(repo_root, "README.md"), os.path.join(dest_root, "README.md"))

# 3. Source_Code
src_code_dir = os.path.join(dest_root, "Source_Code")
os.makedirs(src_code_dir, exist_ok=True)

# Copy src/
shutil.copytree(
    os.path.join(repo_root, "src"),
    os.path.join(src_code_dir, "src"),
    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
)

# Copy scripts/
shutil.copytree(
    os.path.join(repo_root, "scripts"),
    os.path.join(src_code_dir, "scripts"),
    ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
)

# Copy configs/
shutil.copytree(
    os.path.join(repo_root, "configs"),
    os.path.join(src_code_dir, "configs")
)

# Copy tests/
shutil.copytree(
    os.path.join(repo_root, "tests"),
    os.path.join(src_code_dir, "tests"),
    ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
)

# Copy notebooks/
shutil.copytree(
    os.path.join(repo_root, "notebooks"),
    os.path.join(src_code_dir, "notebooks"),
    ignore=shutil.ignore_patterns(".ipynb_checkpoints")
)

# Copy metadata files
for f in ["requirements.txt", "pyproject.toml", "run_isic_pipeline.py"]:
    src_f = os.path.join(repo_root, f)
    if os.path.exists(src_f):
        shutil.copy2(src_f, os.path.join(src_code_dir, f))

# 4. Presentation
pres_dir = os.path.join(dest_root, "Presentation")
os.makedirs(pres_dir, exist_ok=True)
shutil.copy2(
    os.path.join(repo_root, "submission", "presentation", "CSDA_AL_Final_Presentation.pptx"),
    os.path.join(pres_dir, "CSDA_AL_Final_Presentation.pptx")
)
shutil.copy2(
    os.path.join(repo_root, "submission", "presentation", "CSDA_AL_Final_Presentation.pdf"),
    os.path.join(pres_dir, "CSDA_AL_Final_Presentation.pdf")
)

# 5. Project_Report
rep_dir = os.path.join(dest_root, "Project_Report")
os.makedirs(rep_dir, exist_ok=True)
shutil.copy2(
    os.path.join(repo_root, "submission", "report", "CSDA_AL_Final_Report.pdf"),
    os.path.join(rep_dir, "CSDA_AL_Final_Report.pdf")
)

# 6. Documentation
doc_dir = os.path.join(dest_root, "Documentation")
os.makedirs(doc_dir, exist_ok=True)
shutil.copy2(
    os.path.join(repo_root, "submission", "audit", "PROJECT_AUDIT.md"),
    os.path.join(doc_dir, "PROJECT_AUDIT.md")
)
for doc_name, target_name in [
    ("RESEARCH_METHODOLOGY.md", "methodology.md"),
    ("MODULE_GUIDE.md", "module_guide.md"),
    ("EXPERIMENTAL_PROTOCOL.md", "experiment_protocol.md"),
    ("LIMITATIONS.md", "limitations.md")
]:
    src_doc = os.path.join(repo_root, "docs", doc_name)
    if os.path.exists(src_doc):
        shutil.copy2(src_doc, os.path.join(doc_dir, target_name))

# 7. Results (Summary tables and figures, no heavy raw data)
res_dir = os.path.join(dest_root, "Results")
os.makedirs(res_dir, exist_ok=True)
shutil.copytree(
    os.path.join(repo_root, "results", "tables"),
    os.path.join(res_dir, "tables")
)
shutil.copytree(
    os.path.join(repo_root, "results", "figures"),
    os.path.join(res_dir, "figures")
)

print("Folder structure assembled successfully at:", dest_root)

# 8. Create ZIP archive
if os.path.exists(zip_output_path):
    os.remove(zip_output_path)

print("Creating ZIP file:", zip_output_path)
with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(dest_root):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, os.path.dirname(dest_root))
            zf.write(full_path, rel_path)

# Also copy to Downloads folder for easy user access
shutil.copy2(zip_output_path, downloads_zip_path)

print(f"ZIP package created successfully!")
print(f"Size: {os.path.getsize(zip_output_path) / (1024 * 1024):.2f} MB")
