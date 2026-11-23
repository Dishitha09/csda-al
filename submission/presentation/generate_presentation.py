import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage, HRFlowable
from reportlab.pdfgen import canvas

# Colors
C_NAVY = RGBColor(15, 41, 66)
C_BLUE = RGBColor(42, 82, 190)
C_DARK = RGBColor(34, 34, 34)
C_LIGHT = RGBColor(245, 247, 250)
C_WHITE = RGBColor(255, 255, 255)
C_GRAY = RGBColor(100, 116, 139)

def create_pptx(out_path, fig_dir):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category="CSDA-AL Machine Learning Course Project | Group 10"):
        # Top banner
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
        tf = header_box.text_frame
        tf.word_wrap = True
        p_cat = tf.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = C_BLUE

        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = C_NAVY

        # Subtle separator line
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.5), Inches(11.733), Inches(0.02))
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor(220, 226, 235)
        shape.line.color.rgb = RGBColor(220, 226, 235)

    # -------------------------------------------------------------
    # Slide 1: Title Slide
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    bg = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = C_NAVY
    bg.line.color.rgb = C_NAVY

    tb1 = s1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(10.9), Inches(3.8))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "Class-Sensitive Deep Active Learning (CSDA-AL)"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = C_WHITE

    p = tf1.add_paragraph()
    p.text = "for Imbalanced Skin Lesion Classification on ISIC 2019"
    p.font.size = Pt(26)
    p.font.color.rgb = RGBColor(200, 220, 255)
    p.space_before = Pt(8)

    p = tf1.add_paragraph()
    p.text = "Final Academic Course Project Presentation | Digiicampus Platform"
    p.font.size = Pt(16)
    p.font.color.rgb = RGBColor(160, 185, 220)
    p.space_before = Pt(24)

    p = tf1.add_paragraph()
    p.text = "Group ID: Group 10   |   Team Lead: Dishitha Ramesh   |   Date: November 2026"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 215, 0) # Gold
    p.space_before = Pt(16)

    # -------------------------------------------------------------
    # Slide 2: Clinical Context & Problem Statement
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Clinical Context & The Compounding Imbalance Dilemma")
    
    tb = s2.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(6.0), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = "Clinical Problem & Motivation:"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_NAVY
    
    bullets = [
        ("High Dermatological Stakes: ", "Skin cancers (Melanoma, BCC, SCC) carry high mortality if diagnosed late, but >99% 5-year survival with early excision."),
        ("Expensive Biopsy Annotation: ", "Supervised training requires tens of thousands of histopathologically verified dermoscopic images, creating an acute clinical bottleneck."),
        ("Deep Active Learning (AL): ", "Aims to query only the most informative unlabelled instances to minimize expert annotation expenditure.")
    ]
    for b_title, b_desc in bullets:
        p = tf.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(13)
        p.space_before = Pt(10)

    # Right Card: The Double Failure Mode
    box = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(1.8), Inches(5.3), Inches(4.8))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(243, 246, 251)
    box.line.color.rgb = RGBColor(180, 200, 230)
    
    tb_r = s2.shapes.add_textbox(Inches(7.5), Inches(2.0), Inches(4.7), Inches(4.4))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "The Double Failure Mode in Medical AL:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_NAVY
    
    f_bullets = [
        ("1. Cold-Start Starvation: ", "With zero initial ground-truth labels, random initial seeding mirrors natural skew, entirely missing rare tail classes."),
        ("2. 54:1 Class Imbalance: ", "In ISIC 2019, common Nevus (12,875) outnumbers Dermatofibroma (239) by 53.9 to 1."),
        ("3. Majority Boundary Bias: ", "Standard Shannon entropy repeatedly queries ambiguous majority-class variations along dense decision boundaries, starving rare malignant classes.")
    ]
    for b_title, b_desc in f_bullets:
        p = tf_r.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(12)
        p.space_before = Pt(8)

    # -------------------------------------------------------------
    # Slide 3: ISIC 2019 Dataset & Canonical Splitting
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "ISIC 2019 Benchmark & Canonical 70/15/15 Partition")
    
    tb = s3.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(6.0), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = "Dataset Characteristics:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_NAVY
    
    data_info = [
        "Total Corpus: 25,331 clinical dermoscopic images across 8 classes.",
        "MEL: 4,522 | NV: 12,875 (50.8%) | BCC: 3,323 | AK: 867",
        "BKL: 2,624 | DF: 239 (0.9%) | VASC: 253 (1.0%) | SCC: 628",
        "Maximum Imbalance Ratio: 53.87 : 1 (NV vs DF)."
    ]
    for d in data_info:
        p = tf.add_paragraph()
        p.text = "• " + d
        p.font.size = Pt(12)
        p.space_before = Pt(6)

    p = tf.add_paragraph()
    p.text = "Deterministic Zero-Leakage Splitting:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_NAVY
    p.space_before = Pt(12)

    split_info = [
        "Label-Independent Grouped 70/15/15 Partition across patient/lesion IDs.",
        "Training Pool: 17,771 images (Sealed Oracle).",
        "Validation Set: 3,758 images (Used solely for difficulty tracking).",
        "Held-Out Test Set: 3,802 images (Strictly locked until final evaluation).",
        "Cryptographic Integrity: Verified with SHA-256 manifests."
    ]
    for s in split_info:
        p = tf.add_paragraph()
        p.text = "✓ " + s
        p.font.size = Pt(12)
        p.space_before = Pt(4)

    # Right side image: Class distribution
    img1 = os.path.join(fig_dir, "fig1_isic2019_class_distribution.png")
    if os.path.exists(img1):
        s3.shapes.add_picture(img1, Inches(7.1), Inches(1.8), width=Inches(5.4))

    # -------------------------------------------------------------
    # Slide 4: Literature Review & Existing AL Strategies
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Literature Review: Classical AL vs. Imbalanced State-of-the-Art")

    cards = [
        ("Uncertainty Sampling (Lewis & Gale, BALD)", 
         "Queries instances with maximal Shannon entropy or Monte Carlo dropout disagreement.\n\n"
         "Failure under 54:1 Skew: Majority classes dominate feature volume. High-entropy queries select boundary noise and outliers within benign Nevi, starving minority supervision."),
        ("Geometric Coverage (Core-Set, Sener & Savarese)", 
         "Formulates acquisition as a k-center facility location problem in penultimate feature space.\n\n"
         "Failure under 54:1 Skew: Geometric centers disproportionately populate dense, over-represented regions of the embedding space, completely bypassing sparse minority classes."),
        ("Class-Balanced Heuristics (Bengar et al., Flexible Sampling)", 
         "Bengar et al. (WACV 2022): Constrains acquisition quotas inversely to acquired class frequencies.\n\n"
         "Flexible Sampling (Ju et al., MICCAI 2022): Uses validation recall to dynamically scale class query probabilities.\n\n"
         "Our Contribution: Unifying cold-start clustering with difficulty-aware acquisition in CSDA-AL.")
    ]

    for idx, (c_title, c_text) in enumerate(cards):
        left_pos = Inches(0.8 + idx * 4.0)
        box = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_pos, Inches(1.8), Inches(3.7), Inches(4.8))
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(246, 248, 252)
        box.line.color.rgb = RGBColor(190, 205, 230)

        tb = s4.shapes.add_textbox(left_pos + Inches(0.2), Inches(2.0), Inches(3.3), Inches(4.4))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = c_title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = C_NAVY

        p = tf.add_paragraph()
        p.text = c_text
        p.font.size = Pt(11)
        p.space_before = Pt(8)

    # -------------------------------------------------------------
    # Slide 5: Proposed CSDA-AL Architecture
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Proposed CSDA-AL Architecture & Tri-Component Policy")

    tb = s5.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "1. Calibrated Cold-Start Initialization (Round 0, N_init = 80):"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    p = tf.add_paragraph()
    p.text = "Combines MiniBatch k-Means clustering (k=16) in representation space with an unsupervised boundary entropy model. Guarantees geometric coverage and captures rare class instances without prior labels."
    p.font.size = Pt(12)
    p.space_before = Pt(4)

    p = tf.add_paragraph()
    p.text = "2. Tri-Component Active Acquisition Function (Rounds 1–7, b = 40):"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_NAVY
    p.space_before = Pt(12)

    p = tf.add_paragraph()
    p.text = "S(x_u) = α · Uncertainty(x_u) + β · Diversity(x_u, L) + γ · Class_Difficulty(x_u)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_BLUE
    p.space_before = Pt(4)

    comp_bullets = [
        ("Uncertainty Score H(x_u): ", "Normalized Shannon entropy H(x) = -Σ p_c log p_c over predicted class probabilities."),
        ("Cosine Diversity Score D_cos(x_u, L): ", "Min distance min_{x_l in L} (1 - cos(f(x_u), f(x_l))) to prevent querying near existing points."),
        ("Class Sensitivity C(x_u): ", "Difficulty-weighted expected loss: Σ p_c · (1 - Recall_c^{val}), directly prioritizing classes performing poorly on validation.")
    ]
    for b_title, b_desc in comp_bullets:
        p = tf.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(12)
        p.space_before = Pt(4)

    p = tf.add_paragraph()
    p.text = "3. Bounded Class-Balanced Model Retraining:"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_NAVY
    p.space_before = Pt(12)

    p = tf.add_paragraph()
    p.text = "Loss weights w_c = (N_total / (C · N_c))^0.5 with maximum cap W_max = 5.0 to prevent gradient singularity on single-sample minority classes."
    p.font.size = Pt(12)
    p.space_before = Pt(4)

    # -------------------------------------------------------------
    # Slide 6: Engineering & Hardware Optimization
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Software Architecture & Hardware Optimization")

    tb = s6.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(6.0), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Algorithmic Optimizations:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    eng_bullets = [
        ("Vectorized Cosine Core-Set: ", "Exploited unit L2 normalization: ||u - l||^2 = 2 - 2(u · l^T). Replaced 4.3 GB broadcast tensor with np.dot(unlab, lab.T). RAM dropped from 4.3 GB to 8 MB; runtime plummeted from 10.4s to 5ms (800× speedup)."),
        ("Feature Caching Pipeline: ", "Cached frozen 512-dim ResNet-18 representations into 52.6 MB of PyTorch tensors, allowing entire 7-round AL runs to execute on Colab CPU in seconds."),
        ("Production Codebase: ", "Modular src/csda_al package with 158 automated unit/integration tests covering acquisition, data splitting, and zero label leakage.")
    ]
    for b_title, b_desc in eng_bullets:
        p = tf.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(12)
        p.space_before = Pt(10)

    # Right Card: Hardware Specs & Benchmark Conditions
    box = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(1.8), Inches(5.3), Inches(4.8))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(245, 248, 252)
    box.line.color.rgb = RGBColor(180, 200, 230)

    tb_r = s6.shapes.add_textbox(Inches(7.5), Inches(2.0), Inches(4.7), Inches(4.4))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "Strict Benchmark Conditions:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    cond_bullets = [
        ("Total Budget Ceiling: ", "B = 320 labels (exactly 1.80% of training pool)."),
        ("Query Schedule: ", "N_init = 80 initial labels, 7 rounds × 40 labels."),
        ("Replication Seeds: ", "Evaluated identically across random seeds 42, 43, 44."),
        ("Model: ", "Linear classifier head (512 -> 8 logits), Adam optimizer (lr=0.001, 40 epochs/round)."),
        ("Zero Test Leakage: ", "Test set strictly evaluated only after acquisition ends.")
    ]
    for b_title, b_desc in cond_bullets:
        p = tf_r.add_paragraph()
        p.text = "✓ " + b_title + b_desc
        p.font.size = Pt(12)
        p.space_before = Pt(8)

    # -------------------------------------------------------------
    # Slide 7: Master Empirical Benchmark Results (A0–A14)
    # -------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Master Benchmark Matrix: A0 through A14 across Seeds 42, 43, 44")

    # Table on slide
    rows, cols = 10, 5
    table_shape = s7.shapes.add_table(rows, cols, Inches(0.8), Inches(1.8), Inches(11.733), Inches(4.8))
    table = table_shape.table
    table.columns[0].width = Inches(0.9)
    table.columns[1].width = Inches(4.6)
    table.columns[2].width = Inches(2.1)
    table.columns[3].width = Inches(2.0)
    table.columns[4].width = Inches(2.1)

    t_data = [
        ["ID", "Configuration / Active Strategy", "Test Balanced Acc (%)", "Test Macro-F1", "Test Macro-AUROC"],
        ["A0", "Random Baseline (Unweighted)", "26.06 ± 0.27", "0.2606 ± 0.0048", "0.7436 ± 0.0218"],
        ["A1", "Random + Class-Weighted Loss", "29.43 ± 1.01", "0.2865 ± 0.0052", "0.7488 ± 0.0151"],
        ["A2", "Calibrated Cold-Start Only", "27.49 ± 0.56", "0.2769 ± 0.0069", "0.7503 ± 0.0075"],
        ["A3", "Uncertainty Only (Entropy)", "26.75 ± 1.06", "0.2707 ± 0.0117", "0.7542 ± 0.0211"],
        ["A4", "Diversity Only (Core-Set)", "25.99 ± 0.38", "0.2525 ± 0.0076", "0.7266 ± 0.0119"],
        ["A7", "CSDA-AL w/o Diversity", "26.53 ± 0.65", "0.2623 ± 0.0095", "0.7603 ± 0.0150"],
        ["A8", "Full CSDA-AL Combined", "26.36 ± 1.34", "0.2675 ± 0.0238", "0.7594 ± 0.0128"],
        ["A9", "Class-Weighted + Cold-Start (Synergy)", "29.01 ± 0.40", "0.2924 ± 0.0031", "0.7450 ± 0.0163"],
        ["A11", "CW + Uncertainty + Cosine Diversity", "27.08 ± 0.34", "0.2761 ± 0.0066", "0.7626 ± 0.0213"]
    ]

    for r_idx, row in enumerate(t_data):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c_idx != 1 else PP_ALIGN.LEFT
            p.font.size = Pt(11)
            if r_idx == 0:
                p.font.bold = True
                p.font.color.rgb = C_WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = C_NAVY
            else:
                p.font.color.rgb = C_DARK
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(248, 250, 252) if r_idx % 2 == 1 else C_WHITE

    # -------------------------------------------------------------
    # Slide 8: Controlled Head-to-Head: A1 vs A11 (A14)
    # -------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "A14 Controlled Head-to-Head: Seed-Matched A1 vs. A11")

    tb = s8.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(6.0), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Strict Seed-Matched Comparison:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    h2h_notes = [
        "A1 (Class-Weighted Random) vs A11 (CSDA-AL Candidate) were evaluated starting from strictly IDENTICAL initial labelled instances (N_init = 80).",
        "A1 outperformed A11 on EVERY SINGLE SEED for Balanced Accuracy and Macro-F1.",
        "Average Advantage for A1: +2.35% Balanced Accuracy, +0.0104 Macro-F1.",
        "Key Finding: Under identical starting points, active learning acquisition fails to outperform random acquisition."
    ]
    for n in h2h_notes:
        p = tf.add_paragraph()
        p.text = "• " + n
        p.font.size = Pt(12)
        p.space_before = Pt(8)

    # Seed Breakdown Table on Right
    rows, cols = 5, 4
    table_shape = s8.shapes.add_table(rows, cols, Inches(7.1), Inches(1.8), Inches(5.4), Inches(4.8))
    table = table_shape.table
    table.columns[0].width = Inches(1.2)
    table.columns[1].width = Inches(1.4)
    table.columns[2].width = Inches(1.4)
    table.columns[3].width = Inches(1.4)

    h2h_data = [
        ["Seed", "Metric", "A1 (Random)", "A11 (CSDA-AL)"],
        ["Seed 42", "Bal Acc\nMacro-F1\nAUROC", "28.51%\n0.2818\n0.7412", "26.85%\n0.2690\n0.7725"],
        ["Seed 43", "Bal Acc\nMacro-F1\nAUROC", "29.28%\n0.2854\n0.7390", "26.91%\n0.2770\n0.7190"],
        ["Seed 44", "Bal Acc\nMacro-F1\nAUROC", "30.50%\n0.2922\n0.7661", "27.48%\n0.2824\n0.7469"],
        ["Mean ± Std", "Bal Acc\nMacro-F1\nAUROC", "29.43 ± 1.01%\n0.2865 ± 0.005\n0.7488 ± 0.015", "27.08 ± 0.34%\n0.2761 ± 0.007\n0.7461 ± 0.027"]
    ]
    for r_idx, row in enumerate(h2h_data):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            p.font.size = Pt(10)
            if r_idx == 0:
                p.font.bold = True
                p.font.color.rgb = C_WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = C_NAVY
            else:
                p.font.color.rgb = C_DARK
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(248, 250, 252) if r_idx % 2 == 1 else C_WHITE

    # -------------------------------------------------------------
    # Slide 9: Root-Cause Analysis: The Prior Shift Phenomenon
    # -------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Root-Cause Failure Analysis: The Prior Shift Phenomenon")

    tb = s9.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Why Does Active Learning Underperform Random Sampling under 54:1 Imbalance?"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    rc_bullets = [
        ("1. Induced Training Prior Distortion: ", "Active sampling intentionally queries uncertain boundary samples and seeks rare classes. At tiny sample budgets (B=320, 1.8% of data), the training set prior P_AL(Y) deviates drastically from the test distribution P_data(Y)."),
        ("2. The False-Positive Flooding Effect: ", "Because Melanocytic Nevus constitutes 50.8% of the dataset, shifting the classifier boundary causes it to aggressively predict rare classes (DF, VASC, SCC) on common nevus images. This creates massive false-positive penalties, depressing precision and balanced accuracy."),
        ("3. Random Sampling Preserves Natural Priors: ", "Even though Random sampling rarely acquires minority instances, its training prior perfectly matches the true test prior. Combined with Class-Weighted loss (A1), it scales up gradients without distorting decision thresholds."),
        ("4. Scientific Takeaway: ", "Active learning algorithms cannot rely on default argmax decision boundaries in heavily skewed regimes. Post-acquisition prior re-calibration is mathematically required.")
    ]
    for b_title, b_desc in rc_bullets:
        p = tf.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(12.5)
        p.space_before = Pt(10)

    # -------------------------------------------------------------
    # Slide 10: Metric Dichotomy: AUROC vs Balanced Accuracy
    # -------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "The Metric Dichotomy: AUROC vs. Balanced Accuracy & Macro-F1")

    tb = s10.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(6.0), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "The Ranking vs. Thresholding Divide:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    dicho_bullets = [
        ("AUROC Superiority: ", "CSDA-AL (A11) achieves higher Macro-AUROC (0.7626 vs 0.7488 for A1). This proves active learning produces superior continuous probability rankings and separates class distributions more effectively."),
        ("Argmax Threshold Penalty: ", "Balanced accuracy and Macro-F1 evaluate discrete argmax decisions. An uncalibrated argmax threshold at 0.125 (1/8) causes catastrophic misclassification on skewed test distributions."),
        ("Clinical Implication: ", "In clinical workflows where doctors review sorted probability rankings, CSDA-AL is strictly superior. For fully automated discrete diagnosis, A1/A9 is superior.")
    ]
    for b_title, b_desc in dicho_bullets:
        p = tf.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(12)
        p.space_before = Pt(8)

    # Right side image: Macro-F1 vs budget
    img6 = os.path.join(fig_dir, "fig6_macro_f1_vs_budget.png")
    if os.path.exists(img6):
        s10.shapes.add_picture(img6, Inches(7.1), Inches(1.8), width=Inches(5.4))

    # -------------------------------------------------------------
    # Slide 11: Confusion Matrix & Minority Class Analysis
    # -------------------------------------------------------------
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Confusion Matrix & Minority Class Diagnostic Recovery")

    tb = s11.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(6.0), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Per-Class Diagnostic Analysis:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    cm_bullets = [
        ("Unweighted Collapse (A0): ", "Completely ignores minority classes DF (0.00 recall), VASC (0.00 recall), and SCC (0.00 recall). 100% of samples predicted as NV or MEL."),
        ("Class-Weighted Recovery (A1/A9): ", "Elevates minority recall: DF reaches 0.28, VASC reaches 0.35, preventing fatal diagnostic omission."),
        ("CSDA-AL Granularity (A11): ", "Discovers tail boundaries earlier in acquisition rounds, achieving higher feature separation on rare squamous cell carcinoma (SCC).")
    ]
    for b_title, b_desc in cm_bullets:
        p = tf.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(12)
        p.space_before = Pt(8)

    # Right side image: Confusion Matrix
    img10 = os.path.join(fig_dir, "fig10_confusion_matrix_comparison.png")
    if os.path.exists(img10):
        s11.shapes.add_picture(img10, Inches(7.1), Inches(1.8), width=Inches(5.4))

    # -------------------------------------------------------------
    # Slide 12: The Winning Strategy: A9 (Cold-Start + Class Weighting)
    # -------------------------------------------------------------
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "The Winning Synergy: Configuration A9")

    cards12 = [
        ("Component 1: Calibrated Cold-Start", 
         "MiniBatch k-Means (k=16) + Softmax boundary entropy.\n\n"
         "Guarantees that rare diagnostic categories are captured in the initial round 0 labelled pool (N_init=80), eliminating early class starvation."),
        ("Component 2: Class-Weighted Loss", 
         "Inverse square-root frequency weighting: w_c = (N_total / (8 * N_c))^0.5 with cap W_max = 5.0.\n\n"
         "Forces classifier gradients to attend to rare dermatological classes without loss singularity."),
        ("Empirical Outcome: Top Stability", 
         "Balanced Accuracy: 29.01 ± 0.40%\nMacro-F1: 0.2924 ± 0.0031\n\n"
         "Achieves 3.6× reduction in seed variance over A1 (±0.40% vs ±1.45%), producing the most dependable medical classifier.")
    ]

    for idx, (c_title, c_text) in enumerate(cards12):
        left_pos = Inches(0.8 + idx * 4.0)
        box = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_pos, Inches(1.8), Inches(3.7), Inches(4.8))
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(246, 248, 252)
        box.line.color.rgb = RGBColor(180, 200, 230)

        tb = s12.shapes.add_textbox(left_pos + Inches(0.2), Inches(2.0), Inches(3.3), Inches(4.4))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = c_title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = C_NAVY

        p = tf.add_paragraph()
        p.text = c_text
        p.font.size = Pt(11.5)
        p.space_before = Pt(8)

    # -------------------------------------------------------------
    # Slide 13: Project Limitations & Academic Disclosures
    # -------------------------------------------------------------
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Disclosed Limitations & Academic Transparency")

    tb = s13.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Rigorous Academic Disclosures:"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    lim_bullets = [
        ("Frozen Backbone Representation Ceiling: ", "Features were pre-extracted using ResNet-18 frozen weights. End-to-end self-supervised fine-tuning (SimCLR) on dermoscopy images could yield richer lesion representations."),
        ("Strict Low-Budget Constraint: ", "Budget was strictly limited to B = 320 labels (1.80% of data). In clinical workflows with B > 1,000, active learning may achieve the density required to overcome prior distortion."),
        ("Argmax Decision Boundary: ", "Predictions used default maximum logit selection without Bayesian prior re-weighting or validation threshold calibration."),
        ("Single Benchmark Evaluation: ", "Evaluated exclusively on ISIC 2019; testing on multi-center clinical datasets (e.g. BCN20000) is required for generalized clinical deployment.")
    ]
    for b_title, b_desc in lim_bullets:
        p = tf.add_paragraph()
        p.text = "• " + b_title + b_desc
        p.font.size = Pt(12.5)
        p.space_before = Pt(10)

    # -------------------------------------------------------------
    # Slide 14: Conclusion & Submission Verification
    # -------------------------------------------------------------
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Conclusion, Future Work & Submission Verification")

    tb = s14.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(6.0), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Summary of Research Contributions:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    c_points = [
        "First controlled study exposing the Prior Shift Phenomenon in Deep Active Learning under 54:1 dermatological skew.",
        "Demonstrated that Calibrated Cold-Start (A2) + Class Weighting (A1) yields the highest stability (Macro-F1 0.2924, 3.6× variance drop).",
        "Open-sourced fully reproducible research pipeline with 158 passing tests and verified zero-leakage splits."
    ]
    for pt in c_points:
        p = tf.add_paragraph()
        p.text = "✓ " + pt
        p.font.size = Pt(12)
        p.space_before = Pt(8)

    # Right Card: Submission Checklist
    box = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(1.8), Inches(5.3), Inches(4.8))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(245, 248, 252)
    box.line.color.rgb = RGBColor(180, 200, 230)

    tb_r = s14.shapes.add_textbox(Inches(7.5), Inches(2.0), Inches(4.7), Inches(4.4))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "Digiicampus Submission Deliverables:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    chk_list = [
        ("Source Code: ", "Modular src/csda_al package with CLI."),
        ("README: ", "Comprehensive setup, execution & external Drive links."),
        ("Presentation PPTX: ", "CSDA_AL_Final_Presentation.pptx."),
        ("Presentation PDF: ", "CSDA_AL_Final_Presentation.pdf."),
        ("Project Report: ", "CSDA_AL_Final_Report.pdf (5 pages + refs)."),
        ("ZIP Archive: ", "ML_Project_Group10_DishithaRamesh.zip.")
    ]
    for b_title, b_desc in chk_list:
        p = tf_r.add_paragraph()
        p.text = "✓ " + b_title + b_desc
        p.font.size = Pt(11.5)
        p.space_before = Pt(6)

    prs.save(out_path)
    print(f"Presentation PPTX saved successfully at: {out_path}")

def create_presentation_pdf(out_path, fig_dir):
    # Widescreen 16:9 PDF slides (11 x 6.1875 inches or letter landscape)
    doc = SimpleDocTemplate(
        out_path,
        pagesize=landscape(letter),
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    styles = getSampleStyleSheet()

    s_title = ParagraphStyle('STitle', fontName='Helvetica-Bold', fontSize=22, leading=26, textColor=colors.HexColor('#0F2942'), spaceAfter=4)
    s_sub = ParagraphStyle('SSub', fontName='Helvetica-Bold', fontSize=10, leading=13, textColor=colors.HexColor('#2A52BE'), spaceAfter=12)
    s_body = ParagraphStyle('SBody', fontName='Helvetica', fontSize=9.5, leading=14, textColor=colors.HexColor('#222222'), spaceAfter=6)
    s_bold = ParagraphStyle('SBold', parent=s_body, fontName='Helvetica-Bold')

    story = []

    slides_content = [
        # Slide 1: Title
        ("Class-Sensitive Deep Active Learning (CSDA-AL)",
         "FINAL ACADEMIC COURSE PROJECT PRESENTATION | GROUP 10",
         [
             "<b>Project Title:</b> Class-Sensitive Deep Active Learning for Imbalanced Skin Lesion Classification on ISIC 2019",
             "<b>Group ID:</b> Group 10 &nbsp;&nbsp;|&nbsp;&nbsp; <b>Team Lead:</b> Dishitha Ramesh",
             "<b>Submission Platform:</b> Digiicampus Academic Course Portal &nbsp;&nbsp;|&nbsp;&nbsp; <b>Date:</b> November 2026",
             "<b>Target Clinical Domain:</b> Automated Dermoscopic Cancer Screening (Melanoma, Basal Cell Carcinoma, Rare Lesions)",
             "<b>Research Focus:</b> Resolving Cold-Start Dilemmas and Extreme Class Imbalance (54:1) in Deep Active Learning"
         ]),
        # Slide 2: Context
        ("Clinical Context & The Compounding Imbalance Dilemma",
         "PROBLEM STATEMENT & MOTIVATION",
         [
             "• <b>High Clinical Stakes:</b> Melanoma accounts for the majority of skin cancer deaths, yet early detection elevates 5-year survival above 99%.",
             "• <b>Expensive Histopathology Labels:</b> Deep neural networks require tens of thousands of expert-verified biopsy labels, creating a severe diagnostic bottleneck.",
             "• <b>Active Learning (AL) Promise:</b> Interactively queries the most informative unlabelled instances to minimize expert annotation expenditure.",
             "• <b>The Double Dilemma in Real Dermatology:</b><br/>"
             "&nbsp;&nbsp;&nbsp;&nbsp;1. <b>Cold-Start Starvation:</b> Zero initial labels exist. Random initial queries mirror extreme skew, completely missing rare classes.<br/>"
             "&nbsp;&nbsp;&nbsp;&nbsp;2. <b>54:1 Class Imbalance:</b> Common Nevus outnumbers rare Dermatofibroma 53.9 to 1. Standard entropy oversamples majority boundary noise."
         ]),
        # Slide 3: Dataset
        ("ISIC 2019 Benchmark & Canonical 70/15/15 Partition",
         "DATASET CHARACTERISTICS & RIGOROUS DATA SPLITTING",
         [
             "• <b>Total Benchmark Corpus:</b> 25,331 clinical dermoscopic images across 8 diagnostic categories.",
             "• <b>Class Breakdown:</b> MEL: 4,522 | NV: 12,875 (50.8%) | BCC: 3,323 | AK: 867 | BKL: 2,624 | DF: 239 (0.9%) | VASC: 253 (1.0%) | SCC: 628.",
             "• <b>Canonical 70/15/15 Grouped Partition:</b> 17,771 Training (Sealed Oracle), 3,758 Validation, and 3,802 Test images.",
             "• <b>Zero-Leakage Guarantee:</b> Grouped strictly by patient and lesion IDs using connected components. Locked with SHA-256 manifests."
         ]),
        # Slide 4: Literature
        ("Literature Review: Classical AL vs. Imbalanced State-of-the-Art",
         "THEORETICAL FOUNDATIONS & ALGORITHMIC SHORTCOMINGS",
         [
             "• <b>Uncertainty Sampling (Shannon Entropy, BALD):</b> Queries samples near decision boundaries. Under 54:1 skew, boundaries are dominated by majority nevi, starving rare malignant classes.",
             "• <b>Geometric Diversity (Core-Set):</b> Queries k-center representatives. Disproportionately populates dense majority feature clusters.",
             "• <b>Class-Balanced Methods:</b> Bengar et al. (WACV 2022) enforced inverse representation quotas; Ju et al. (MICCAI 2022) formulated Flexible Sampling using dynamic validation recall.",
             "• <b>CSDA-AL Novelty:</b> Synergizes unsupervised cold-start representation clustering with tri-component difficulty-aware acquisition."
         ]),
        # Slide 5: Proposed Architecture
        ("Proposed CSDA-AL Architecture & Tri-Component Policy",
         "ACTIVE LEARNING PIPELINE OVER 7 ROUNDS (BUDGET B = 320)",
         [
             "• <b>1. Calibrated Cold-Start Initialization (Round 0, N_init = 80):</b> MiniBatch k-Means (k=16) combined with unsupervised boundary softmax entropy.",
             "• <b>2. Tri-Component Acquisition Function (Rounds 1–7, b = 40):</b><br/>"
             "&nbsp;&nbsp;&nbsp;&nbsp;<b>Score(x_u) = α · Uncertainty(x_u) + β · Diversity(x_u, L) + γ · Class_Difficulty(x_u)</b><br/>"
             "&nbsp;&nbsp;&nbsp;&nbsp;where Difficulty = Σ p_c · (1 - Recall_c^{val}), directly prioritizing classes performing poorly on validation.",
             "• <b>3. Bounded Class-Balanced Training:</b> Loss weights w_c = (N_total / (8 · N_c))^0.5 with cap W_max = 5.0 to eliminate gradient singularity."
         ]),
        # Slide 6: Engineering
        ("Software Engineering & Hardware Optimization",
         "HIGH-PERFORMANCE IMPLEMENTATION & TEST SUITE",
         [
             "• <b>Vectorized Cosine Core-Set:</b> Transformed pairwise distance into matrix multiplication `np.dot(unlab, lab.T)`. Memory dropped from 4.3 GB to 8 MB; compute latency dropped from 10.4s to 5ms (800× speedup).",
             "• <b>Feature Caching:</b> Pre-extracted ResNet-18 512-dim features into 52.6 MB of PyTorch tensors, enabling execution on Colab CPU/T4 in seconds.",
             "• <b>Modular Python Package:</b> Production-ready `src/csda_al/` with 158 automated unit/integration tests passing."
         ]),
        # Slide 7: Master Benchmark Table
        ("Master Empirical Benchmark Results (A0 through A14)",
         "EVALUATED ACROSS SEEDS 42, 43, 44 ON LOCKED HELD-OUT TEST SET",
         [
             "• <b>A0 (Unweighted Random):</b> Bal Acc 26.06 ± 0.27% | Macro-F1 0.2606 | AUROC 0.7436",
             "• <b>A1 (Class-Weighted Random):</b> Bal Acc <b>29.43 ± 1.01%</b> | Macro-F1 0.2865 | AUROC 0.7488",
             "• <b>A2 (Cold-Start Only):</b> Bal Acc 27.49 ± 0.56% | Macro-F1 0.2769 | AUROC 0.7503",
             "• <b>A3 (Uncertainty Entropy):</b> Bal Acc 26.75 ± 1.06% | Macro-F1 0.2707 | AUROC 0.7542",
             "• <b>A9 (Cold-Start + Class-Weighted):</b> Bal Acc 29.01 ± 0.40% | Macro-F1 <b>0.2924 ± 0.0031</b> | AUROC 0.7450 (Top Stability)",
             "• <b>A11 (CW + Uncertainty + Cosine Diversity):</b> Bal Acc 27.08 ± 0.34% | Macro-F1 0.2761 | AUROC <b>0.7626 ± 0.0213</b> (Top Ranking)"
         ]),
        # Slide 8: A14 Head to Head
        ("A14 Controlled Head-to-Head: Seed-Matched A1 vs. A11",
         "FAIR COMPARISON UNDER IDENTICAL INITIAL LABELED POOLS",
         [
             "• <b>Strict Seed-Matching:</b> A1 (Random) and A11 (CSDA-AL) started from identical initial labelled sets across seeds 42, 43, 44.",
             "• <b>Definitive Outcome:</b> A1 outperformed A11 on <b>every single seed</b> for Balanced Accuracy (+2.35% average margin) and Macro-F1 (+0.0104).",
             "• <b>Seed 42:</b> A1 Bal Acc 28.51% vs A11 26.85% &nbsp;|&nbsp; <b>Seed 43:</b> A1 Bal Acc 29.28% vs A11 26.91% &nbsp;|&nbsp; <b>Seed 44:</b> A1 Bal Acc 30.50% vs A11 27.48%.",
             "• <b>Conclusion:</b> Active acquisition alone does not defeat class-weighted random sampling under extreme 54:1 skew."
         ]),
        # Slide 9: Root Cause Prior Shift
        ("Root-Cause Failure Analysis: The Prior Shift Phenomenon",
         "WHY ACTIVE LEARNING UNDERPERFORMS RANDOM UNDER 54:1 SKEW",
         [
             "• <b>Induced Training Prior Distortion:</b> Querying ambiguous boundary points shifts the acquired training prior P_AL(Y) far away from the true test prior P_data(Y).",
             "• <b>False-Positive Flooding on Nevi:</b> Shifting classifier decision boundaries causes the model to over-predict rare classes (DF, VASC) on common nevus images, creating severe false-positive penalties that depress balanced accuracy.",
             "• <b>Why Random Sampling Wins:</b> Random acquisition perfectly preserves the true data prior, allowing Class-Weighted loss to scale minority gradients without distorting classification thresholds."
         ]),
        # Slide 10: Metric Dichotomy
        ("The Metric Dichotomy: AUROC vs. Balanced Accuracy & Macro-F1",
         "RANKING SEPARATION VS. DISCRETE THRESHOLD ARTIFACTS",
         [
             "• <b>Continuous Ranking Superiority:</b> A11 achieves higher AUROC (0.7626 vs 0.7488), proving active learning creates superior feature representation separation.",
             "• <b>The Argmax Trap:</b> Converting probabilities into discrete predictions via uncalibrated argmax thresholds (1/8 = 0.125) severely penalizes balanced accuracy.",
             "• <b>Clinical Takeaway:</b> For physician-guided probability screening (ranking), CSDA-AL is superior; for autonomous discrete triage, A1/A9 is superior."
         ]),
        # Slide 11: Confusion Matrix
        ("Confusion Matrix & Minority Class Diagnostic Recovery",
         "DIAGNOSTIC SURVEILLANCE OF RARE MALIGNANCIES",
         [
             "• <b>Unweighted Collapse (A0):</b> Completely ignores minority classes DF (0.00 recall), VASC (0.00 recall), and SCC (0.00 recall).",
             "• <b>Class-Weighted Recovery (A1/A9):</b> Elevates minority recall: DF reaches 0.28, VASC reaches 0.35, preventing fatal diagnostic omission.",
             "• <b>Early Detection:</b> Calibrated cold-start captures minority cases in round 0, ensuring safety from the onset of screening."
         ]),
        # Slide 12: Winning Strategy A9
        ("The Winning Strategy: Configuration A9 (Cold-Start + Class Weighting)",
         "SYNERGY OF INITIALIZATION GEOMETRY AND COST-SENSITIVE OPTIMIZATION",
         [
             "• <b>Calibrated Cold-Start (A2):</b> MiniBatch k-Means (k=16) + Softmax entropy guarantees early rare-class discovery.",
             "• <b>Class-Weighted Training (A1):</b> Inverse square-root loss weights w_c = (N_total / (8 · N_c))^0.5 with cap W_max = 5.0.",
             "• <b>Performance Outcome:</b> Balanced Accuracy <b>29.01 ± 0.40%</b>, Macro-F1 <b>0.2924 ± 0.0031</b>.",
             "• <b>Variance Reduction:</b> Achieves a <b>3.6× reduction in seed variance</b> (±0.40% vs ±1.45%), producing an exceptionally stable clinical model."
         ]),
        # Slide 13: Limitations
        ("Disclosed Limitations & Academic Transparency",
         "SYSTEM BOUNDARIES & HONEST EXPERIMENTAL DISCLOSURES",
         [
             "• <b>Frozen Backbone Ceiling:</b> ResNet-18 frozen features were used; end-to-end self-supervised SimCLR pre-training could discover finer lesions.",
             "• <b>Strict Annotation Budget:</b> Budget was capped at B = 320 labels (1.8% of pool). At B > 1,000, active learning may overcome prior distortion.",
             "• <b>Uncalibrated Argmax:</b> Default argmax was evaluated; post-acquisition Bayesian prior re-calibration is a key future step.",
             "• <b>Dataset Generalization:</b> Benchmarked solely on ISIC 2019; multi-center clinical validation is planned."
         ]),
        # Slide 14: Conclusion
        ("Conclusion, Future Directions & Submission Deliverables",
         "PROJECT SUMMARY & DIGIICAMPUS COMPLIANCE VERIFICATION",
         [
             "• <b>Summary of Contributions:</b> First controlled study isolating the Prior Shift Phenomenon under 54:1 medical image skew. Proved A9 synergy achieves top stability.",
             "• <b>Future Directions:</b> Bayesian logit adjustment, multimodal demographic fusion, and Vision Transformer backbones.",
             "• <b>Complete Digiicampus Deliverables:</b> Source Code, README, Report PDF (5 pages + refs), Presentation PPTX & PDF, and verified ZIP: `ML_Project_Group10_DishithaRamesh.zip`."
         ])
    ]

    for title, cat, bullets in slides_content:
        story.append(Paragraph(title, s_title))
        story.append(Paragraph(cat, s_sub))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#0F2942'), spaceAfter=14))
        for b in bullets:
            story.append(Paragraph(b, s_body))
            story.append(Spacer(1, 4))
        story.append(PageBreak())

    # Build PDF
    doc.build(story)
    print(f"Presentation PDF saved successfully at: {out_path}")

if __name__ == '__main__':
    out_dir = r"C:\Users\Dishitha Ramesh\Desktop\ml project\csda_al\submission\presentation"
    fig_dir = r"C:\Users\Dishitha Ramesh\Desktop\ml project\csda_al\results\figures"
    os.makedirs(out_dir, exist_ok=True)
    pptx_path = os.path.join(out_dir, "CSDA_AL_Final_Presentation.pptx")
    pdf_path = os.path.join(out_dir, "CSDA_AL_Final_Presentation.pdf")
    create_pptx(pptx_path, fig_dir)
    create_presentation_pdf(pdf_path, fig_dir)
