# -*- coding: utf-8 -*-
"""
Application Web Streamlit pour la Vérification de Raccord de Motif Textile.
Version : Mitwill "Dark Glass" Edition
Dépendances : pip install reportlab streamlit pillow numpy pandas
"""
import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageEnhance, ImageFilter
import numpy as np
import pandas as pd
from io import BytesIO
import os
import uuid
import datetime
import json

# --- CONFIGURATION INITIALE & THEME ---
st.set_page_config(
    page_title="Mitwill Seamless Check",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Protection contre les bombes de décompression
Image.MAX_IMAGE_PIXELS = None 

# --- IMPORT OPTIONNEL PDF ---
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# --- CSS PERSONNALISÉ (STYLE DARK GLASSMORPHISM) ---
st.markdown("""
<style>
    /* VARIABLES DARK MODE */
    :root {
        --mitwill-orange: rgb(250, 125, 80);
        --mitwill-orange-glow: rgba(250, 125, 80, 0.4);
        --glass-bg: rgba(20, 20, 25, 0.75); /* Fond sombre vitré */
        --glass-border: rgba(255, 255, 255, 0.08); /* Bordure subtile */
        --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        --text-color: #f5f5f7; /* Blanc cassé Apple */
        --text-muted: #a1a1a6;
    }

    /* FOND GLOBAL SOMBRE */
    .stApp {
        background: radial-gradient(circle at 50% -20%, rgb(40, 40, 50) 0%, rgb(10, 10, 12) 80%);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: var(--text-color);
    }

    /* FIX PLEIN ÉCRAN (MODAL) */
    div[data-testid="stFullScreenFrame"] {
        background-color: #050505 !important;
        backdrop-filter: blur(0px) !important;
        z-index: 999999 !important;
    }
    div[data-testid="stFullScreenFrame"] button {
        color: white !important;
    }

    /* HEADER DARK GLASS - MODIFIÉ POUR ALIGNEMENT GAUCHE AVEC ESPACE */
    .mitwill-header {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 1.5rem 2rem;
        margin-top: -30px;
        margin-bottom: 2rem;
        display: flex;
        justify_content: flex-start; /* Changé de space-between à flex-start */
        align-items: center;
        gap: 30px; /* Espace ajouté entre le logo et la bulle */
        box-shadow: var(--glass-shadow);
        position: relative;
        z-index: 50;
    }
    .mitwill-logo {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .mitwill-logo span {
        color: var(--mitwill-orange);
    }
    .mitwill-subtitle {
        font-size: 13px;
        font-weight: 500;
        color: var(--text-muted);
        background: rgba(255,255,255,0.05);
        padding: 5px 12px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
    }

    /* SIDEBAR "SMOKED GLASS" */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
        color: #e0e0e0 !important;
    }

    /* BOUTONS DARK MODE */
    .stButton > button {
        background: linear-gradient(180deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.03) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        color: #ffffff;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(180deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.05) 100%);
        border-color: var(--mitwill-orange);
        color: white;
    }

    /* BOUTON PRIMAIRE ORANGE (GLOW) */
    div[data-testid="stButton"] button[kind="primary"] {
        background: var(--mitwill-orange);
        color: #000;
        font-weight: 700;
        border: none;
        box-shadow: 0 0 20px var(--mitwill-orange-glow);
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: rgb(255, 140, 95);
        box-shadow: 0 0 30px var(--mitwill-orange-glow);
    }

    /* CONTAINERS / EXPANDERS DARK */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        color: white !important;
    }
    [data-testid="stExpander"] {
        background: rgba(0, 0, 0, 0.4);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    
    /* SLIDERS & RADIOS */
    div[role="radiogroup"] {
        background: rgba(255,255,255,0.03);
        padding: 10px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    div[role="radiogroup"] label {
        color: #eee !important;
    }

    /* UPLOAD BOX DARK */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        padding: 20px;
        border-radius: 20px;
        border: 1px dashed rgba(255,255,255,0.2);
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--mitwill-orange);
        background: rgba(255, 255, 255, 0.05);
    }
    
    /* ALERTS */
    div[data-baseweb="notification"] {
        background-color: rgba(20, 20, 25, 0.9);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* PROGRESS BAR */
    .stProgress > div > div > div > div {
        background-color: var(--mitwill-orange);
    }

</style>

<div class="mitwill-header">
    <div class="mitwill-logo">MITWILL <span>TEXTILES</span></div>
    <div class="mitwill-subtitle">Digital Microfactory Tools • Dark Mode</div>
</div>
""", unsafe_allow_html=True)

# --- GESTION DU DATASET (STOCKAGE) ---
DATASET_ROOT = "mitwill_dataset"
os.makedirs(os.path.join(DATASET_ROOT, "ok"), exist_ok=True)
os.makedirs(os.path.join(DATASET_ROOT, "ko"), exist_ok=True)
os.makedirs(os.path.join(DATASET_ROOT, "logs"), exist_ok=True)

def save_to_dataset(img, filename, label, metrics):
    """Sauvegarde pour Mitwill Digital Archive."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        save_name = f"{timestamp}_{unique_id}_{filename}"
        
        save_path = os.path.join(DATASET_ROOT, label, save_name)
        img.save(save_path)
        
        log_entry = {
            "filename": save_name,
            "original_name": filename,
            "timestamp": timestamp,
            "label": label,
            "metrics": metrics,
            "source": "Mitwill Web Tool"
        }
        
        log_file = os.path.join(DATASET_ROOT, "logs", "archive_index.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        return True
    except Exception:
        return False

# --- CORE LOGIC (ANALYSIS) ---

def process_image_for_analysis(img):
    img.load() 
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass 

    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    else:
        return img.convert("RGB")

def calculate_ai_features(img):
    """Extraction features IA (Entropie, Contraste)."""
    gray = ImageOps.grayscale(img)
    arr = np.array(gray)
    
    histogram, _ = np.histogram(arr, bins=256, range=(0, 256), density=True)
    histogram = histogram[histogram > 0]
    entropy = -np.sum(histogram * np.log2(histogram))
    contrast = np.std(arr)
    
    return {
        "entropy": float(entropy),
        "contrast": float(contrast)
    }

def get_perceptual_diff(arr1, arr2, blur_radius=2):
    im1 = Image.fromarray(arr1.astype('uint8'))
    im2 = Image.fromarray(arr2.astype('uint8'))
    
    im1 = im1.filter(ImageFilter.GaussianBlur(blur_radius))
    im2 = im2.filter(ImageFilter.GaussianBlur(blur_radius))
    
    a1 = np.array(im1, dtype=np.int16)
    a2 = np.array(im2, dtype=np.int16)
    
    diff = np.abs(a1 - a2)
    diff_gray = np.mean(diff, axis=2)
    return diff_gray

def check_shading(img, threshold=15):
    """Détection Tuilage."""
    gray = ImageOps.grayscale(img)
    arr = np.array(gray)
    h, w = arr.shape
    
    strip_w = max(1, w // 10)
    strip_h = max(1, h // 10)
    
    left_mean = np.mean(arr[:, :strip_w])
    right_mean = np.mean(arr[:, -strip_w:])
    top_mean = np.mean(arr[:strip_h, :])
    bottom_mean = np.mean(arr[-strip_h:, :])
    
    diff_h = abs(left_mean - right_mean)
    diff_v = abs(top_mean - bottom_mean)
    
    issues = []
    if diff_h > threshold: issues.append(f"Horizontal (Δ {diff_h:.1f})")
    if diff_v > threshold: issues.append(f"Vertical (Δ {diff_v:.1f})")
        
    return issues, diff_h, diff_v

@st.cache_data(show_spinner=False)
def generate_simulation(img_bytes, repeat_mode, mirror_h=False, mirror_v=False):
    img = Image.open(BytesIO(img_bytes))
    img_sim = process_image_for_analysis(img)
    w, h = img_sim.size
    
    MAX_PREVIEW = 1000 
    if w > MAX_PREVIEW or h > MAX_PREVIEW:
        img_sim.thumbnail((MAX_PREVIEW, MAX_PREVIEW), Image.Resampling.LANCZOS)
        w, h = img_sim.size 
    
    cols, rows = 3, 3
    canvas_w = w * cols
    canvas_h = h * rows

    simulation = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    
    img_normal = img_sim
    img_flip_h = ImageOps.mirror(img_sim) if mirror_h else img_sim
    img_flip_v = ImageOps.flip(img_sim) if mirror_v else img_sim
    img_flip_hv = ImageOps.flip(img_flip_h) if mirror_v else img_flip_h

    for c in range(cols):
        for r in range(rows):
            x, y = c * w, r * h
            current_img = img_normal
            is_col_flipped = mirror_h and (c % 2 != 0)
            is_row_flipped = mirror_v and (r % 2 != 0)
            
            if is_col_flipped and is_row_flipped: current_img = img_flip_hv
            elif is_col_flipped: current_img = img_flip_h
            elif is_row_flipped: current_img = img_flip_v
            
            if repeat_mode == 'half_drop':
                if c % 2 != 0: 
                    y += (h // 2)
                    if r == 0: simulation.paste(current_img, (x, y - h))
            
            simulation.paste(current_img, (x, y))
            
    return simulation

def attempt_auto_fix(img):
    img = process_image_for_analysis(img)
    w, h = img.size
    margin = 20 
    if w < margin * 2 or h < margin * 2: return img 
    crop_w = int(w * 0.02)
    crop_h = int(h * 0.02)
    fixed_img = ImageOps.crop(img, border=min(crop_w, crop_h))
    fixed_img = fixed_img.resize((w, h), Image.Resampling.LANCZOS)
    return fixed_img

def draw_error_overlay(base_img, error_mask, axis, repeat_mode, shift_val=0):
    debug_img = base_img.copy()
    draw = ImageDraw.Draw(debug_img)
    w, h = debug_img.size
    indices = np.where(error_mask)[0]
    if len(indices) == 0: return debug_img
    
    # Couleurs adaptées au thème Orange
    color_static = (250, 125, 80) # Orange Mitwill
    color_moving = (200, 200, 200)   # Gris clair pour contraste sur fond noir

    if axis == 'H': 
        for idx in indices:
            draw.line([(w-5, idx), (w, idx)], fill=color_static, width=3) 
            target_y = (idx - shift_val) % h if repeat_mode == 'half_drop' else idx
            draw.line([(0, target_y), (5, target_y)], fill=color_moving, width=3)

    elif axis == 'V':
        for idx in indices:
            draw.line([(idx, h-5), (idx, h)], fill=color_static, width=3)
            draw.line([(idx, 0), (idx, 5)], fill=color_static, width=3)
                
    return debug_img

def check_pattern_seam(img, repeat_mode='standard', tolerance=15):
    clean_img = process_image_for_analysis(img)
    arr = np.array(clean_img)
    h, w, _ = arr.shape
    
    left = arr[:, 0:2, :] 
    right = arr[:, w-2:w, :]
    top = arr[0:2, :, :]
    bottom = arr[h-2:h, :, :]
    
    left_edge = np.mean(left, axis=1).astype(np.uint8)[np.newaxis, :, :]
    right_edge = np.mean(right, axis=1).astype(np.uint8)[np.newaxis, :, :]
    top_edge = np.mean(top, axis=0).astype(np.uint8)[np.newaxis, :, :]
    bottom_edge = np.mean(bottom, axis=0).astype(np.uint8)[np.newaxis, :, :]

    img_left = np.transpose(left_edge, (1, 0, 2))
    img_right = np.transpose(right_edge, (1, 0, 2))
    
    shift_h = h // 2 if repeat_mode == 'half_drop' else 0
    shifted_left = np.roll(img_left, shift_h, axis=0)
    diff_map_h = get_perceptual_diff(img_right, shifted_left, blur_radius=1)
    
    shift_v = 0
    shifted_top = np.roll(top_edge, shift_v, axis=1)
    diff_map_v = get_perceptual_diff(bottom_edge, shifted_top, blur_radius=1)
    
    score_h = np.percentile(diff_map_h, 95)
    score_v = np.percentile(diff_map_v, 95)
    
    error_mask_h = (diff_map_h > tolerance).flatten()
    error_mask_v = (diff_map_v > tolerance).flatten()
    
    return score_h, score_v, error_mask_h, error_mask_v, shift_h, shift_v, clean_img

def generate_pdf_report(data_list):
    if not HAS_REPORTLAB: return None
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    
    styles = getSampleStyleSheet()
    # Titre Style Mitwill
    title_style = styles['Title']
    title_style.textColor = colors.black
    title = Paragraph("MITWILL TEXTILES - RAPPORT QUALITÉ", title_style)
    elements.append(title)
    elements.append(Spacer(1, 30))
    
    table_data = [["FICHIER", "STATUT", "ERR. H", "ERR. V", "ENTROPIE", "CONTRASTE"]]
    for row in data_list:
        table_data.append([
            row["Fichier"][:20], # Tronquer nom long
            row["Statut"],
            str(row["Err. H"]),
            str(row["Err. V"]),
            str(row["IA Entropie"]),
            str(row["IA Contraste"])
        ])
    
    # Table Style Sobre (Orange accent)
    table = Table(table_data)
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(250/255, 125/255, 80/255)), # Orange header
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
    ])
    table.setStyle(style)
    
    elements.append(table)
    elements.append(Spacer(1, 30))
    
    footer_text = f"Analyse générée le {datetime.datetime.now().strftime('%d/%m/%Y')} via Mitwill Digital Microfactory Tools."
    footer = Paragraph(footer_text, styles['Normal'])
    elements.append(footer)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# --- INTERFACE UTILISATEUR (UI) ---

# SIDEBAR
with st.sidebar:
    st.markdown("### ⚙️ PARAMÈTRES")
    tolerance = st.slider("SEUIL DE TOLÉRANCE", 55, 65, 60, 
                          help="Ajustez la sensibilité de détection des coupures.")
    
    # DIGITAL ARCHIVE (CACHÉ MAIS ACTIF)
    enable_data_collection = True 
    # Le paramètre UI est supprimé comme demandé
    
    # SIMULATION MIROIR (SUPPRIMÉE)
    # Les paramètres UI sont supprimés comme demandé
    use_mirror_h = False
    use_mirror_v = False
    
    st.markdown("---")
    st.caption("Mitwill Textiles Europe © 2025")

# MAIN AREA
c_upload, c_mode = st.columns([2, 1])

with c_upload:
    uploaded_files = st.file_uploader(
        "IMPORTER DES FICHIERS (JPEG, PNG, TIFF)", 
        type=['jpg', 'png', 'tif', 'tiff'], 
        accept_multiple_files=True
    )

with c_mode:
    st.write("TYPE DE RACCORD")
    mode_choice = st.radio(
        "Type de raccord", 
        ('standard', 'half_drop'), 
        horizontal=False,
        label_visibility="collapsed",
        format_func=lambda x: {'standard':"🔁 STANDARD (Full Drop)", 'half_drop':"⬇️ SAUTÉ (Half-Drop)"}[x]
    )

if uploaded_files:
    if st.button(f"LANCER L'ANALYSE ({len(uploaded_files)} FICHIERS)", type="primary"):
        
        report_data = [] 
        
        # BARRE DE PROGRESSION MITWILL STYLE
        progress_bar = st.progress(0)
        
        for i, up_file in enumerate(uploaded_files):
            file_bytes = up_file.read()
            img_org = Image.open(BytesIO(file_bytes))
            
            # --- ANALYSE ---
            s_h, s_v, mask_h, mask_v, shift_h, shift_v, clean_img = check_pattern_seam(img_org, mode_choice, tolerance)
            shading_issues, diff_lum_h, diff_lum_v = check_shading(clean_img)
            ai_features = calculate_ai_features(clean_img)
            
            # --- VERDICT ---
            seam_ok = (s_h <= tolerance) and (s_v <= tolerance)
            shading_ok = len(shading_issues) == 0
            is_perfect = seam_ok and shading_ok
            
            status_txt = "CONFORME" if is_perfect else ("NON-CONFORME" if not seam_ok else "TUILAGE")
            icon = "✅" if is_perfect else ("❌" if not seam_ok else "⚠️")
            
            if enable_data_collection:
                label = "ok" if is_perfect else "ko"
                metrics = {
                    "seam_h": float(s_h), "seam_v": float(s_v), 
                    "entropy": ai_features['entropy']
                }
                save_to_dataset(img_org, up_file.name, label, metrics)

            report_data.append({
                "Fichier": up_file.name,
                "Statut": status_txt,
                "Err. H": round(s_h, 1),
                "Err. V": round(s_v, 1),
                "IA Entropie": round(ai_features['entropy'], 2),
                "IA Contraste": round(ai_features['contrast'], 2)
            })
            
            # --- VISUALISATION ---
            with st.expander(f"{icon} {up_file.name} — {status_txt}", expanded=not is_perfect):
                
                c_left, c_right = st.columns([1, 1.5])
                
                with c_left:
                    st.markdown("#### DIAGNOSTIC TECHNIQUE")
                    
                    # 1. RACCORD
                    if seam_ok:
                        st.info("Raccord géométrique : **OK**")
                    else:
                        st.error(f"Défaut de raccord détecté (H:{int(s_h)} V:{int(s_v)})")
                        debug_viz = clean_img.copy()
                        enhancer = ImageEnhance.Brightness(debug_viz)
                        debug_viz = enhancer.enhance(0.4)
                        if s_h > tolerance: debug_viz = draw_error_overlay(debug_viz, mask_h, 'H', mode_choice, shift_h)
                        if s_v > tolerance: debug_viz = draw_error_overlay(debug_viz, mask_v, 'V', mode_choice, shift_v)
                        st.image(debug_viz, caption="Localisation des coupures", use_container_width=True)
                        
                        # Auto-Fix Button (Styled)
                        if st.button("🛠️ TENTER UNE CORRECTION AUTO", key=f"fix_{up_file.name}"):
                            fixed = attempt_auto_fix(img_org)
                            st.image(fixed, caption="Proposition corrigée", use_container_width=True)
                            buf = BytesIO()
                            fixed.save(buf, format="JPEG", quality=95)
                            st.download_button("💾 SAUVEGARDER CORRECTION", data=buf.getvalue(), file_name=f"FIXED_{up_file.name}", mime="image/jpeg")

                    # 2. TUILAGE
                    if shading_ok:
                        st.caption("Luminosité uniforme (Pas de tuilage)")
                    else:
                        st.warning(f"Risque de Tuilage : {', '.join(shading_issues)}")
                        
                    st.markdown("---")
                    st.markdown("#### ANALYSE IA")
                    k1, k2 = st.columns(2)
                    k1.metric("Complexité", f"{ai_features['entropy']:.2f}")
                    k2.metric("Netteté", f"{ai_features['contrast']:.0f}")

                with c_right:
                    st.markdown("#### SIMULATION VISUELLE")
                    
                    # Simulation sans miroir (options supprimées)
                    sim_img = generate_simulation(file_bytes, mode_choice, use_mirror_h, use_mirror_v)
                    st.image(sim_img, use_container_width=True)
                    
                    # ZOOM AUTOMATIQUE
                    st.markdown("**INSPECTION DÉTAILLÉE (ZOOM X2 - RACCORD DROIT)**")
                    factor = 2
                    cw, ch = sim_img.size
                    crop_w, crop_h = cw // (3 * factor), ch // (3 * factor)
                    
                    # Décalage à 75% vers la droite pour visualiser la jointure
                    cx = int(cw * 0.75) 
                    cy = ch // 2
                    
                    zoom_box = (cx - crop_w, cy - crop_h, cx + crop_w, cy + crop_h)
                    zoom_img = sim_img.crop(zoom_box).resize((crop_w*4, crop_h*4), Image.Resampling.NEAREST)
                    st.image(zoom_img, use_container_width=True)
                    
                    # Download simulation
                    buf_sim = BytesIO()
                    sim_img.save(buf_sim, format="JPEG", quality=90)
                    st.download_button("⬇️ JPG SIMULATION", data=buf_sim.getvalue(), file_name=f"SIM_{up_file.name}.jpg", mime="image/jpeg", key=f"dl_{up_file.name}")

            progress_bar.progress((i + 1) / len(uploaded_files))

        # --- RAPPORT DE LOT ---
        st.markdown("### 📋 RAPPORT DE PRODUCTION")
        df_report = pd.DataFrame(report_data)
        st.dataframe(df_report, use_container_width=True)
        
        if HAS_REPORTLAB:
            pdf_buffer = generate_pdf_report(report_data)
            st.download_button(
                "📥 TÉLÉCHARGER LE RAPPORT PDF OFFICIEL",
                pdf_buffer,
                "mitwill_report.pdf",
                "application/pdf",
                key='download-pdf',
                type="secondary"
            )
        else:
            st.warning("Module PDF manquant.")

else:
    # MESSAGE D'ACCUEIL SOBRE
    st.info("Prêt pour l'analyse. Veuillez charger vos fichiers dans la zone ci-dessus.")