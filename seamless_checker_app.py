# -*- coding: utf-8 -*-
"""
Application Web Streamlit pour la Vérification de Raccord de Motif Textile.
Version Sécurisée + Optimisation Mémoire (Garbage Collection) pour éviter les crashs.

Dépendances: streamlit, Pillow, numpy
Lancement: python -m streamlit run seamless_checker_app.py
"""
import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageEnhance
import numpy as np
from io import BytesIO
import pandas as pd
import traceback
import gc # Garbage Collector pour vider la mémoire manuellement

# --- CONFIGURATION SÉCURITÉ ---
MOT_DE_PASSE = "textile2025" 

Image.MAX_IMAGE_PIXELS = None 

st.set_page_config(
    page_title="Vérificateur Textile Pro",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- FONCTION DE CONNEXION ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🔒 Accès Restreint")
        pwd = st.text_input("Mot de passe :", type="password")
        if st.button("Se connecter"):
            if pwd == MOT_DE_PASSE:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Mot de passe incorrect.")
        st.stop()

check_password()

# --- MOTEUR D'ANALYSE ---

def process_image_for_analysis(img):
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass 
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        bg = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        bg.paste(img, mask=img.split()[3]) 
        return bg
    else:
        return img.convert("RGB")

def get_optimal_shift_fft(static_arr, moving_arr, axis):
    s_gray = np.mean(static_arr, axis=1)
    m_gray = np.mean(moving_arr, axis=1)
    f_s = np.fft.fft(s_gray)
    f_m = np.fft.fft(m_gray)
    corr = np.fft.ifft(f_s * np.conj(f_m))
    best_shift = np.argmax(np.abs(corr))
    return int(best_shift)

def generate_simulation(img, repeat_mode):
    # On redimensionne tout de suite pour économiser la RAM
    img_sim = process_image_for_analysis(img)
    w, h = img_sim.size
    
    MAX_PREVIEW_SIZE = 800 # Réduit légèrement pour économiser de la mémoire
    if w > MAX_PREVIEW_SIZE or h > MAX_PREVIEW_SIZE:
        img_sim.thumbnail((MAX_PREVIEW_SIZE, MAX_PREVIEW_SIZE), Image.Resampling.LANCZOS)
        w, h = img_sim.size 
    
    cols, rows = 3, 3
    if repeat_mode == 'half_drop':
        canvas_w, canvas_h = w * cols, h * rows 
    elif repeat_mode == 'brick':
        canvas_w, canvas_h = w * cols + (w // 2), h * rows
    else:
        canvas_w, canvas_h = w * cols, h * rows
        
    simulation = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    
    for c in range(cols):
        for r in range(rows):
            x, y = c * w, r * h
            if repeat_mode == 'half_drop':
                if c % 2 != 0: 
                    y += (h // 2)
                    if r == 0: simulation.paste(img_sim, (x, y - h))
            elif repeat_mode == 'brick':
                if r % 2 != 0: 
                    x += (w // 2)
                    if c == 0: simulation.paste(img_sim, (x - w, y))
            simulation.paste(img_sim, (x, y))
    
    return simulation

def draw_error_overlay(draw, debug_w, debug_h, orig_w, orig_h, error_mask, axis, repeat_mode, img_array, shift_val=0):
    scale_x, scale_y = debug_w / orig_w, debug_h / orig_h
    THICKNESS, WHITE_THRESHOLD = 4, 250
    error_indices = np.where(error_mask)[0]
    
    for idx in error_indices:
        if axis == 'H':
            y_pos = int(idx * scale_y)
            pixel_val = img_array[idx, -1] 
            if np.mean(pixel_val) < WHITE_THRESHOLD: 
                draw.rectangle([debug_w - THICKNESS*2, y_pos, debug_w, y_pos + 2], fill="#FF0000")
            
            if repeat_mode == 'half_drop':
                idx_left = (idx - shift_val) % orig_h
                y_pos_left = int(idx_left * scale_y)
                pixel_val_left = img_array[idx_left, 0]
                if np.mean(pixel_val_left) < WHITE_THRESHOLD:
                    draw.rectangle([0, y_pos_left, THICKNESS*2, y_pos_left + 2], fill="#0088FF")
            else:
                pixel_val_left = img_array[idx, 0]
                if np.mean(pixel_val_left) < WHITE_THRESHOLD:
                    draw.rectangle([0, y_pos, THICKNESS*2, y_pos + 2], fill="#FF0000")
                
        elif axis == 'V':
            x_pos = int(idx * scale_x)
            pixel_val = img_array[-1, idx] 
            if np.mean(pixel_val) < WHITE_THRESHOLD:
                draw.rectangle([x_pos, debug_h - THICKNESS*2, x_pos + 2, debug_h], fill="#FF0000")
            
            if repeat_mode == 'brick':
                idx_top = (idx - shift_val) % orig_w
                x_pos_top = int(idx_top * scale_x)
                pixel_val_top = img_array[0, idx_top]
                if np.mean(pixel_val_top) < WHITE_THRESHOLD:
                    draw.rectangle([x_pos_top, 0, x_pos_top + 2, THICKNESS*2], fill="#0088FF")
            else:
                pixel_val_top = img_array[0, idx]
                if np.mean(pixel_val_top) < WHITE_THRESHOLD:
                    draw.rectangle([x_pos, 0, x_pos + 2, THICKNESS*2], fill="#FF0000")

def check_pattern_seam(img, repeat_mode='standard', tolerance=0, generate_debug=True):
    clean_img = process_image_for_analysis(img)
    img_array = np.array(clean_img, dtype=np.int16) 
    height, width, channels = img_array.shape
    if width < 2 or height < 2: return 255, 255, None
    left_edge, right_edge = img_array[:, 0, :], img_array[:, width - 1, :] 
    top_edge, bottom_edge = img_array[0, :, :], img_array[height - 1, :, :] 
    error_info_h, error_info_v = None, None

    # Horizontal
    final_shift_h = 0
    if repeat_mode == 'half_drop':
        detected_shift = get_optimal_shift_fft(right_edge, left_edge, axis=0)
        shifted_left = np.roll(left_edge, detected_shift, axis=0)
        diff_horizontal = np.abs(right_edge - shifted_left)
        final_shift_h = detected_shift
    else:
        diff_horizontal = np.abs(right_edge - left_edge)
    max_diff_horizontal = np.max(diff_horizontal)
    if max_diff_horizontal > tolerance:
        error_mask_H = (np.max(diff_horizontal, axis=1) > tolerance)
        if np.any(error_mask_H):
            error_info_h = {'mask': error_mask_H, 'shift': final_shift_h}

    # Vertical
    final_shift_v = 0
    if repeat_mode == 'brick':
        detected_shift_x = get_optimal_shift_fft(bottom_edge, top_edge, axis=0)
        shifted_top = np.roll(top_edge, detected_shift_x, axis=0)
        diff_vertical = np.abs(bottom_edge - shifted_top)
        final_shift_v = detected_shift_x
    else:
        diff_vertical = np.abs(bottom_edge - top_edge)
    max_diff_vertical = np.max(diff_vertical)
    if max_diff_vertical > tolerance:
        error_mask_V = (np.max(diff_vertical, axis=1) > tolerance)
        if np.any(error_mask_V):
             error_info_v = {'mask': error_mask_V, 'shift': final_shift_v}

    debug_img = None
    if generate_debug and (error_info_h or error_info_v):
        debug_img = clean_img.copy()
        debug_img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        enhancer = ImageEnhance.Brightness(debug_img)
        debug_img = enhancer.enhance(0.4) 
        draw = ImageDraw.Draw(debug_img)
        if error_info_h:
            draw_error_overlay(draw, debug_img.width, debug_img.height, width, height, error_info_h['mask'], 'H', repeat_mode, img_array, error_info_h['shift'])
        if error_info_v:
            draw_error_overlay(draw, debug_img.width, debug_img.height, width, height, error_info_v['mask'], 'V', repeat_mode, img_array, error_info_v['shift'])
    
    # NETTOYAGE MÉMOIRE IMMÉDIAT
    del clean_img
    del img_array
    
    return max_diff_horizontal, max_diff_vertical, debug_img

def detect_best_mode(img, tolerance=0):
    modes = ['standard', 'half_drop', 'brick']
    results = {}
    for m in modes:
        h, v, _ = check_pattern_seam(img, m, tolerance, generate_debug=False)
        results[m] = h + v 
    return min(results, key=results.get), results[min(results, key=results.get)]

# --- INTERFACE UTILISATEUR PRINCIPALE ---

st.title("🧵 Vérificateur Textile Pro")

with st.sidebar:
    st.header("Réglages")
    tolerance = st.slider(
        "Tolérance (Seuil d'erreur)", 
        min_value=55, 
        max_value=65, 
        value=60,
        help="Plage restreinte entre 55 et 65 pour filtrer les erreurs mineures et artefacts."
    )
    st.info(f"Tolérance : {tolerance}")
    if st.button("Se déconnecter"):
        st.session_state.authenticated = False
        st.rerun()
    
st.markdown("Outil de vérification de raccord. Technologie FFT.")
mode_choice = st.radio("Mode :", ('standard', 'half_drop', 'brick'), format_func=lambda x: {'standard': "Standard", 'half_drop': "Sauté (Half-Drop)", 'brick': "Quinconce"}[x])
st.divider()

uploaded_files = st.file_uploader("Fichiers (PNG, JPG, TIFF) :", type=['png', 'jpg', 'jpeg', 'tiff', 'tif'], accept_multiple_files=True)

if uploaded_files:
    if st.button(f"Analyser {len(uploaded_files)} fichier(s)", type="primary"):
        
        # Barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Zone pour afficher les résultats au fur et à mesure
        results_container = st.container()
        
        # Liste légère pour le résumé final (sans images)
        summary_data = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Traitement de {uploaded_file.name} ({i+1}/{len(uploaded_files)})...")
            
            try:
                # 1. Chargement
                image_data = uploaded_file.read()
                image = Image.open(BytesIO(image_data))
                
                # 2. Analyse
                max_h, max_v, debug_img = check_pattern_seam(image, mode_choice, tolerance)
                sim_img = generate_simulation(image, mode_choice)
                
                is_ok = (max_h <= tolerance) and (max_v <= tolerance)
                
                # Suggestion (si nécessaire)
                suggestion = None
                if not is_ok:
                    best_mode, best_err = detect_best_mode(image, tolerance)
                    if best_mode != mode_choice and best_err <= tolerance:
                        suggestion = best_mode
                
                statut_icon = "✅ OK" if is_ok else "❌ KO"
                
                # 3. Affichage IMMÉDIAT (pour libérer la variable image ensuite)
                with results_container:
                    container = st.expander(f"{uploaded_file.name} - {statut_icon}", expanded=(not is_ok))
                    with container:
                        if suggestion: 
                            st.warning(f"Suggestion : Essayez le mode {suggestion.upper()}")
                        
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            # Affichage miniature de l'original pour économiser RAM navigateur
                            # On n'affiche plus l'original HD ici, juste le debug ou la simulation
                            if max_h > tolerance: st.error(f"H: Erreur {max_h}")
                            else: st.success(f"H: OK")
                            if max_v > tolerance: st.error(f"V: Erreur {max_v}")
                            else: st.success(f"V: OK")
                            
                            if debug_img: 
                                st.image(debug_img, caption="Erreurs", use_container_width=True)
                            else:
                                # Si pas d'erreur, on montre une version réduite de l'original
                                small_preview = image.copy()
                                small_preview.thumbnail((500, 500))
                                st.image(small_preview, caption="Aperçu", use_container_width=True)
                                del small_preview

                        with c2:
                            st.image(sim_img, use_container_width=True)

                # 4. Stockage données légères pour résumé
                summary_data.append({
                    "Fichier": uploaded_file.name, 
                    "Statut": statut_icon,
                    "Err. H": max_h, 
                    "Err. V": max_v
                })

                # 5. NETTOYAGE VIOLENT DE LA MÉMOIRE
                del image
                del sim_img
                del debug_img
                del image_data
                gc.collect() # Force le nettoyage de la RAM
            
            except Exception as e:
                st.error(f"Erreur sur {uploaded_file.name}: {e}")
            
            # Mise à jour barre
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("Terminé !")
        
        # Affichage du tableau récapitulatif à la fin (léger)
        st.divider()
        st.subheader("Résumé Global")
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
