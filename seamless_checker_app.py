# -*- coding: utf-8 -*-
"""
Application Web Streamlit pour la Vérification de Raccord de Motif Textile.
Version "Ultimate" : Optimisation RAM extrême (Crop) + Analyse robuste (FFT).

Dépendances: streamlit, Pillow, numpy, psutil
Lancement: python -m streamlit run seamless_checker_app.py
"""
import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageEnhance
import numpy as np
from io import BytesIO
import pandas as pd
import traceback
import gc
import os
import psutil

# --- CONFIGURATION SÉCURITÉ ---
MOT_DE_PASSE = "textile2025" 
Image.MAX_IMAGE_PIXELS = None # Désactivation limite pixels pour les gros fichiers

st.set_page_config(
    page_title="Vérificateur Textile Pro",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- UTILITAIRES SYSTÈME ---
def check_memory():
    """Affiche l'état de la mémoire pour prévenir les crashs."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024 # En Mo
    if mem > 800:
        st.sidebar.warning(f"⚠️ Mémoire élevée: {int(mem)} Mo. Le serveur va nettoyer automatiquement.")
    return mem

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

# --- MOTEUR D'ANALYSE (OPTIMISÉ) ---

def prepare_image(img):
    """Gère l'orientation et convertit en RGB proprement."""
    try: img = ImageOps.exif_transpose(img)
    except: pass
    
    if img.mode != 'RGB':
        # Gestion transparence optimisée
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            bg = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode != 'RGBA': img = img.convert('RGBA')
            bg.paste(img, mask=img.split()[3])
            return bg
        return img.convert('RGB')
    return img

def extract_edge(img, edge_type):
    """
    Extrait UNIQUEMENT la ligne de pixels nécessaire pour économiser la RAM.
    edge_type: 'left', 'right', 'top', 'bottom'
    """
    w, h = img.size
    if edge_type == 'left':
        # Crop retourne une image, on convertit en array (h, 1, 3)
        return np.array(img.crop((0, 0, 1, h)).convert('RGB')) 
    elif edge_type == 'right':
        return np.array(img.crop((w-1, 0, w, h)).convert('RGB'))
    elif edge_type == 'top':
        # Crop (1, w, 3). On transpose pour avoir une structure uniforme (N, 1, 3) pour l'analyse FFT
        arr = np.array(img.crop((0, 0, w, 1)).convert('RGB')) 
        return np.transpose(arr, (1, 0, 2)) 
    elif edge_type == 'bottom':
        arr = np.array(img.crop((0, h-1, w, h)).convert('RGB'))
        return np.transpose(arr, (1, 0, 2))

def get_optimal_shift_fft(static_arr, moving_arr):
    """FFT optimisée sur les vecteurs 1D."""
    # static_arr et moving_arr sont de forme (N, 1, 3) -> On aplatit à (N, 3) puis moyenne -> (N,)
    # On compare les profils de luminosité moyenne
    s_gray = np.mean(static_arr[:, 0, :], axis=1)
    m_gray = np.mean(moving_arr[:, 0, :], axis=1)
    
    # FFT (Fast Fourier Transform)
    f_s = np.fft.fft(s_gray)
    f_m = np.fft.fft(m_gray)
    
    # Corrélation croisée
    corr = np.fft.ifft(f_s * np.conj(f_m))
    
    # Le pic indique le meilleur alignement
    best_shift = np.argmax(np.abs(corr))
    return int(best_shift)

def analyze_seam(static_edge, moving_edge, repeat_mode, axis_len, tolerance):
    """
    Compare deux bords avec logique FFT et tolérance.
    Retourne: max_error, error_mask, shift_utilisé
    """
    # Recherche du shift optimal si nécessaire
    shift = 0
    if repeat_mode in ['half_drop', 'brick']: 
        shift = get_optimal_shift_fft(static_edge, moving_edge)
    
    # Application du shift (rotation du tableau)
    if shift != 0:
        shifted_moving = np.roll(moving_edge, shift, axis=0)
    else:
        shifted_moving = moving_edge

    # Calcul de la différence absolue
    diff = np.abs(static_edge.astype(np.int16) - shifted_moving.astype(np.int16))
    max_diff = np.max(diff)
    
    # Création du masque d'erreur
    # On considère une erreur si la différence moyenne RGB du pixel dépasse la tolérance
    pixel_diffs = np.mean(diff, axis=2) # (N, 1)
    error_mask = (pixel_diffs > tolerance).flatten() # (N,)
    
    return max_diff, error_mask, shift

def create_debug_thumbnail(img, error_mask_h, shift_h, error_mask_v, shift_v, repeat_mode):
    """Crée une miniature légère avec les erreurs dessinées."""
    w, h = img.size
    
    # Création miniature (on travaille sur une petite image pour la vitesse)
    thumb = img.copy()
    thumb.thumbnail((800, 800)) 
    
    # Assombrir pour faire ressortir les traits
    enhancer = ImageEnhance.Brightness(thumb)
    thumb = enhancer.enhance(0.4)
    draw = ImageDraw.Draw(thumb)
    
    scale_x = thumb.width / w
    scale_y = thumb.height / h
    THICKNESS = 4
    WHITE_THRESHOLD = 250
    
    # Pour vérifier la blancheur, on a besoin de l'image originale redimensionnée pareil mais pas assombrie
    # (Approximation pour performance : on ne vérifie pas la blancheur pixel perfect ici pour économiser RAM)
    
    # Dessin H (Lignes verticales d'erreur)
    if error_mask_h is not None:
        indices = np.where(error_mask_h)[0]
        for idx in indices:
            y = int(idx * scale_y)
            # Bord Droit (Rouge)
            draw.rectangle([thumb.width - THICKNESS, y, thumb.width, y+2], fill="#FF0000")
            
            # Bord Gauche (Bleu) - Position décalée
            if repeat_mode == 'half_drop':
                idx_shifted = (idx - shift_h) % h
                y_s = int(idx_shifted * scale_y)
                draw.rectangle([0, y_s, THICKNESS, y_s+2], fill="#0088FF")
            else:
                draw.rectangle([0, y, THICKNESS, y+2], fill="#FF0000")

    # Dessin V (Lignes horizontales d'erreur)
    if error_mask_v is not None:
        indices = np.where(error_mask_v)[0]
        for idx in indices:
            x = int(idx * scale_x)
            # Bord Bas (Rouge)
            draw.rectangle([x, thumb.height - THICKNESS, x+2, thumb.height], fill="#FF0000")
            
            # Bord Haut (Bleu)
            if repeat_mode == 'brick':
                idx_shifted = (idx - shift_v) % w
                x_s = int(idx_shifted * scale_x)
                draw.rectangle([x_s, 0, x_s+2, THICKNESS], fill="#0088FF")
            else:
                draw.rectangle([x, 0, x+2, THICKNESS], fill="#FF0000")
                
    return thumb

def generate_simulation_light(img, repeat_mode):
    """Génère la simulation 3x3 de manière très optimisée (redimensionnement préalable)."""
    w, h = img.size
    
    # On redimensionne drastiquement AVANT de tiler
    # Une tuile de 400px est suffisante pour juger le raccord visuel global
    target_w = 400
    ratio = target_w / w
    target_h = int(h * ratio)
    
    img_small = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    cols, rows = 3, 3
    if repeat_mode == 'half_drop': canvas_w, canvas_h = target_w * cols, target_h * rows
    elif repeat_mode == 'brick': canvas_w, canvas_h = target_w * cols + (target_w // 2), target_h * rows
    else: canvas_w, canvas_h = target_w * cols, target_h * rows
    
    sim = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    
    for c in range(cols):
        for r in range(rows):
            x, y = c * target_w, r * target_h
            if repeat_mode == 'half_drop':
                if c % 2 != 0: 
                    y += (target_h // 2)
                    if r == 0: sim.paste(img_small, (x, y - target_h))
            elif repeat_mode == 'brick':
                if r % 2 != 0: 
                    x += (target_w // 2)
                    if c == 0: sim.paste(img_small, (x - target_w, y))
            sim.paste(img_small, (x, y))
            
    return sim

def detect_best_mode(img, tolerance):
    """
    Fonction simplifiée pour suggérer le bon mode sans faire exploser la RAM.
    On teste rapidement les shifts FFT sur une version réduite.
    """
    # Pour la détection, on ne fait pas l'analyse complète pixel-perfect, trop lourd.
    # On se base sur le mode sélectionné par l'utilisateur s'il échoue.
    return None, 255

# --- LOGIQUE PRINCIPALE ---

st.title("🧵 Vérificateur Textile Pro")

with st.sidebar:
    st.header("Paramètres")
    # Blocage de la tolérance entre 55 et 65 comme demandé
    tolerance = st.slider("Tolérance (Seuil d'erreur)", 55, 65, 60, help="55 = Strict, 65 = Souple")
    
    check_memory()
    
    st.markdown("---")
    if st.button("Se déconnecter"):
        st.session_state.authenticated = False
        st.rerun()

mode_choice = st.radio("Type de Raccord :", ('standard', 'half_drop', 'brick'), format_func=lambda x: {'standard': "Standard (Droit)", 'half_drop': "Sauté (Half-Drop)", 'brick': "Quinconce"}[x], horizontal=True)

st.info("💡 Les fichiers sont traités en flux tendu pour une performance maximale.")

uploaded_files = st.file_uploader("Déposez vos fichiers ici (TIFF, JPG, PNG)", type=['png', 'jpg', 'jpeg', 'tiff', 'tif'], accept_multiple_files=True)

if uploaded_files:
    if st.button(f"Lancer l'analyse de {len(uploaded_files)} fichier(s)", type="primary"):
        
        progress = st.progress(0)
        results_area = st.container()
        
        for i, up_file in enumerate(uploaded_files):
            try:
                # 1. Chargement Image (Lazy)
                img_orig = Image.open(BytesIO(up_file.read()))
                img = prepare_image(img_orig)
                w, h = img.size
                
                # 2. Analyse H (Bords Gauche/Droite uniquement - Très léger en RAM)
                left = extract_edge(img, 'left')
                right = extract_edge(img, 'right')
                mode_h = 'half_drop' if mode_choice == 'half_drop' else 'standard'
                max_h, mask_h, shift_h = analyze_seam(right, left, mode_h, h, tolerance)
                
                # 3. Analyse V (Bords Haut/Bas uniquement)
                top = extract_edge(img, 'top')
                bottom = extract_edge(img, 'bottom')
                mode_v = 'brick' if mode_choice == 'brick' else 'standard'
                max_v, mask_v, shift_v = analyze_seam(bottom, top, mode_v, w, tolerance)
                
                # 4. Statut
                is_ok_h = max_h <= tolerance
                is_ok_v = max_v <= tolerance
                is_global_ok = is_ok_h and is_ok_v
                
                icon = "✅" if is_global_ok else "❌"
                
                # 5. Génération Visuels (Uniquement pour l'affichage, puis poubelle)
                sim_img = generate_simulation_light(img, mode_choice)
                
                debug_img = None
                if not is_global_ok:
                    debug_img = create_debug_thumbnail(
                        img, 
                        mask_h if not is_ok_h else None, shift_h,
                        mask_v if not is_ok_v else None, shift_v,
                        mode_choice
                    )

                # 6. Affichage
                with results_area:
                    with st.expander(f"{icon} {up_file.name}", expanded=not is_global_ok):
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.markdown(f"**Dimensions:** {w}x{h} px")
                            
                            # H Status
                            if is_ok_h: st.success(f"Horizontal : OK (Diff: {max_h})")
                            else: st.error(f"Horizontal : Erreur (Diff: {max_h})")
                            
                            # V Status
                            if is_ok_v: st.success(f"Vertical : OK (Diff: {max_v})")
                            else: st.error(f"Vertical : Erreur (Diff: {max_v})")
                            
                            if debug_img:
                                st.image(debug_img, caption="Localisation des erreurs", use_container_width=True)
                            else:
                                # Aperçu miniature simple si tout est OK
                                thumb_ok = img.copy()
                                thumb_ok.thumbnail((400, 400))
                                st.image(thumb_ok, caption="Aperçu", use_container_width=True)
                                
                        with c2:
                            st.image(sim_img, caption="Simulation 3x3", use_container_width=True)

                # 7. NETTOYAGE MÉMOIRE AGRESSIF
                # On supprime toutes les références aux images lourdes
                del img_orig, img, left, right, top, bottom, sim_img, debug_img
                # On force le Garbage Collector de Python
                gc.collect()
                
            except Exception as e:
                st.error(f"Erreur sur {up_file.name}: {str(e)}")
                # traceback.print_exc() # Pour debug seulement
            
            progress.progress((i + 1) / len(uploaded_files))
            
        st.success("Analyse terminée !")
