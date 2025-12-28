import streamlit as st
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import psutil
import os
import gc
import math

# --- CONFIGURATION INITIALE & SÉCURITÉ ---
st.set_page_config(
    page_title="Mitwill Seamless Checker Pro",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)

Image.MAX_IMAGE_PIXELS = None

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def check_password():
    def password_entered():
        if st.session_state["password"] == "textile2025":
            st.session_state.authenticated = True
            del st.session_state["password"]
        else:
            st.session_state.authenticated = False
            st.error("Mot de passe incorrect")

    if not st.session_state.authenticated:
        st.text_input("Mot de passe Technicien", type="password", on_change=password_entered, key="password")
        return False
    return True

if not check_password():
    st.stop()

# --- FONCTIONS UTILITAIRES ---

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def load_image_optimized(uploaded_file):
    img = Image.open(uploaded_file)
    img.load() 
    return img

def safe_crop_to_array(img, box):
    crop = img.crop(box)
    # Conversion 'L' (Luminance) suffisante pour détecter les structures
    arr = np.array(crop.convert('L'), dtype=np.int16) 
    del crop
    return arr

# --- CŒUR MATHÉMATIQUE AVANCÉ (V2) ---

def smooth_signal(signal, window_size=3):
    """Lisse le signal pour ignorer le bruit JPG/Grain (Moyenne glissante)."""
    kernel = np.ones(window_size) / window_size
    # Mode 'same' garde la même taille de tableau
    return np.convolve(signal, kernel, mode='same')

def analyze_seam_fft(signal1, signal2):
    """
    Analyse robuste V2 :
    1. Guard: Check si zone unie (évite faux positifs sur fond blanc).
    2. Smooth: Lissage pour éviter le bruit.
    3. FFT: Corrélation de phase.
    """
    # 1. SOLID COLOR GUARD (Protection Fond Uni)
    # Si l'écart-type est très faible (< 5 sur 255), c'est du blanc/noir/uni.
    std1 = np.std(signal1)
    std2 = np.std(signal2)
    
    if std1 < 5 and std2 < 5:
        # C'est un fond uni, donc c'est "raccord" par définition.
        return 100.0, 0

    # 2. SIGNAL SMOOTHING (Réduction Bruit JPG)
    s1 = smooth_signal(signal1)
    s2 = smooth_signal(signal2)

    # Normalisation (Centrer sur 0)
    s1 = s1 - np.mean(s1)
    s2 = s2 - np.mean(s2)
    
    # 3. FFT CORRELATION
    f1 = np.fft.fft(s1)
    f2 = np.fft.fft(s2)
    correlation = np.fft.ifft(f1 * np.conj(f2))
    
    shift = np.argmax(np.abs(correlation))
    
    # Calcul du score visuel basé sur la différence moyenne des signaux lissés
    # On aligne mathématiquement les signaux selon le shift trouvé pour voir si ça matche
    # (Note: Pour l'UI simple, on compare sans shift pour pénaliser le décalage actuel)
    diff = np.mean(np.abs(s1 - s2))
    
    # Score 0-100. Une différence moyenne de 0 = 100. Diff de 20 = 80.
    # On est un peu plus tolérant grâce au lissage.
    score_visual = max(0, 100 - diff)
    
    return score_visual, shift

def get_shift_message(shift, axis, img_dim):
    if shift == 0:
        return None
    msg = ""
    abs_shift = abs(int(shift))
    
    # Seuil minimal pour suggérer une correction (éviter de dire "bouger de 1px" pour rien)
    if abs_shift < 2: 
        return "Décalage infime (<2px), vérifier visuellement."

    if axis == "V": 
        direction = "VERS LE HAUT" if shift > 0 else "VERS LE BAS"
        msg = f"Décaler le bord Droit de {abs_shift}px {direction}"
    else: 
        direction = "VERS LA GAUCHE" if shift > 0 else "VERS LA DROITE"
        msg = f"Décaler le Bas de {abs_shift}px {direction}"
    return msg

def check_seamlessness(img, mode="Standard", tolerance=60):
    w, h = img.size
    
    # V2: Augmentation de la bande d'analyse à 5px pour plus de robustesse
    strip_size = 5 
    
    # --- 1. Raccord Vertical (Bord Gauche vs Bord Droit) ---
    # Crop Gauche (Hauteur h, Largeur 5)
    left_arr = safe_crop_to_array(img, (0, 0, strip_size, h))
    # Crop Droit
    right_arr = safe_crop_to_array(img, (w - strip_size, 0, w, h))
    
    # Important : On fait la moyenne sur la LARGEUR de la bandelette (axis=1)
    # pour obtenir un profil 1D de longueur H (le motif qui descend).
    left_profile = np.mean(left_arr, axis=1)
    right_profile = np.mean(right_arr, axis=1)
    
    score_v, shift_v = analyze_seam_fft(left_profile, right_profile)
    
    del left_arr, right_arr, left_profile, right_profile
    gc.collect()

    # --- 2. Raccord Horizontal (Haut vs Bas) ---
    top_arr = safe_crop_to_array(img, (0, 0, w, strip_size))
    bottom_arr = safe_crop_to_array(img, (0, h - strip_size, w, h))
    
    if mode == "Half-Drop (Sauté)":
        shift_amount = w // 2
        bottom_arr = np.roll(bottom_arr, shift_amount, axis=1)

    # Important : On fait la moyenne sur la HAUTEUR de la bandelette (axis=0)
    # pour obtenir un profil 1D de longueur W (le motif qui traverse).
    top_profile = np.mean(top_arr, axis=0)
    bottom_profile = np.mean(bottom_arr, axis=0)

    score_h, shift_h = analyze_seam_fft(top_profile, bottom_profile)
    
    del top_arr, bottom_arr, top_profile, bottom_profile
    gc.collect()
    
    # --- Interprétation ---
    is_seamless_v = score_v >= tolerance
    is_seamless_h = score_h >= tolerance
    
    return {
        "score_v": score_v,
        "score_h": score_h,
        "shift_v_detected": shift_v if shift_v < h/2 else shift_v - h,
        "shift_h_detected": shift_h if shift_h < w/2 else shift_h - w,
        "is_seamless": is_seamless_v and is_seamless_h
    }

# --- VISUALISATION ---

def generate_seam_zoom(img, mode, axis="H"):
    w, h = img.size
    zoom_size = 150
    
    if axis == "H":
        top_sample = img.crop((w//2 - zoom_size//2, 0, w//2 + zoom_size//2, zoom_size))
        if mode == "Half-Drop (Sauté)":
            start_x = (w//2 + w//2) % w
            bottom_sample = img.crop((start_x - zoom_size//2, h - zoom_size, start_x + zoom_size//2, h))
        else:
            bottom_sample = img.crop((w//2 - zoom_size//2, h - zoom_size, w//2 + zoom_size//2, h))

        proof = Image.new('RGB', (zoom_size, zoom_size*2))
        proof.paste(bottom_sample, (0, 0)) 
        proof.paste(top_sample, (0, zoom_size))
        draw = ImageDraw.Draw(proof)
        draw.line([(0, zoom_size), (zoom_size, zoom_size)], fill="red", width=1)
        
    else: # axis == "V"
        right_sample = img.crop((w - zoom_size, h//2 - zoom_size//2, w, h//2 + zoom_size//2))
        left_sample = img.crop((0, h//2 - zoom_size//2, zoom_size, h//2 + zoom_size//2))
        
        proof = Image.new('RGB', (zoom_size*2, zoom_size))
        proof.paste(right_sample, (0, 0))
        proof.paste(left_sample, (zoom_size, 0))
        draw = ImageDraw.Draw(proof)
        draw.line([(zoom_size, 0), (zoom_size, zoom_size)], fill="red", width=1)
    
    return proof

def generate_3x3_preview(img, mode):
    thumb = img.copy()
    thumb.thumbnail((300, 300))
    tw, th = thumb.size
    sim_w, sim_h = tw * 3, th * 3
    sim_img = Image.new('RGB', (sim_w, sim_h))
    
    for r in range(3):
        for c in range(3):
            x = c * tw
            y = r * th
            if mode == "Half-Drop (Sauté)" and c % 2 != 0:
                y += th // 2
                if y > sim_h: y -= sim_h
            sim_img.paste(thumb, (x, y))
    return sim_img

# --- UI PRINCIPALE ---

st.sidebar.title("🏭 Mitwill Microfactory")
st.sidebar.markdown("---")

mode = st.sidebar.radio("Mode de Raccord", ["Standard (Droit)", "Half-Drop (Sauté)"])
tolerance = st.sidebar.slider("Tolérance (Anti-Bruit)", 55, 75, 60, help="Augmentez si vos fichiers sont très bruités (JPG)")

st.sidebar.markdown("### 📊 État Système")
ram_placeholder = st.sidebar.empty()

st.title("🧵 Smart Seamless Checker V2")
st.markdown("*Analyse mathématique robuste (Anti-bruit & détection zones unies)*")

uploaded_file = st.file_uploader("Glisser le motif ici (Max 500Mo)", type=['png', 'jpg', 'jpeg', 'tiff', 'tif'])

if uploaded_file:
    try:
        img = load_image_optimized(uploaded_file)
        w, h = img.size
        st.success(f"Image chargée : {w}x{h} px | Mode : {img.mode}")
        ram_placeholder.metric("RAM Utilisée", f"{get_ram_usage():.1f} Mo")

        if st.button("Lancer l'Analyse (FFT V2)", type="primary"):
            with st.spinner("Extraction, Lissage & Calcul FFT..."):
                results = check_seamlessness(img, mode, tolerance)
                
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Score Vertical (G/D)", f"{results['score_v']:.1f}/100", 
                           delta="OK" if results['score_v'] >= tolerance else "CASSURE",
                           delta_color="normal" if results['score_v'] >= tolerance else "inverse")
                
                col2.metric("Score Horizontal (H/B)", f"{results['score_h']:.1f}/100",
                           delta="OK" if results['score_h'] >= tolerance else "CASSURE",
                           delta_color="normal" if results['score_h'] >= tolerance else "inverse")
                
                status_color = "green" if results['is_seamless'] else "red"
                status_text = "RACCORD VALIDE" if results['is_seamless'] else "CORRECTION REQUISE"
                col3.markdown(f"<h3 style='color:{status_color}; text-align:center;'>{status_text}</h3>", unsafe_allow_html=True)

                if not results['is_seamless']:
                    with st.expander("🛠️ Diagnostic & Correction", expanded=True):
                        st.warning("Le motif présente des discontinuités.")
                        
                        if results['score_v'] < tolerance:
                            shift_v = results['shift_v_detected']
                            msg_v = get_shift_message(shift_v, "V", h)
                            st.info(f"**Problème Vertical** : {msg_v}")

                        if results['score_h'] < tolerance:
                            shift_h = results['shift_h_detected']
                            msg_h = get_shift_message(shift_h, "H", w)
                            st.info(f"**Problème Horizontal** : {msg_h}")

                st.markdown("---")
                st.subheader("🔍 Preuve Visuelle")
                c_vis1, c_vis2, c_vis3 = st.columns([1, 1, 2])
                
                with c_vis1:
                    st.caption("Zoom Vertical (G/D)")
                    proof_v = generate_seam_zoom(img, mode, axis="V")
                    st.image(proof_v, use_container_width=True)

                with c_vis2:
                    st.caption("Zoom Horizontal (H/B)")
                    proof_h = generate_seam_zoom(img, mode, axis="H")
                    st.image(proof_h, use_container_width=True)
                
                with c_vis3:
                    st.caption("Simulation Tissu")
                    sim = generate_3x3_preview(img, mode)
                    st.image(sim, use_container_width=True)

        gc.collect()
        ram_placeholder.metric("RAM (Post-Clean)", f"{get_ram_usage():.1f} Mo")

    except Exception as e:
        st.error(f"Erreur technique : {e}")
        
else:
    st.info("Le système V2 est prêt (Noyau de lissage actif).")
