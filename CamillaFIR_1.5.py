import numpy as np
import scipy.signal
import scipy.io.wavfile
import scipy.fft
import sys
import os
import io
import json
import locale
from datetime import datetime

# --- MATPLOTLIB SETUP ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- GUI LIBRARY ---
from pywebio.input import *
from pywebio.output import *
from pywebio import start_server, config

# --- CONSTANTS ---
CONFIG_FILE = 'config.json'
FINE_TUNE_LIMIT = 45.0
MAX_SAFE_BOOST = 8.0
VERSION = "v1.5.0"
PROGRAM_NAME = "CamillaFIR"

# --- LANGUAGE DETECTION & TRANSLATIONS ---
def detect_language():
    try:
        sys_lang = locale.getdefaultlocale()[0]
        if sys_lang and 'fi' in sys_lang.lower():
            return 'fi'
    except:
        pass
    return 'en'

CURRENT_LANG = detect_language()

TRANSLATIONS = {
    'en': {
        'title': "CamillaFIR",
        'subtitle': "By VilhoValittu & GeminiPro",
        
        # GUIDE 1: TAPS
        'guide_title': "❓ Guide: How to choose Sample Rate and Taps?",
        'guide_rule': "Rule of Thumb: When Sample Rate doubles, Taps count should also double.",
        'guide_formula': "FIR filter bass resolution (below 100 Hz) depends directly on ratio: Sample Rate / Taps.",
        'guide_desc': "If you use a high Sample Rate (e.g. 192 kHz) but keep Taps low (65k), filter resolution decreases and bass correction becomes inaccurate.",
        'guide_rec': "Recommended Values:",
        'guide_note': "Note: Higher Taps value causes more latency, but is acoustically more accurate.",
        
        # GUIDE 2: FILTER TYPE
        'guide_ft_title': "❓ Guide: Linear Phase vs. Minimum Phase?",
        'guide_ft_lin_h': "1. Linear Phase (Default)",
        'guide_ft_lin_desc': "• Corrects phase timing errors.\n• High Latency.\n• Heavy for CPU.",
        'guide_ft_min_h': "2. Minimum Phase (Zero Latency)",
        'guide_ft_min_desc': "• Zero Latency (Gaming/TV).\n• No Pre-ringing.\n• Very light for CPU.",
        'guide_ft_rec': "Rec: Linear for Music. Minimum for Movies/Gaming/DSD.",

        # GUIDE 3: FDW (NEW v1.6.3)
        'guide_fdw_title': "❓ Guide: How to choose FDW Cycles?",
        'guide_fdw_desc': "FDW (Frequency Dependent Windowing) determines how much of the 'room sound' is corrected vs. the direct speaker sound.",
        'guide_fdw_low': "• Low Value (3 - 6): Aggressive. Removes almost all room reflections. Sound becomes very 'dry' and direct. Good for nearfield monitoring.",
        'guide_fdw_mid': "• Standard (15): RECOMMENDATION. Good balance. Corrects room modes in bass but keeps natural airiness in treble.",
        'guide_fdw_high': "• High Value (30 - 100): Gentle. Includes most room reflections. Corrects the steady-state room curve. Good for 'lively' rooms.",

        'grp_files': "Input Files (Choose Upload OR Local Path)",
        'upload_l': "Upload Measurement L (.txt)",
        'upload_r': "Upload Measurement R (.txt)",
        'path_l': "OR Local File Path L",
        'path_r': "OR Local File Path R",
        'path_help': "Paste full path (e.g. C:\\Audio\\L.txt) to save location for next time.",
        'ph_l': "Choose Left Channel measurement",
        'ph_r': "Choose Right Channel measurement",
        'fmt': "Output Format",
        'layout': "Channel Layout",
        'layout_mono': "Mono (Separate files)",
        'layout_stereo': "Stereo (Single file)",
        'fs': "Sample Rate",
        'taps': "Taps",
        'taps_help': "Filter length. Higher = Better bass resolution but more CPU load.",
        'filter_type': "Filter Type (Latency)",
        'ft_linear': "Linear Phase (Phase Correction, High Latency)",
        'ft_min': "Minimum Phase (No Phase Corr, Zero Latency)",
        'ft_help': "Linear: Perfect step response, delay. Minimum: No delay, fast loading.",
        'gain': "Global Gain (dB)",
        'gain_help': "Adjusts output volume. Use negative values (e.g. -3.0) to prevent clipping.",
        'smooth_type': "Smoothing Type",
        'smooth_std': "Standard 1/48 oct",
        'smooth_psy': "Psychoacoustic",
        'fdw': "FDW Cycles",
        'fdw_help': "Controls phase correction window. 15 = Balanced. 6 = Aggressive (Direct sound).",
        'hc_mode': "House Curve Mode",
        'hc_mode_builtin': "Built-in ('not Dr Toole')",
        'hc_mode_upload': "Upload Custom...",
        'hc_custom': "Custom House Curve (Optional)",
        'hc_custom_help': "Required if 'Upload Custom...' mode is selected above.",
        'corr_mode': "Correction Mode",
        'enable_corr': "Enable Frequency Response Correction",
        'min_freq': "Correction Min Freq (Hz)",
        'max_freq': "Correction Max Freq (Hz)",
        'max_boost': "Max Boost (dB)",
        'boost_help': f"Max safe boost: {MAX_SAFE_BOOST} dB",
        'hpf': "High Pass Filter (Protection)",
        'hpf_enable': "Enable",
        'hpf_freq': "HPF Freq (Hz)",
        'hpf_freq_help': "Cuts frequencies below this point. Protects woofers.",
        'hpf_slope': "HPF Slope (dB/oct)",
        'xo_freq': "Freq (Hz)",
        'xo_slope': "Slope (dB/oct)",
        'xo_help': "Linearizes existing crossovers (Linear Phase only).",
        'phase_lin': "(Phase Linearization)",
        'grp_settings': "Filter Settings",
        'stat_reading': "Reading files...",
        'err_missing_file': "Error: No files found.",
        'err_file_not_found': "Error: Local file not found:",
        'err_parse': "Error: Could not parse measurement files.",
        'err_upload_custom': "Error: 'Upload Custom...' selected but no file uploaded.",
        'err_inv_custom': "Error: Invalid Custom House Curve file.",
        'stat_calc': "Calculating filters...",
        'saved': "Saved:",
        'saved_plot': "Saved Plot:",
        'stat_plot': "Generating and saving plots...",
        'err_plot': "Could not generate plots.",
        'err_plot_fail': "Failed to generate plots:",
        'stat_done': "Done!",
        'done_msg': "Analysis Complete. You can close this window.",
        'tab_l': "Left Channel",
        'tab_r': "Right Channel",
        'title_plot': "Frequency Response",
        'legend_orig': "Original",
        'legend_pred': "Predicted",
        'title_phase': "Phase Response",
        'legend_orig_sm': "Original (Smoothed)",
        'legend_pred_dr': "Predicted"
    },
    'fi': {
        'title': "CamillaFIR",
        'subtitle': "Tekijät: VilhoValittu & GeminiPro",
        
        # GUIDE 1
        'guide_title': "❓ Opas: Miten valitsen Sample Raten ja Taps-määrän?",
        'guide_rule': "Nyrkkisääntö: Kun Sample Rate (näytteenottotaajuus) kaksinkertaistuu, Taps-määrän tulisi myös kaksinkertaistua.",
        'guide_formula': "FIR-filtterin tarkkuus bassopäässä (alle 100 Hz) riippuu suoraan suhteesta: Sample Rate / Taps.",
        'guide_desc': "Jos nostat Sample Raten korkeaksi (esim. 192 kHz) mutta pidät Taps-määrän pienenä (65k), suotimen resoluutio heikkenee ja bassokorjaus muuttuu epätarkaksi.",
        'guide_rec': "Suositellut Taps-arvot:",
        'guide_note': "Huom: Suurempi Taps-arvo aiheuttaa enemmän viivettä (latency), mutta äänenlaadullisesti se on tarkempi.",
        
        # GUIDE 2
        'guide_ft_title': "❓ Opas: Linear Phase vai Minimum Phase?",
        'guide_ft_lin_h': "1. Linear Phase (Oletus, Audiofiili)",
        'guide_ft_lin_desc': "• Korjaa vaihevirheet (esim. oikaisee jakosuotimet).\n• Suuri viive.\n• Raskas prosessorille.",
        'guide_ft_min_h': "2. Minimum Phase (Nollaviive, Kevyt)",
        'guide_ft_min_desc': "• Nollaviive (Pelit/TV/Live).\n• Ei Pre-ringing -ilmiötä.\n• Erittäin kevyt prosessorille.",
        'guide_ft_rec': "Suositus: Linear Phase musiikille. Minimum Phase leffoille/peleille tai jos DSD pätkii.",

        # GUIDE 3 (NEW)
        'guide_fdw_title': "❓ Opas: Miten valitsen FDW-arvon (Cycles)?",
        'guide_fdw_desc': "FDW (Frequency Dependent Windowing) määrittää, kuinka paljon huoneen kaikua otetaan mukaan korjaukseen vs. kaiuttimen suora ääni.",
        'guide_fdw_low': "• Pieni arvo (3 - 6): Aggressiivinen. Poistaa huonekaiut lähes kokonaan. Ääni on 'kuiva' ja erotteleva. Sopii lähikenttäkuunteluun.",
        'guide_fdw_mid': "• Vakio (15): SUOSITUS. Hyvä tasapaino. Korjaa bassomoodit, mutta säilyttää diskantin luonnollisen ilmavuuden.",
        'guide_fdw_high': "• Suuri arvo (30 - 100): Hellävarainen. Ottaa huoneen heijastukset mukaan. Korjaa 'tehovastetta'. Sopii elävään akustointiin.",

        'grp_files': "Tiedostot (Lataa TAI käytä polkua)",
        'upload_l': "Lataa Mittaus L (.txt)",
        'upload_r': "Lataa Mittaus Oikea (.txt)",
        'path_l': "TAI Paikallinen polku L",
        'path_r': "TAI Paikallinen polku R",
        'path_help': "Liitä koko polku (esim. C:\\Mittaukset\\L.txt) tallentaaksesi sijainnin.",
        'ph_l': "Valitse vasemman kanavan mittaus",
        'ph_r': "Valitse oikean kanavan mittaus",
        'fmt': "Tiedostomuoto",
        'layout': "Kanava-asettelu",
        'layout_mono': "Mono (Erilliset tiedostot)",
        'layout_stereo': "Stereo (Yksi tiedosto)",
        'fs': "Näytteenottotaajuus (Hz)",
        'taps': "Taps (Pituus)",
        'taps_help': "Suotimen pituus. Suurempi luku = parempi bassoresoluutio, mutta raskaampi prosessorille.",
        'filter_type': "Suotimen Tyyppi (Latency)",
        'ft_linear': "Linear Phase (Vaihekorjaus, Suuri viive)",
        'ft_min': "Minimum Phase (Ei vaihekorjausta, 0ms viive)",
        'ft_help': "Linear: Täydellinen vaihetoisto, mutta viivettä. Minimum: Salamannopea, kevyt (paras raskaaseen DSD-käyttöön).",
        'gain': "Vahvistus (dB)",
        'gain_help': "Säätää suotimen tasoa. Käytä esim -3.0dB jos korostat taajuuksia.",
        'smooth_type': "Silotustyyppi",
        'smooth_std': "Vakio 1/48 okt",
        'smooth_psy': "Psykoakustinen",
        'fdw': "FDW Jaksot (Cycles)",
        'fdw_help': "Vaiheikkunan pituus. 15 = Suositus. 6 = Aggressiivinen (Vain suora ääni).",
        'hc_mode': "Tavoitevaste (House Curve)",
        'hc_mode_builtin': "Sisäänrakennettu ('not Dr Toole')",
        'hc_mode_upload': "Lataa oma tiedosto...",
        'hc_custom': "Oma tavoitevaste (Valinnainen)",
        'hc_custom_help': "Pakollinen, jos yllä on valittu 'Lataa oma tiedosto...'.",
        'corr_mode': "Korjaustila",
        'enable_corr': "Ota taajuusvasteen korjaus käyttöön",
        'min_freq': "Korjauksen alaraja (Hz)",
        'max_freq': "Korjauksen yläraja (Hz)",
        'max_boost': "Maksimikorostus (dB)",
        'boost_help': f"Turvaraja: {MAX_SAFE_BOOST} dB",
        'hpf': "Ylipäästösuodin (Subsonic-suoja)",
        'hpf_enable': "Käytä",
        'hpf_freq': "HPF Taajuus (Hz)",
        'hpf_freq_help': "Leikkaa matalimmat bassot (esim. 20Hz).",
        'hpf_slope': "HPF Jyrkkyys (dB/oct)",
        'xo_freq': "Taajuus (Hz)",
        'xo_slope': "Jyrkkyys (dB/oct)",
        'xo_help': "Syötä nykyiset jakotaajuudet (Vaikuttaa vain Linear Phase -tilassa).",
        'phase_lin': "(Vaiheen linearisointi)",
        'grp_settings': "Suotimen asetukset",
        'stat_reading': "Luetaan tiedostoja...",
        'err_missing_file': "Virhe: Tiedostoja ei löytynyt.",
        'err_file_not_found': "Virhe: Paikallista tiedostoa ei löydy:",
        'err_parse': "Virhe: Mittaustiedostojen luku epäonnistui.",
        'err_upload_custom': "Virhe: Valitsit 'Lataa oma...' mutta tiedosto puuttuu.",
        'err_inv_custom': "Virhe: Viallinen tavoitevaste-tiedosto.",
        'stat_calc': "Lasketaan suotimia...",
        'saved': "Tallennettu:",
        'saved_plot': "Tallennettu kuva:",
        'stat_plot': "Piirretään ja tallennetaan kuvaajia...",
        'err_plot': "Kuvaajia ei voitu luoda.",
        'err_plot_fail': "Virhe piirrossa:",
        'stat_done': "Valmis!",
        'done_msg': "Analyysi valmis. Voit sulkea tämän ikkunan.",
        'tab_l': "Vasen kanava",
        'tab_r': "Oikea kanava",
        'title_plot': "Taajuusvaste",
        'legend_orig': "Alkuperäinen",
        'legend_pred': "Ennuste",
        'title_phase': "Vaihevaste",
        'legend_orig_sm': "Alkuperäinen (Silotettu)",
        'legend_pred_dr': "Ennuste"
    }
}

def t(key):
    return TRANSLATIONS[CURRENT_LANG].get(key, key)

# --- HELPER: STATUS & UI ---
def update_status(msg):
    with use_scope('status_area', clear=True):
        put_text(msg).style('font-weight: bold; color: #4CAF50;')

def put_guide():
    """Renders the collapsible guides."""
    # GUIDE 1: TAPS
    content_taps = [
        put_markdown(f"**{t('guide_rule')}**"),
        put_markdown(f"`{t('guide_formula')}`"),
        put_text(t('guide_desc')),
        put_markdown(f"**{t('guide_rec')}**"),
        put_markdown("""
        * 44.1 / 48 kHz: **65 536 taps**
        * 88.2 / 96 kHz: **131 072 taps**
        * 176.4 / 192 kHz: **262 144 taps**
        * 352.8 / 384 kHz: **524 288 taps**
        """),
        put_text(t('guide_note')).style('font-style: italic;')
    ]
    put_collapse(t('guide_title'), content_taps)

    # GUIDE 2: FILTER TYPE
    content_ft = [
        put_markdown(f"**{t('guide_ft_lin_h')}**"),
        put_text(t('guide_ft_lin_desc')),
        put_markdown("---"),
        put_markdown(f"**{t('guide_ft_min_h')}**"),
        put_text(t('guide_ft_min_desc')),
        put_markdown("---"),
        put_markdown(f"_{t('guide_ft_rec')}_").style('font-weight: bold;')
    ]
    put_collapse(t('guide_ft_title'), content_ft)

    # GUIDE 3: FDW (NEW)
    content_fdw = [
        put_text(t('guide_fdw_desc')),
        put_markdown(f"**{t('guide_fdw_mid')}**").style('color: #4CAF50;'), # Highlight Recommended
        put_text(t('guide_fdw_low')),
        put_text(t('guide_fdw_high'))
    ]
    put_collapse(t('guide_fdw_title'), content_fdw)

# --- CONFIG MANAGEMENT ---
def load_config():
    default_conf = {
        'fmt': 'WAV', 'layout': 'Stereo (Single file)', 'fs': 48000, 'taps': 65536,
        'filter_type': t('ft_linear'),
        'gain': 0.0, 'hc_mode': t('hc_mode_builtin'), 
        'mag_correct': [t('enable_corr')],
        'smoothing_type': t('smooth_psy'),
        'fdw_cycles': 15,
        'hc_min': 10, 'hc_max': 200, 'max_boost': 5.0,
        'hpf_enable': [], 'hpf_freq': None, 'hpf_slope': 24,
        'local_path_l': '', 'local_path_r': '',
        'xo1_f': None, 'xo1_s': 12, 'xo2_f': None, 'xo2_s': 12,
        'xo3_f': None, 'xo3_s': 12, 'xo4_f': None, 'xo4_s': 12, 'xo5_f': None, 'xo5_s': 12
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved_conf = json.load(f)
                default_conf.update(saved_conf)
        except: pass
    return default_conf

def save_config(data):
    save_data = data.copy()
    for key in ['file_l', 'file_r', 'hc_custom_file']:
        if key in save_data: del save_data[key]
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(save_data, f, indent=4)
    except Exception as e: print(f"Failed to save config: {e}")

# --- DATA PARSING ---
def get_default_house_curve():
    freqs = np.array([20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 20000.0])
    mags = np.array([6.0, 5.9, 5.8, 5.6, 5.3, 4.9, 4.3, 3.5, 2.5, 1.4, 0.4, 0.0, -4.0])
    return freqs, mags

def parse_measurements_from_bytes(file_content):
    try:
        f = io.BytesIO(file_content)
        data = np.loadtxt(f, comments=['#', '*'])
        if data.shape[1] < 3: return None, None, None
        return data[:, 0], data[:, 1], data[:, 2] 
    except: return None, None, None

def parse_measurements_from_path(filepath):
    try:
        if not os.path.exists(filepath): return None, None, None
        data = np.loadtxt(filepath, comments=['#', '*'])
        if data.shape[1] < 3: return None, None, None
        return data[:, 0], data[:, 1], data[:, 2]
    except: return None, None, None

def parse_house_curve_from_bytes(file_content):
    try:
        f = io.BytesIO(file_content)
        data = np.loadtxt(f, comments=['#', '*'])
        return data[:, 0], data[:, 1]
    except: return None, None

# --- PRO DSP FUNCTIONS ---
def calculate_minimum_phase(mags_lin_fft):
    n_fft = (len(mags_lin_fft) - 1) * 2
    log_mag = np.log(np.maximum(mags_lin_fft, 1e-10))
    cepstrum = scipy.fft.irfft(log_mag, n=n_fft)
    window = np.zeros_like(cepstrum)
    window[0] = 1.0
    window[1:n_fft//2] = 2.0
    window[n_fft//2] = 1.0
    cepstrum_mp = cepstrum * window
    min_phase_complex = scipy.fft.rfft(cepstrum_mp)
    return np.angle(min_phase_complex)

def psychoacoustic_smoothing(freqs, mags, oct_bw=1/3.0):
    mags_heavy, _ = apply_smoothing_std(freqs, mags, np.zeros_like(mags), oct_bw)
    mags_light, _ = apply_smoothing_std(freqs, mags, np.zeros_like(mags), 1/12.0)
    return np.maximum(mags_heavy, mags_light)

def apply_fdw_smoothing(freqs, phases, cycles):
    safe_cycles = max(cycles, 1.0)
    phase_u = np.unwrap(np.deg2rad(phases))
    oct_width = 2.0 / safe_cycles
    dummy_mags = np.zeros_like(freqs)
    _, smoothed_phase_deg = apply_smoothing_std(freqs, dummy_mags, np.rad2deg(phase_u), oct_width)
    return np.deg2rad(smoothed_phase_deg)

def apply_smoothing_std(freqs, mags, phases, octave_fraction=1.0):
    f_min = max(freqs[0], 1.0)
    f_max = freqs[-1]
    points_per_octave = 96
    num_points = int(np.log2(f_max / f_min) * points_per_octave)
    if num_points < 10: num_points = 10
    log_freqs = np.geomspace(f_min, f_max, num_points)
    log_mags = np.interp(log_freqs, freqs, mags)
    phase_unwrap = np.unwrap(np.deg2rad(phases))
    log_phases = np.interp(log_freqs, freqs, phase_unwrap)
    window_size = int(points_per_octave * octave_fraction)
    if window_size < 1: window_size = 1
    window = np.ones(window_size) / window_size
    pad_len = window_size // 2
    m_padded = np.pad(log_mags, (pad_len, pad_len), mode='edge')
    p_padded = np.pad(log_phases, (pad_len, pad_len), mode='edge')
    if pad_len > 0:
        sm_mags = np.convolve(m_padded, window, mode='same')[pad_len:-pad_len]
        sm_phases = np.convolve(p_padded, window, mode='same')[pad_len:-pad_len]
    else:
        sm_mags = np.convolve(m_padded, window, mode='same')
        sm_phases = np.convolve(p_padded, window, mode='same')
    return np.interp(freqs, log_freqs, sm_mags), np.rad2deg(np.interp(freqs, log_freqs, sm_phases))

def calculate_theoretical_phase(freq_axis, crossovers):
    if not crossovers: return np.zeros_like(freq_axis)
    total_phase_rad = np.zeros_like(freq_axis)
    for xo in crossovers:
        b, a = scipy.signal.butter(xo['order'], 2 * np.pi * xo['freq'], btype='low', analog=True)
        w, h = scipy.signal.freqs(b, a, worN=2 * np.pi * freq_axis)
        total_phase_rad += np.unwrap(np.angle(h))
    return total_phase_rad

def interpolate_response(input_freqs, input_values, target_freqs):
    return np.interp(target_freqs, input_freqs, input_values)

def find_zero_crossing_raw(freqs, phases, search_min=20, search_max=1000):
    mask = (freqs >= search_min) & (freqs <= search_max)
    f_sub = freqs[mask]
    p_sub = phases[mask]
    if len(f_sub) < 2: return search_min
    for i in range(len(p_sub) - 1):
        if np.sign(p_sub[i]) != np.sign(p_sub[i+1]):
            if abs(p_sub[i] - p_sub[i+1]) < 90.0: return f_sub[i]
    return f_sub[np.argmin(np.abs(p_sub))]

def find_closest_to_zero_raw(freqs, phases, search_min=800, search_max=2000):
    mask = (freqs >= search_min) & (freqs <= search_max)
    f_sub = freqs[mask]
    p_sub = phases[mask]
    if len(f_sub) == 0: return search_max
    return f_sub[np.argmin(np.abs(p_sub))]

def generate_filter(freqs, raw_mags, raw_phases, crossovers, 
                    phase_c_min, phase_c_max, mag_c_min, mag_c_max,
                    house_freqs, house_mags, fs, num_taps, fine_phase_limit, 
                    max_boost_db, global_gain_db, hpf_settings, enable_mag_correction,
                    smoothing_type='Standard', fdw_cycles=15, is_min_phase=False):
    
    n_fft = num_taps if num_taps % 2 != 0 else num_taps + 1
    nyquist = fs / 2.0
    freq_axis = np.linspace(0, nyquist, n_fft // 2 + 1)
    
    if smoothing_type == t('smooth_psy'):
        smoothed_mags = psychoacoustic_smoothing(freqs, raw_mags)
    else:
        smoothed_mags, _ = apply_smoothing_std(freqs, raw_mags, raw_phases, 1/48.0)
    
    meas_mags = interpolate_response(freqs, smoothed_mags, freq_axis)
    target_mags = np.zeros_like(freq_axis)
    use_house_curve = (house_freqs is not None and house_mags is not None)
    if use_house_curve:
        hc_interp = interpolate_response(house_freqs, house_mags, freq_axis)
        idx_align = np.argmin(np.abs(freq_axis - mag_c_max))
        target_mags = hc_interp + (meas_mags[idx_align] - hc_interp[idx_align])
    
    hpf_complex = np.ones_like(freq_axis, dtype=complex)
    if hpf_settings and hpf_settings['enabled']:
        b, a = scipy.signal.butter(hpf_settings['order'], 2 * np.pi * hpf_settings['freq'], btype='high', analog=True)
        w, h = scipy.signal.freqs(b, a, worN=2 * np.pi * freq_axis)
        hpf_complex = h
        hpf_complex[0] = 0.0

    gain_linear = np.ones_like(freq_axis)
    for i, f in enumerate(freq_axis):
        if f > 0:
            g_db = 0.0
            if enable_mag_correction and use_house_curve and (mag_c_min <= f <= mag_c_max):
                safe_max_boost = min(max_boost_db, MAX_SAFE_BOOST)
                g_db = np.clip(target_mags[i] - meas_mags[i], -15.0, safe_max_boost)
            g_db += global_gain_db
            gain_linear[i] = 10.0 ** (g_db / 20.0)

    if is_min_phase:
        total_mag_response = gain_linear * np.abs(hpf_complex)
        filt_min_phase_rad = calculate_minimum_phase(total_mag_response)
        final_complex = total_mag_response * np.exp(1j * filt_min_phase_rad)
        impulse = scipy.fft.irfft(final_complex, n=n_fft)
        window = np.ones(n_fft)
        fade_len = int(n_fft * 0.1)
        if fade_len > 0:
            fade_curve = scipy.signal.windows.hann(2 * fade_len)[fade_len:]
            window[-fade_len:] = fade_curve
        impulse_final = impulse * window 
        return impulse_final, 0.0, 0.0
    else:
        meas_min_phase_rad = calculate_minimum_phase(10**(meas_mags/20.0))
        meas_phase_rad_raw = np.deg2rad(interpolate_response(freqs, raw_phases, freq_axis))
        meas_phase_rad_unwrapped = np.unwrap(meas_phase_rad_raw)
        excess_phase_rad = meas_phase_rad_unwrapped - meas_min_phase_rad
        excess_phase_deg = np.rad2deg(excess_phase_rad)
        excess_phase_fdw_rad = apply_fdw_smoothing(freq_axis, excess_phase_deg, fdw_cycles)
        theoretical_xo_phase = calculate_theoretical_phase(freq_axis, crossovers)
        
        phase_corr_rad = np.zeros_like(freq_axis)
        limit_rad = np.deg2rad(fine_phase_limit)
        
        for i, f in enumerate(freq_axis):
            if f > 0:
                fine_correction = 0.0
                if phase_c_min <= f <= phase_c_max:
                    fine_correction = np.clip(-excess_phase_fdw_rad[i], -limit_rad, limit_rad)
                phase_corr_rad[i] = -theoretical_xo_phase[i] + fine_correction

        correction_complex = gain_linear * np.exp(1j * phase_corr_rad)
        final_complex = correction_complex * hpf_complex
        final_phase_unwrapped = np.unwrap(np.angle(final_complex))
        gd_samples = -np.gradient(final_phase_unwrapped) * (len(freq_axis) / np.pi) 
        gd_ms = (gd_samples / fs) * 1000.0
        mask = (freq_axis >= 20) & (freq_axis <= 20000)
        if np.sum(mask) > 0:
            valid_gd = gd_ms[mask]
            gd_min, gd_max = np.min(valid_gd), np.max(valid_gd)
        else:
            gd_min, gd_max = 0.0, 0.0

        impulse = scipy.fft.irfft(final_complex, n=n_fft)
        window = scipy.signal.windows.blackman(n_fft)
        return np.roll(impulse, n_fft // 2) * window, gd_min, gd_max

def save_summary(filename, settings, l_stats, r_stats):
    with open(filename, 'w') as f:
        f.write(f"=== {PROGRAM_NAME} - Filter Generation Summary ===\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("--- Settings ---\n")
        for key, val in settings.items():
            if 'file' not in key:
                f.write(f"{key}: {val}\n")
        f.write("\n--- Analysis (20Hz - 20kHz) ---\n")
        f.write(f"Left Channel GD: {l_stats['gd_min']:.2f} ms to {l_stats['gd_max']:.2f} ms\n")
        f.write(f"Right Channel GD: {r_stats['gd_min']:.2f} ms to {r_stats['gd_max']:.2f} ms\n")

def generate_prediction_plot(orig_freqs, orig_mags, orig_phases, filt_ir, fs, title, save_filename=None):
    try:
        n_fft = len(filt_ir)
        freq_axis_lin = scipy.fft.rfftfreq(n_fft, d=1/fs)
        h_filt_complex = scipy.fft.rfft(filt_ir)
        orig_mags_lin = np.interp(freq_axis_lin, orig_freqs, orig_mags)
        orig_phases_lin = np.interp(freq_axis_lin, orig_freqs, orig_phases)
        orig_complex_lin = 10**(orig_mags_lin/20.0) * np.exp(1j * np.deg2rad(orig_phases_lin))
        total_complex = orig_complex_lin * h_filt_complex
        
        ir_total = scipy.fft.irfft(total_complex, n=n_fft)
        peak_idx = np.argmax(np.abs(ir_total))
        ir_centered = np.roll(ir_total, -peak_idx)
        total_complex_centered = scipy.fft.rfft(ir_centered)
        
        final_mags_lin = 20 * np.log10(np.abs(total_complex) + 1e-12)
        final_phases_rad = np.angle(total_complex_centered)
        final_phases_deg = np.rad2deg(final_phases_rad)
        
        final_mags_plot = np.interp(orig_freqs, freq_axis_lin, final_mags_lin)
        final_phases_plot = np.interp(orig_freqs, freq_axis_lin, final_phases_deg)
        
        _, plot_phase_orig = apply_smoothing_std(orig_freqs, orig_mags, orig_phases, 1.0)
        _, plot_phase_pred = apply_smoothing_std(orig_freqs, final_mags_plot, final_phases_plot, 1.0)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ticks_val = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        ticks_lab = ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k']
        x_min = max(orig_freqs[0], 10)
        x_max = orig_freqs[-1]

        ax1.semilogx(orig_freqs, orig_mags, label=t('legend_orig'), color='blue', alpha=0.5)
        ax1.semilogx(orig_freqs, final_mags_plot, label=t('legend_pred'), color='orange', linewidth=2)
        ax1.set_title(f"{title} - {t('title_plot')}")
        ax1.set_ylabel("Amplitude (dB)")
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend()
        ax1.set_xlim(x_min, x_max)
        ax1.set_xticks(ticks_val)
        ax1.set_xticklabels(ticks_lab)
        
        plot_phase_orig_wrapped = (plot_phase_orig + 180) % 360 - 180
        plot_phase_pred_wrapped = (plot_phase_pred + 180) % 360 - 180

        ax2.semilogx(orig_freqs, plot_phase_orig_wrapped, label=t('legend_orig_sm'), color='blue', alpha=0.5)
        ax2.semilogx(orig_freqs, plot_phase_pred_wrapped, label=t('legend_pred_dr'), color='orange', linewidth=2)
        
        ax2.set_title(t('title_phase'))
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (deg)")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.set_xlim(x_min, x_max)
        ax2.set_xticks(ticks_val)
        ax2.set_xticklabels(ticks_lab)
        ax2.set_ylim(-360, 360)
        ax2.set_yticks(np.arange(-360, 361, 90))
        ax2.legend()
        
        plt.tight_layout()
        if save_filename: fig.savefig(save_filename, format='png')
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        print(f"Plotting Error: {e}")
        return None

# --- MAIN GUI APP ---
@config(theme="dark")
def main():
    put_markdown(f"# 🎛️ {PROGRAM_NAME} {VERSION}")
    put_markdown(f"### {t('subtitle')}")

    put_guide()

    defaults = load_config()
    hc_options = [t('hc_mode_builtin'), t('hc_mode_upload')]
    fs_options = [44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000]
    taps_options = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]

    inputs_list = [
        file_upload(t('upload_l'), name='file_l', accept='.txt', placeholder=t('ph_l')),
        input(t('path_l'), value=defaults.get('local_path_l', ''), name='local_path_l', help_text=t('path_help')),
        
        file_upload(t('upload_r'), name='file_r', accept='.txt', placeholder=t('ph_r')),
        input(t('path_r'), value=defaults.get('local_path_r', ''), name='local_path_r', help_text=t('path_help')),

        select(t('fmt'), options=['WAV', 'CSV'], value=defaults['fmt'], name='fmt'),
        radio(t('layout'), options=[t('layout_mono'), t('layout_stereo')], value=defaults['layout'], name='layout'),
        select(t('fs'), options=fs_options, value=defaults['fs'], name='fs', type=NUMBER),
        select(t('taps'), options=taps_options, value=defaults['taps'], name='taps', type=NUMBER, help_text=t('taps_help')),
        
        radio(t('filter_type'), options=[t('ft_linear'), t('ft_min')], value=defaults.get('filter_type', t('ft_linear')), name='filter_type', help_text=t('ft_help')),

        input(t('gain'), value=defaults['gain'], type=FLOAT, name='gain', help_text=t('gain_help')),
        
        select(t('smooth_type'), options=[t('smooth_std'), t('smooth_psy')], value=defaults.get('smoothing_type', t('smooth_psy')), name='smoothing_type'),
        input(t('fdw'), value=defaults.get('fdw_cycles', 15), type=FLOAT, name='fdw_cycles', help_text=t('fdw_help')),

        select(t('hc_mode'), options=hc_options, value=defaults['hc_mode'], name='hc_mode'),
        file_upload(t('hc_custom'), name='hc_custom_file', accept='.txt', help_text=t('hc_custom_help')),
        
        checkbox(t('corr_mode'), options=[t('enable_corr')], value=defaults['mag_correct'], name='mag_correct'),
        input(t('min_freq'), value=defaults['hc_min'], type=FLOAT, name='hc_min'),
        input(t('max_freq'), value=defaults['hc_max'], type=FLOAT, name='hc_max'),
        input(t('max_boost'), value=defaults['max_boost'], type=FLOAT, name='max_boost', help_text=t('boost_help')),
        
        checkbox(label=t('hpf'), options=[t('hpf_enable')], value=defaults['hpf_enable'], name='hpf_enable'),
        input(t('hpf_freq'), value=defaults['hpf_freq'], type=FLOAT, name='hpf_freq', placeholder="e.g. 20", help_text=t('hpf_freq_help')), 
        select(t('hpf_slope'), options=[6, 12, 18, 24, 36, 48], value=defaults['hpf_slope'], type=NUMBER, name='hpf_slope')
    ]

    slope_opts = [6, 12, 18, 24, 36, 48]
    for i in range(1, 6):
        label_txt = f"XO {i} {t('xo_freq')}"
        if i == 1: label_txt += f" {t('phase_lin')}"
        if i == 1:
            inputs_list.append(input(label_txt, value=defaults[f'xo{i}_f'], name=f'xo{i}_f', type=FLOAT, placeholder="Leave empty if unused", help_text=t('xo_help')))
        else:
            inputs_list.append(input(label_txt, value=defaults[f'xo{i}_f'], name=f'xo{i}_f', type=FLOAT, placeholder="Leave empty if unused"))
        inputs_list.append(select(f"XO {i} {t('xo_slope')}", options=slope_opts, value=defaults[f'xo{i}_s'], name=f'xo{i}_s'))

    data = input_group(t('grp_settings'), inputs_list)
    save_config(data)

    clear()
    put_markdown(f"# 🎛️ {PROGRAM_NAME} {VERSION}")
    put_markdown(f"### {t('subtitle')}")
    put_guide() 

    try:
        put_processbar('bar')
        put_scope('status_area')
        set_processbar('bar', 0.1)
        update_status(t('stat_reading'))

        freqs_l, mags_l, phases_l = None, None, None
        freqs_r, mags_r, phases_r = None, None, None

        if data['file_l']:
            freqs_l, mags_l, phases_l = parse_measurements_from_bytes(data['file_l']['content'])
        elif data['local_path_l']:
            freqs_l, mags_l, phases_l = parse_measurements_from_path(data['local_path_l'])
            if freqs_l is None:
                put_error(f"{t('err_file_not_found')} {data['local_path_l']}")
                return

        if data['file_r']:
            freqs_r, mags_r, phases_r = parse_measurements_from_bytes(data['file_r']['content'])
        elif data['local_path_r']:
            freqs_r, mags_r, phases_r = parse_measurements_from_path(data['local_path_r'])
            if freqs_r is None:
                put_error(f"{t('err_file_not_found')} {data['local_path_r']}")
                return

        if freqs_l is None or freqs_r is None:
            put_error(t('err_missing_file'))
            return

        hc_freqs, hc_mags = None, None
        if data['hc_mode'] == t('hc_mode_builtin'):
            hc_freqs, hc_mags = get_default_house_curve()
        elif data['hc_mode'] == t('hc_mode_upload'):
            if not data['hc_custom_file']:
                put_error(t('err_upload_custom'))
                return
            hc_freqs, hc_mags = parse_house_curve_from_bytes(data['hc_custom_file']['content'])
            if hc_freqs is None:
                put_error(t('err_inv_custom'))
                return

        if hc_freqs is None:
            data['hc_min'], data['hc_max'] = 0, 0

        set_processbar('bar', 0.4)
        update_status(t('stat_calc'))
        
        do_mag_correct = t('enable_corr') in data['mag_correct']
        crossovers = []
        for i in range(1, 6):
            if data[f'xo{i}_f']:
                crossovers.append({'freq': float(data[f'xo{i}_f']), 'order': int(data[f'xo{i}_s']) // 6})

        hpf_settings = None
        if t('hpf_enable') in data['hpf_enable'] and data['hpf_freq']:
            hpf_settings = {'enabled': True, 'freq': data['hpf_freq'], 'order': data['hpf_slope'] // 6}

        c_min = (find_zero_crossing_raw(freqs_l, phases_l) + find_zero_crossing_raw(freqs_r, phases_r)) / 2
        c_max = (find_closest_to_zero_raw(freqs_l, phases_l) + find_closest_to_zero_raw(freqs_r, phases_r)) / 2
        
        smoothing_mode = data.get('smoothing_type', t('smooth_psy'))
        if t('smooth_psy') in smoothing_mode: smoothing_mode = t('smooth_psy')
        else: smoothing_mode = 'Standard'
        
        fdw = float(data.get('fdw_cycles', 15))
        is_min_phase = (data.get('filter_type') == t('ft_min'))

        l_imp, l_min, l_max = generate_filter(
            freqs_l, mags_l, phases_l, crossovers, c_min, c_max,
            data['hc_min'], data['hc_max'], hc_freqs, hc_mags, data['fs'], data['taps'],
            FINE_TUNE_LIMIT, data['max_boost'], data['gain'], hpf_settings, do_mag_correct,
            smoothing_mode, fdw, is_min_phase
        )
        r_imp, r_min, r_max = generate_filter(
            freqs_r, mags_r, phases_r, crossovers, c_min, c_max,
            data['hc_min'], data['hc_max'], hc_freqs, hc_mags, data['fs'], data['taps'],
            FINE_TUNE_LIMIT, data['max_boost'], data['gain'], hpf_settings, do_mag_correct,
            smoothing_mode, fdw, is_min_phase
        )

        set_processbar('bar', 0.8)

        now = datetime.now()
        ts = now.strftime('%d%m%y_%H%M')
        ext = ".wav" if data['fmt'] == 'WAV' else ".csv"
        is_stereo = 'Stereo' in data['layout'] or 'Yksi tiedosto' in data['layout']
        
        fn_l = f"L_corr_{data['fs']}Hz_{ts}{ext}"
        fn_r = f"R_corr_{data['fs']}Hz_{ts}{ext}"
        fn_s = f"Stereo_corr_{data['fs']}Hz_{ts}{ext}"
        fn_sum = f"Summary_{ts}.txt"
        
        fn_plot_l = f"L_plot_{data['fs']}Hz_{ts}.png"
        fn_plot_r = f"R_plot_{data['fs']}Hz_{ts}.png"

        if data['fmt'] == 'WAV':
            if is_stereo:
                scipy.io.wavfile.write(fn_s, data['fs'], np.column_stack((l_imp, r_imp)).astype(np.float32))
                put_success(f"{t('saved')} {fn_s}")
            else:
                scipy.io.wavfile.write(fn_l, data['fs'], l_imp.astype(np.float32))
                scipy.io.wavfile.write(fn_r, data['fs'], r_imp.astype(np.float32))
                put_success(f"{t('saved')} {fn_l}, {fn_r}")
        else:
            if is_stereo:
                np.savetxt(fn_s, np.column_stack((l_imp, r_imp)), fmt='%.18f', delimiter=' ')
                put_success(f"{t('saved')} {fn_s}")
            else:
                np.savetxt(fn_l, l_imp, fmt='%.18f')
                np.savetxt(fn_r, r_imp, fmt='%.18f')
                put_success(f"{t('saved')} {fn_l}, {fn_r}")

        settings_dict = data.copy()
        settings_dict['Magnitude Correction'] = "Enabled" if do_mag_correct else "Disabled"
        settings_dict['Crossovers'] = str(crossovers)
        settings_dict['Phase Range'] = f"{c_min:.0f}-{c_max:.0f} Hz"
        save_summary(fn_sum, settings_dict, {'gd_min': l_min, 'gd_max': l_max}, {'gd_min': r_min, 'gd_max': r_max})
        put_info(f"Summary saved to {fn_sum}")
        
        set_processbar('bar', 0.9)
        update_status(t('stat_plot'))
        
        try:
            img_l = generate_prediction_plot(freqs_l, mags_l, phases_l, l_imp, data['fs'], t('tab_l'), fn_plot_l)
            img_r = generate_prediction_plot(freqs_r, mags_r, phases_r, r_imp, data['fs'], t('tab_r'), fn_plot_r)
            
            put_success(f"{t('saved_plot')} {fn_plot_l}, {fn_plot_r}")
            
            if img_l and img_r:
                put_tabs([
                    {'title': t('tab_l'), 'content': put_image(img_l)},
                    {'title': t('tab_r'), 'content': put_image(img_r)}
                ])
            else:
                put_error(t('err_plot'))
        except Exception as e:
            put_error(f"{t('err_plot_fail')} {e}")
            
        set_processbar('bar', 1.0)
        update_status(t('stat_done'))
        put_markdown(f"### {t('done_msg')}")
        
    except Exception as e:
        put_error(f"Unexpected Error: {e}")
    finally:
        pass

if __name__ == '__main__':
    start_server(main, port=8080, debug=True, auto_open_webbrowser=True)
