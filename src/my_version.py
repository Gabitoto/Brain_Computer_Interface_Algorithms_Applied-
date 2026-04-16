import numpy as np
import scipy.io as sio
import scipy.signal as signal
from mne.io import read_raw_gdf
from mne import events_from_annotations
from modules.preprocessing.filtering import get_bandpass_coef
import glob
import logging


FS = 250                  # frecuencia de muestreo (250 muestras por segundo)
N_SUBJECTS = 9
N_CHANNELS_DEFAULT = 22

# Códigos de estímulo → cada número identifica qué imaginó el sujeto
# 769=mano izquierda, 770=mano derecha, 771=pies, 772=lengua
STIM_CODES = [769, 770, 771, 772]

# Nombres de los 22 canales EEG
CHANNEL_NAMES = [
    'Fz',  'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5',  'C3',  'C1',  'Cz',  'C2',  'C4',
    'C6',  'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1',  'Pz',  'P2',  'POz'
]


def read_BCI_IV_DS2a(
    filename,
    class_labels=[1, 2, 3, 4],
    band_pass=None,
    segment_length=2,
    offset=0.5,
    debug=False,
    selected_channels=None
):
    """
    Lee y segmenta el dataset BCI Competition IV 2a.

     Esta función:
      1. Carga la cinta de audio.
      2. La filtra para quedarse solo con las frecuencias útiles.
      3. Encuentra las marcas de tiempo.
      4. Recorta fragmentos de 2 segundos alrededor de cada marca.
      5. Los etiqueta y empaqueta.

    Retorna un dict con los datos de entrenamiento y test por sujeto.
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

    # Convertir clases seleccionadas a sus códigos de estímulo reales
    # Ej: class_labels=[1,2] → stim_codes=[769, 770]
    stim_codes = [STIM_CODES[i - 1] for i in class_labels] ### ver que hace esta parte
    logging.debug(f"Códigos de estímulo usados: {stim_codes}")

    # ── Paso 1: Encontrar todos los archivos del dataset ──────────────────────
    # Analogía: buscar todas las "cajas de ingredientes" en el depósito
    train_eeg_files  = sorted(glob.glob(filename + '**/*T.gdf', recursive=True))
    test_eeg_files   = sorted(glob.glob(filename + '**/*E.gdf', recursive=True))
    train_label_files = sorted(glob.glob(filename + '**/*T.mat', recursive=True))
    test_label_files  = sorted(glob.glob(filename + '**/*E.mat', recursive=True))

    logging.debug(f"Archivos de entrenamiento encontrados: {len(train_eeg_files)}")
    logging.debug(f"Archivos de test encontrados: {len(test_eeg_files)}")

    subject_data = {}

    for subject_idx in range(N_SUBJECTS):

        logging.debug(f"\n── Procesando sujeto {subject_idx + 1} ──")

        # ── Paso 2: Cargar señales EEG crudas ────────────────────────────────
        train_raw = read_raw_gdf(train_eeg_files[subject_idx], preload=True, verbose=False)
        test_raw  = read_raw_gdf(test_eeg_files[subject_idx],  preload=True, verbose=False)

        # ── Paso 3: Seleccionar canales ───────────────────────────────────────
        # Analogía: elegir qué micrófonos del casco EEG queremos usar
        train_raw, test_raw, n_ch = _select_channels(
            train_raw, test_raw, selected_channels, N_CHANNELS_DEFAULT
        )

        # Extraer arrays numpy: shape (n_channels, n_samples)
        train_signal = train_raw.get_data()
        test_signal  = test_raw.get_data()

        # ── Paso 4: Filtrado en banda (opcional) ──────────────────────────────
        # Analogía: pasar el audio por un ecualizador para quedarse solo
        # con las frecuencias importantes (ej: 8-30 Hz para ritmos mu/beta)
        if band_pass:
            train_signal, test_signal = _apply_bandpass(
                train_signal, test_signal, band_pass, FS
            )

        # ── Paso 5: Segmentar trials de entrenamiento ─────────────────────────
        # Analogía: encontrar las marcas en la cinta y recortar clips de 2 seg
        X_train, y_train = _segment_train(
            train_raw, train_signal,
            train_label_files[subject_idx],
            stim_codes, class_labels,
            segment_length, offset, FS, n_ch
        )

        # ── Paso 6: Segmentar trials de test ──────────────────────────────────
        X_test, y_test = _segment_test(
            test_raw, test_signal,
            test_label_files[subject_idx],
            class_labels,
            segment_length, offset, FS, n_ch
        )

        subject_data[f'subject_{subject_idx + 1}'] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test':  X_test,
            'y_test':  y_test
        }

    return subject_data

# FUNCIONES AUXILIARES

def _select_channels(train_raw, test_raw, selected_channels, n_channels_default):
    """
    Selecciona un subconjunto de canales si se especifica.
    Si no, usa los primeros 22 (EEG, excluyendo EOG).

    Analogía: elegir qué instrumentos de la orquesta escuchar;
    si no especificás nada, se usan todos los de cuerda (22 EEG).
    """
    if selected_channels is None:
        # Quedarse con los 22 canales EEG (descartar los 3 EOG del final)
        eeg_ch_names = train_raw.ch_names[:n_channels_default]
        train_raw.pick_channels(eeg_ch_names)
        test_raw.pick_channels(eeg_ch_names)
        return train_raw, test_raw, n_channels_default

    # Mapear nombre amigable → nombre real en el archivo
    # Ej: 'C3' → 'EEG-C3'
    name_map = {
        friendly: real
        for friendly, real in zip(CHANNEL_NAMES, train_raw.ch_names[:22])
    }

    try:
        real_names = [name_map[ch] for ch in selected_channels]
    except KeyError as e:
        raise ValueError(
            f"Canal {e} no encontrado. Nombres válidos: {list(name_map.keys())}"
        )

    logging.debug(f"Canales seleccionados: {selected_channels} → {real_names}")
    train_raw.pick_channels(real_names)
    test_raw.pick_channels(real_names)

    return train_raw, test_raw, len(real_names)


def _apply_bandpass(train_signal, test_signal, band_pass, fs):
    """
    Aplica un filtro pasa-banda a las señales.

    Analogía: como un ecualizador de audio que corta todo lo que
    esté fuera del rango [low, high] Hz. Para BCI motor imagery,
    el rango 8-30 Hz captura los ritmos mu y beta, que son los
    que cambian cuando imaginamos movimiento.
    """
    b, a = get_bandpass_coef(N=4, low=band_pass[0], high=band_pass[1], fs=fs)
    train_filtered = signal.filtfilt(b, a, train_signal, axis=1)
    test_filtered  = signal.filtfilt(b, a, test_signal,  axis=1)
    return train_filtered, test_filtered


def _segment_train(raw, raw_signal, label_file, stim_codes, class_labels,
                   segment_length, offset, fs, n_channels):
    """
    Extrae los trials (segmentos) de entrenamiento.

    Analogía: tienes una cinta de 6 minutos. Hay 288 marcas de tiempo.
    Esta función recorta un clip de `segment_length` segundos después
    de cada marca (con un pequeño delay = offset), y le pone una etiqueta.

    Retorna:
        X: array (n_muestras, n_canales, n_trials)
        y: array (n_trials,) con las etiquetas de clase
    """
    seg_samples    = int(segment_length * fs)
    offset_samples = int(offset * fs)

    events, events_id = events_from_annotations(raw)
    # Traducir stim_codes a los índices internos de MNE
    # Ej: {769: 2, 770: 3} → queremos los eventos con valor 2 o 3
    index_codes = [events_id[str(code)] for code in stim_codes]

    # Cargar etiquetas reales desde el .mat y filtrar solo las clases pedidas
    all_labels = sio.loadmat(label_file, squeeze_me=True)['classlabel']
    labels     = np.array([l for l in all_labels if l in class_labels])

    n_trials = int(np.sum(np.isin(events[:, -1], index_codes)))
    if n_trials != len(labels):
        raise ValueError(
            f"Discrepancia: {len(labels)} etiquetas en .mat vs {n_trials} eventos en .gdf"
        )

    X = np.zeros((seg_samples, n_channels, n_trials))
    y = np.zeros((n_trials,))

    # Diccionario inverso: índice MNE → código original (ej: 2 → 769)
    idx_to_code = {v: int(k) for k, v in events_id.items()}

    trial = 0
    for event in events:
        if event[-1] in index_codes:
            start = int(event[0] + offset_samples)
            end   = start + seg_samples
            X[:, :, trial] = raw_signal[:, start:end].T
            y[trial]       = idx_to_code[event[-1]] - 768  # 769→1, 770→2, etc.
            trial += 1

    # Verificación de sanidad: ¿las etiquetas coinciden con el .mat?
    if not np.all(y == labels):
        raise ValueError("Las etiquetas extraídas no coinciden con el archivo .mat")

    logging.debug(f"  Train trials: {n_trials}, clases: {np.unique(y)}")
    return X, y


def _segment_test(raw, raw_signal, label_file, class_labels,
                  segment_length, offset, fs, n_channels):
    """
    Extrae los trials de test.

    Diferencia con train: en el archivo de test las clases están ocultas
    (código 783 = "clase desconocida"). Las etiquetas reales están en el .mat.
    Esta función recorta los clips igual que en train y luego filtra
    solo los trials que pertenecen a las clases pedidas.
    """
    seg_samples    = int(segment_length * fs)
    offset_samples = int(offset * fs)

    events, events_id = events_from_annotations(raw)
    all_labels = sio.loadmat(label_file, squeeze_me=True)['classlabel']
    n_total    = all_labels.size

    # Recortar TODOS los trials (no sabemos la clase todavía)
    X_all = np.zeros((seg_samples, n_channels, n_total))
    trial = 0
    for event in events:
        if event[-1] == events_id['783']:  # cue onset desconocido
            start = int(event[0] + offset_samples)
            end   = start + seg_samples
            X_all[:, :, trial] = raw_signal[:, start:end].T
            trial += 1

    # Ahora filtrar solo los trials de las clases pedidas
    mask   = np.isin(all_labels, class_labels)
    X_test = X_all[:, :, mask]
    y_test = all_labels[mask]

    logging.debug(f"  Test trials: {X_test.shape[2]}, clases: {np.unique(y_test)}")
    return X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    filename = './data/BCI_IV_2a/'
    data = read_BCI_IV_DS2a(
        filename,
        class_labels=[1, 2],
        band_pass=[8, 30],
        segment_length=2,
        offset=0.5,
        debug=True,
        selected_channels=['FC3', 'C5', 'C3', 'C1', 'CP3']
    )

    np.save('./data/BCI_IV_2a/segmentadas/BCI_IV_2a_right_upper_limb_8_30.npy', data)

    s1 = data['subject_1']
    print(f"X_train: {s1['X_train'].shape}")  # (500, n_ch, n_trials)
    print(f"y_train: {s1['y_train'].shape}")
    print(f"X_test:  {s1['X_test'].shape}")
    print(f"y_test:  {s1['y_test'].shape}")