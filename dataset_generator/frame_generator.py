import numpy as np
import os

# --- CONFIGURAZIONE ---
INPUT_FILE = "000003.bin"
OUTPUT_DIR = "generated_bounded_data"
NUM_FRAMES = 100
TARGET_MEAN = 2000000  # Media punti per frame
VARIANCE_PERCENT = 0.10  # Variazione casuale +/- 10%

# --- I TUOI BOUNDS ---
# (Questi devono matchare il tuo params.hpp)
MAX_X = 50.0
MAX_Y = 50.0
MAX_Z = 10.0
MIN_X = -50.0
MIN_Y = -50.0
MIN_Z = -10.0

# Movimento simulato
SPEED_X = 1.5  # Metri per frame (più veloce per vedere l'effetto)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Caricamento o Creazione Dummy
if not os.path.exists(INPUT_FILE):
    print("File input non trovato, creo dati random...")
    # Creiamo punti random già dentro il box per sicurezza
    dummy_points = 50000
    dummy = np.zeros((dummy_points, 4), dtype=np.float32)
    dummy[:, 0] = np.random.uniform(MIN_X, MAX_X, dummy_points)  # X
    dummy[:, 1] = np.random.uniform(MIN_Y, MAX_Y, dummy_points)  # Y
    dummy[:, 2] = np.random.uniform(MIN_Z, MAX_Z, dummy_points)  # Z
    dummy[:, 3] = 0.5  # Intensità
    base_cloud = dummy
else:
    base_cloud = np.fromfile(INPUT_FILE, dtype=np.float32).reshape(-1, 4)

base_points = base_cloud.shape[0]
print(f"Base: {base_points} punti. Bounds X: [{MIN_X}, {MAX_X}]")

# Calcoliamo le dimensioni del box per il wrapping
WIDTH_X = MAX_X - MIN_X
WIDTH_Y = MAX_Y - MIN_Y
WIDTH_Z = MAX_Z - MIN_Z

for i in range(NUM_FRAMES):

    # 2. Target casuale per testare allocazione dinamica
    sigma = TARGET_MEAN * VARIANCE_PERCENT
    current_target = int(np.random.normal(TARGET_MEAN, sigma))
    if current_target < base_points: current_target = base_points

    # 3. Generazione Nuvola (Copia + Rumore)
    points_collected = []

    num_full_copies = current_target // base_points
    remainder = current_target % base_points


    # Funzione helper per aggiungere rumore
    def get_noisy_chunk(source, size):
        chunk = source[:size].copy()
        # Rumore uniforme +/- 20cm per "spalmare" i punti nei voxel
        noise = np.random.uniform(-0.2, 0.2, size=(size, 4))
        noise[:, 3] = 0
        return chunk + noise


    # Copie intere
    for k in range(num_full_copies):
        points_collected.append(get_noisy_chunk(base_cloud, base_points))

    # Resto
    if remainder > 0:
        points_collected.append(get_noisy_chunk(base_cloud, remainder))

    final_cloud = np.vstack(points_collected).astype(np.float32)

    # 4. MOVIMENTO & WRAPPING (Effetto Pac-Man)
    # Spostiamo tutto in avanti
    move_offset = i * SPEED_X
    final_cloud[:, 0] += move_offset

    # --- LOGICA DI WRAPPING ---
    # Questa formula magica mantiene i punti sempre dentro [MIN, MAX]
    # Se X esce a 51, torna a -49.

    # 1. Normalizza a 0..Width (togliendo MIN)
    # 2. Modulo Width
    # 3. Riaggiungi MIN
    final_cloud[:, 0] = ((final_cloud[:, 0] - MIN_X) % WIDTH_X) + MIN_X
    final_cloud[:, 1] = ((final_cloud[:, 1] - MIN_Y) % WIDTH_Y) + MIN_Y

    # Per Z solitamente clippiamo e basta (non ha senso che sbuchino dal pavimento)
    np.clip(final_cloud[:, 2], MIN_Z, MAX_Z, out=final_cloud[:, 2])

    # 5. Salvataggio
    out_name = f"{i:06d}.bin"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    final_cloud.tofile(out_path)

    if i % 10 == 0:
        print(f"Frame {i}: {final_cloud.shape[0]} punti validi nel box.")

print("Finito. Tutti i punti sono garantiti dentro i bounds.")