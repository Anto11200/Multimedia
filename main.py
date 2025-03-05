import numpy as np
import librosa
import soundfile as sf
from reedsolo import RSCodec

from functions.FileToByte import * 
from functions.Steganography import *



# Configurazione Reed-Solomon
RS_ECC_SYMBOLS = 40  # Corregge fino a 20 byte danneggiati

# Load the audio file
# audio_path = "audio/me_saying_merhaba_dunya.wav"  # Change to your file path
audio_path = "audio/relaxing-guitar.wav"  # Change to your file path
y, sr = librosa.load(audio_path, sr=None)  # Load with original sampling rate

# Compute the Short-Time Fourier Transform (STFT)
D = librosa.stft(y, n_fft=1024, hop_length=512)

# Convert to magnitude and phase
magnitude, phase = np.abs(D), np.angle(D)

# Save magnitude and phase to files
np.savez('file_numpy/magnitude.npz', magnitude=magnitude)
np.savez('file_numpy/phase.npz', phase=phase)

# Load magnitude and phase as bytes
magnitude_file_bytes = file_to_bytes('file_numpy/magnitude.npz')
phase_file_bytes = file_to_bytes('file_numpy/phase.npz')

# Create payload: <LENGTH_MAGNITUDE><MAGNITUDE_BYTES><LENGTH_PHASE><PHASE_BYTES>
len_mag = len(magnitude_file_bytes).to_bytes(4, byteorder='big')  # 4 bytes for length
len_phase = len(phase_file_bytes).to_bytes(4, byteorder='big')  # 4 bytes for length

payload = bytearray(len_mag) + magnitude_file_bytes + bytearray(len_phase) + phase_file_bytes

# Durante l'embedding
rsc = RSCodec(RS_ECC_SYMBOLS)
protected_payload = rsc.encode(payload)
steganography_embed("image/paesaggio.jpg", protected_payload, "encode/output_r.jpg")

# Durante l'estrazione
# bt = steganography_extract("encode/Immagine_WhatsApp_2025-02-25_ore_17.28.56_c6cc2a35.jpg", len(payload))
bt = steganography_extract("encode/output_r.jpg", len(payload))
decoded_payload = rsc.decode(bt)[0]

# Extract length of magnitude
len_mag_r = int.from_bytes(bt[:4], byteorder='big')  # First 4 bytes are length

# Extract magnitude bytes
magnitude_file_bytes_R = bt[4:4 + len_mag_r]

# Extract length of phase
len_phase_r = int.from_bytes(bt[4 + len_mag_r:8 + len_mag_r], byteorder='big')  # Next 4 bytes are length

# Extract phase bytes
phase_file_bytes_R = bt[8 + len_mag_r:8 + len_mag_r + len_phase_r]

# Converti i byte in un array NumPy
magnitude_extracted = np.load('file_numpy/magnitude.npz')['magnitude']
phase_extracted = np.load('file_numpy/phase.npz')['phase']

magnitude_extracted = magnitude_extracted.reshape(magnitude.shape)
phase_extracted = phase_extracted.reshape(phase.shape)

# Reconstruct the signal using Inverse STFT (ISTFT)
reconstructed_signal = librosa.istft(magnitude_extracted * np.exp(1j * phase_extracted), hop_length=512)

# Save the reconstructed audio
sf.write("decode/reconstructed_w.wav", reconstructed_signal, sr)
