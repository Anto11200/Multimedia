import cv2
import numpy as np
from reedsolo import RSCodec
from scipy.fftpack import dct, idct


BLOCK_SIZE = 8
CHANNEL = 2  # Canale Cr nello spazio colore YCbCr (più resistente)
COEF_POSITIONS = [(4,4), (5,5)]  # Posizioni coefficienti DCT di media frequenza
STRENGTH = 25  # Forza di embedding (regolabile)

RS_ECC_SYMBOLS = 40  # Corregge fino a 20 byte danneggiati


# ----------------------------------- Funzioni per incorporare ------------------------------------

def dct_embed_block(block, bit_sequence, index):
    """Incorpora dati in un singolo blocco DCT"""
    coeffs = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
    
    for i, (x,y) in enumerate(COEF_POSITIONS):
        if index + i >= len(bit_sequence):
            return coeffs, index + i
        
        bit = bit_sequence[index + i]
        coeffs[x,y] = STRENGTH * bit + np.sign(coeffs[x,y]) * (abs(coeffs[x,y]) // STRENGTH * STRENGTH)
    
    return idct(idct(coeffs, axis=1, norm='ortho'), axis=0, norm='ortho'), index + len(COEF_POSITIONS)


def steganography_embed(image_path, byte_data, output_path):
    # Codifica Reed-Solomon
    rsc = RSCodec(RS_ECC_SYMBOLS)
    protected_data = rsc.encode(byte_data)
    
    # Converti dati in bit
    bit_sequence = np.unpackbits(np.frombuffer(protected_data, dtype=np.uint8))
    
    # Carica immagine e converti in YCbCr
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Processa canale Cr
    channel = ycbcr[:,:,CHANNEL].astype(np.float32)
    rows, cols = channel.shape
    data_index = 0
    
    for i in range(0, rows - BLOCK_SIZE, BLOCK_SIZE):
        for j in range(0, cols - BLOCK_SIZE, BLOCK_SIZE):
            block = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            modified_block, data_index = dct_embed_block(block, bit_sequence, data_index)
            channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = modified_block
            
            if data_index >= len(bit_sequence):
                break
        if data_index >= len(bit_sequence):
            break
    
    # Ricostruisci l'immagine
    ycbcr[:,:,CHANNEL] = np.clip(channel, 0, 255).astype(np.uint8)
    embedded_image = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)
    
    # Salva con qualità ottimizzata per WhatsApp
    cv2.imwrite(output_path, embedded_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    

# ----------------------------------- Funzioni per l'estrazione ------------------------------------

def dct_extract_block(block):
    """Estrai dati da un singolo blocco DCT"""
    coeffs = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
    bits = []
    
    for (x,y) in COEF_POSITIONS:
        bits.append(round(coeffs[x,y] / STRENGTH) % 2)
    
    return bits


def steganography_extract(image_path, original_data_length):
    # Carica immagine e converti in YCbCr
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Estrai bit dal canale Cr
    channel = ycbcr[:,:,CHANNEL].astype(np.float32)
    rows, cols = channel.shape
    extracted_bits = []
    
    for i in range(0, rows - BLOCK_SIZE, BLOCK_SIZE):
        for j in range(0, cols - BLOCK_SIZE, BLOCK_SIZE):
            block = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            extracted_bits.extend(dct_extract_block(block))
    
    # Converti bit in byte
    protected_data = np.packbits(extracted_bits).tobytes()
    
    # Decodifica Reed-Solomon
    return RSCodec(RS_ECC_SYMBOLS).decode(protected_data)[0][:original_data_length]