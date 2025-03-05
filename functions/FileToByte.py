def file_to_bytes(file_path):
    """
    Converte il contenuto di un file in un array di byte.

    Parametri:
    - file_path (str): Il percorso del file da leggere.

    Restituisce:
    - bytearray: Un array di byte che rappresenta il contenuto del file.
    """
    try:
        with open(file_path, 'rb') as file:
            byte_data = file.read()
        return bytearray(byte_data)
    except Exception as e:
        print(f"Errore nella lettura del file: {e}")
        return bytearray()
