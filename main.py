import sys
import os
import numpy as np

sys.path.insert(0, r"C:\Users\delga\Documents\VISION\Compression_Algorithms")

from src.logic.tools       import process_and_binarize
from src.logic.chain_codes import chain_f4, chain_f8, chain_vcc, chain_3ot
from src.transforms.Burrow_wheeler import bwt, ibwt

IMG_PATH = r"C:\Users\delga\Documents\VISION\ShapeMetrics-Interface\tests\data\apple8.gif"

img = process_and_binarize(IMG_PATH, threshold=128, padding=2)
print(f"Imagen cargada: {img.shape[0]}x{img.shape[1]} px\n")

cadenas = {
    "F4":  chain_f4(img),
    "F8":  chain_f8(img),
    "VCC": chain_vcc(img),
    "3OT": chain_3ot(img),
}

for nombre, cadena in cadenas.items():
    if not cadena:
        print(f"{nombre}: (vacía)\n")
        continue

    B, index = bwt(cadena)
    reconstruida = ibwt(B, index)
    ok = "lossless" if reconstruida == cadena else "ERROR "

    print(f"--- {nombre} ---")
    print(f"Longitud     : {len(cadena)}")
    print(f"Original     : {cadena}")
    print(f"BWT          : {B}  (index={index})")
    print(f"Round-trip   : {ok}\n")