import sys
import os
import numpy as np

sys.path.insert(0, r"C:\Users\delga\Documents\VISION\Compression_Algorithms")

from src.logic.tools       import process_and_binarize
from src.logic.chain_codes import chain_f4, chain_f8, chain_vcc, chain_3ot
from src.transforms.Burrow_wheeler import bwt, ibwt
from src.transforms.Move_to_front import move_to_front_transform, inverse_move_to_front_transform,  ALFABETOS

from src.compression.compression_zip_huffman import (comprimir_huffman, descomprimir_huffman,
                                                     comprimir_zip, descomprimir_zip,
                                                     comprimir_aritmetico, descomprimir_aritmetico,
                                                     comprimir_ppm, descomprimir_ppm)

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

    print(f"{'='*55}")
    print(f"{nombre} - {len(cadena)} símbolos\n")
    print(f"{'-'*55}")

    #BWT
    B, index = bwt(cadena)
    reconstruida = ibwt(B, index)
    ok = "lossless" if reconstruida == cadena else "ERROR "

    print(f"--- {nombre} ---")
    print(f"Longitud     : {len(cadena)}")
    print(f"Original     : {cadena}")
    print(f"BWT          : {B}  (index={index})")
    print(f"Round-trip   : {ok}\n")

    # MTFT
    base  = ALFABETOS.get(nombre.upper(), None)
    simbolos_en_B = sorted(set(B))
    L0_usado = list(base) if (base and all(s in base for s in simbolos_en_B))  else simbolos_en_B
    
    #ctype = nombre
    MZ, n, H_out = move_to_front_transform(B, chain_type=nombre)
    rec_mtft = inverse_move_to_front_transform(MZ, n, L0_override=L0_usado)


    #verificacion
    ok_mtft = "lossless" if rec_mtft == B else "ERROR "
    print(f"MTFT         : ")
    print(f"Iteraciones usadas: {n}")
    print(f"Entropía de salida: {H_out:.4f} bits/símbolo")
    print(f"Round-trip   : {ok_mtft}\n")

    #Resultados de los compresores:
    print(f"[Compresores] (entrada: cadena post-MTFT, {len(MZ)} símbolos)")
    print(f"{'Algoritmo':<15} {'Tamaño (bytes)':<15} {'Tasa de compresión':<20}")
    print(f"{'-'*14}, {'-'*8}, {'-'*7}, {'-'*9}")

    tam_original = len(cadena)

    for alg, fn_c, fn_d in [
        ("Huffman", comprimir_huffman, descomprimir_huffman),
        ("ZIP",     comprimir_zip,     descomprimir_zip),
        ("Aritmético", comprimir_aritmetico, descomprimir_aritmetico),
        ("PPM", comprimir_ppm, descomprimir_ppm)    
    ]:
        datos, info = fn_c(MZ)
        rec = fn_d(datos, info)

        #Descompresión total y verificación vs el chain code original
        rec_bwt_total = ibwt(inverse_move_to_front_transform(rec, n, L0_override=L0_usado), index)  
        ok = "lossless" if rec_bwt_total == cadena else "ERROR "

        ratio = len(datos) / tam_original 
        print(f"{alg:<15} {len(datos):<15} {ratio:<20.4f} {ok:<10}")

    print()