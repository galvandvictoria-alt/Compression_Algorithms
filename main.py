"""
Módulo principal para el procesamiento de imágenes.

Extrae códigos de cadena, aplica transformaciones (BWT, MTFT) y compresión.
Crea una carpeta estructurada por cada imagen para almacenar de forma
organizada su archivo de log de terminal y sus gráficos correspondientes.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# sys.path.insert(0, r"C:\Users\delga\Documents\VISION\Compression_Algorithms")

from src.logic.tools import process_and_binarize
from src.logic.chain_codes import chain_f4, chain_f8, chain_vcc, chain_3ot
from src.transforms.Burrow_wheeler import bwt, ibwt
from src.transforms.Move_to_front import (
    move_to_front_transform,
    inverse_move_to_front_transform,
    ALFABETOS
)
from src.compression.compression_zip_huffman import (
    comprimir_huffman, descomprimir_huffman,
    comprimir_zip, descomprimir_zip,
    comprimir_aritmetico, descomprimir_aritmetico,
    comprimir_ppm, descomprimir_ppm
)
from src.logic.decoding_functions import (
    decode_f4_to_matrix, decode_f8_to_matrix,
    decode_af8_to_matrix, decode_vcc_to_matrix, decode_3ot_to_matrix
)


class LoggerDoble:
    """
    Clase para duplicar la salida estándar (sys.stdout).

    Imprime en la terminal y guarda simultáneamente en un archivo de texto.
    """

    def __init__(self, ruta_archivo):
        self.terminal = sys.stdout
        self.archivo_log = open(ruta_archivo, "w", encoding="utf-8")

    def write(self, mensaje):
        """Escribe el mensaje tanto en terminal como en el archivo."""
        self.terminal.write(mensaje)
        self.archivo_log.write(mensaje)

    def flush(self):
        """Fuerza la escritura del búfer."""
        self.terminal.flush()
        self.archivo_log.flush()

    def close(self):
        """Cierra el archivo de texto asociado."""
        self.archivo_log.close()


def main():
    """Función principal para procesar imágenes y empaquetar resultados."""
    directorio_img = "img"
    
    if not os.path.exists(directorio_img):
        print(f"El directorio '{directorio_img}' no existe.")
        return

    images = os.listdir(directorio_img)
    
    for image in images:
        # 1. Crear carpeta destino individual para la imagen actual
        nombre_base, _ = os.path.splitext(image)
        carpeta_destino = f"resultados/resultado_{nombre_base}"
        os.makedirs(carpeta_destino, exist_ok=True)
        
        # Definimos la ruta del archivo de texto dentro de su propia carpeta
        archivo_texto = os.path.join(carpeta_destino, f"output_{nombre_base}.txt")
        
        # Redirigimos sys.stdout a nuestro LoggerDoble
        logger = LoggerDoble(archivo_texto)
        sys.stdout = logger

        try:
            print(f"Nombre de la imagen: {image}")
            ruta_imagen = os.path.join(directorio_img, image)
            img = process_and_binarize(ruta_imagen, threshold=128, padding=2)
            print(f"Imagen cargada: {img.shape[0]}x{img.shape[1]} px\n")

            cadenas = {
                "F4":  chain_f4(img),
                "F8":  chain_f8(img),
                "VCC": chain_vcc(img),
                "3OT": chain_3ot(img),
            }

            resultados = {}

            for nombre, cadena in cadenas.items():
                if not cadena:
                    print(f"{nombre}: (vacía)\n")
                    continue

                print(f"{'='*55}")
                print(f"{nombre} - {len(cadena)} símbolos\n")
                print(f"{'-'*55}")

                # Burrow-Wheeler Transform (BWT)
                B, index = bwt(cadena)
                reconstruida = ibwt(B, index)
                ok = "lossless" if reconstruida == cadena else "ERROR "

                print(f"--- {nombre} ---")
                print(f"Longitud     : {len(cadena)}")
                print(f"Original     : {cadena}")
                print(f"BWT          : {B}  (index={index})")
                print(f"Round-trip   : {ok}\n")

                # Move-To-Front Transform (MTFT)
                base = ALFABETOS.get(nombre.upper(), None)
                simbolos_en_B = sorted(set(B))
                L0_usado = (
                    list(base) 
                    if (base and all(s in base for s in simbolos_en_B)) 
                    else simbolos_en_B
                )
                
                MZ, n, H_out = move_to_front_transform(B, chain_type=nombre)
                rec_mtft = inverse_move_to_front_transform(
                    MZ, n, L0_override=L0_usado
                )

                # Verificación MTFT
                ok_mtft = "lossless" if rec_mtft == B else "ERROR "
                print(f"MTFT         : ")
                print(f"Iteraciones usadas: {n}")
                print(f"Entropía de salida: {H_out:.4f} bits/símbolo")
                print(f"Round-trip   : {ok_mtft}\n")

                # Resultados de los compresores
                print(
                    f"[Compresores] (entrada: cadena post-MTFT, "
                    f"{len(MZ)} símbolos)"
                )
                print(
                    f"{'Algoritmo':<15} {'Tamaño (bytes)':<15} "
                    f"{'Tasa de compresión':<20}"
                )
                print(f"{'-'*14} {'-'*14} {'-'*18}")

                tam_original = len(cadena)
                resultados[nombre] = {}

                compresores = [
                    ("Huffman", comprimir_huffman, descomprimir_huffman),
                    ("ZIP", comprimir_zip, descomprimir_zip),
                    ("Aritmético", comprimir_aritmetico, descomprimir_aritmetico),
                    ("PPM", comprimir_ppm, descomprimir_ppm)
                ]

                for alg, fn_c, fn_d in compresores:
                    datos, info = fn_c(MZ)
                    rec = fn_d(datos, info)

                    # Descompresión total y verificación vs original
                    rec_inv_mtf = inverse_move_to_front_transform(
                        rec, n, L0_override=L0_usado
                    )
                    rec_bwt_total = ibwt(rec_inv_mtf, index)  
                    ok_comp = "lossless" if rec_bwt_total == cadena else "ERROR "

                    ratio = len(datos) / tam_original 
                    print(
                        f"{alg:<15} {len(datos):<15} "
                        f"{ratio:<20.4f} {ok_comp:<10}"
                    )
                    
                    # Almacenar en diccionario de resultados para las gráficas
                    resultados[nombre][alg] = {
                        "bytes": len(datos),
                        "ratio": ratio,
                    }

            print("\n")

            # ─────────────────────────────────────────────────────────────────
            # FIGURA 1: Imagen original + contornos F4, F8, VCC, 3OT
            # ─────────────────────────────────────────────────────────────────
            fig1, axes = plt.subplots(1, 5, figsize=(18, 4))
            fig1.suptitle(
                "Figura 1. Imagen original y contornos reconstruidos "
                "desde cada código de cadena\n",
                fontsize=11, y=1.02
            )
            
            # Panel 0: imagen original
            axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
            axes[0].set_title("Original\n(260×260 px)", fontsize=9)
            axes[0].axis('off')
            
            # Paneles 1-4: contornos reconstruidos
            reconstrucciones = [
                ("F4",  cadenas.get("F4",  []), decode_f4_to_matrix),
                ("F8",  cadenas.get("F8",  []), decode_f8_to_matrix),
                ("VCC", cadenas.get("VCC", []), decode_vcc_to_matrix),
                ("3OT", cadenas.get("3OT", []), lambda c: decode_3ot_to_matrix(c)[0]),
            ]
            
            for ax, (nombre, cadena, fn_decode) in zip(axes[1:], reconstrucciones):
                if cadena:
                    try:
                        mat = fn_decode(cadena)
                        ax.imshow(mat, cmap='gray')
                        ax.set_title(f"Reconstruido\ndesde {nombre}", fontsize=9)
                    except Exception as e:
                        ax.text(0.5, 0.5, f"Error:\n{e}", ha='center', va='center',
                                transform=ax.transAxes, fontsize=7)
                else:
                    ax.text(0.5, 0.5, "vacío", ha='center', va='center',
                            transform=ax.transAxes)
                ax.axis('off')
            
            fig1.tight_layout()
            nombre_plot_1 = os.path.join(
                carpeta_destino, f"plot_{nombre_base}_imagen_contornos.png"
            )
            fig1.savefig(nombre_plot_1, dpi=150, bbox_inches='tight')
            print(f"  Guardado: {nombre_plot_1}")
            
            # ─────────────────────────────────────────────────────────────────
            # FIGURA 2: Tamaño comprimido en bytes por algoritmo
            # ─────────────────────────────────────────────────────────────────
            nombres_cc  = list(resultados.keys())
            algoritmos  = ["Huffman", "Aritmético", "ZIP", "PPM"]
            colores     = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
            
            x     = np.arange(len(nombres_cc))
            width = 0.18
            
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            
            for j, (alg, color) in enumerate(zip(algoritmos, colores)):
                bytes_vals = [resultados[cc][alg]["bytes"] for cc in nombres_cc]
                offset     = (j - 1.5) * width
                bars       = ax2.bar(
                    x + offset, bytes_vals, width, label=alg, color=color
                )
                for bar, val in zip(bars, bytes_vals):
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 4,
                        str(val),
                        ha='center', va='bottom', fontsize=7.5
                    )
            
            # Línea de referencia = longitud original
            longitudes = [len(cadenas[cc]) for cc in nombres_cc]
            ax2.plot(x, longitudes, 'k--o', linewidth=1.2, markersize=5,
                    label='Sin comprimir (longitud original)')
            
            ax2.set_ylabel("Tamaño (bytes)", fontsize=11)
            ax2.set_xlabel("Código de cadena", fontsize=11)
            ax2.set_title(
                "Figura 2. Tamaño comprimido por algoritmo "
                "y tipo de código de cadena\n",
                fontsize=10
            )
            ax2.set_xticks(x)
            ax2.set_xticklabels(nombres_cc)
            ax2.legend(fontsize=9)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.grid(axis='y', linestyle='--', alpha=0.35)
            fig2.tight_layout()
            
            nombre_plot_2 = os.path.join(
                carpeta_destino, f"plot_{nombre_base}_bytes_comprimidos.png"
            )
            fig2.savefig(nombre_plot_2, dpi=150, bbox_inches='tight')
            print(f"  Guardado: {nombre_plot_2}")
            
            # ─────────────────────────────────────────────────────────────────
            # FIGURA 3: Tasa de compresión (ratio)
            # ─────────────────────────────────────────────────────────────────
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            
            for j, (alg, color) in enumerate(zip(algoritmos, colores)):
                ratio_vals = [resultados[cc][alg]["ratio"] for cc in nombres_cc]
                offset     = (j - 1.5) * width
                bars       = ax3.bar(
                    x + offset, ratio_vals, width, label=alg, color=color
                )
                for bar, val in zip(bars, ratio_vals):
                    ax3.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.3f}",
                        ha='center', va='bottom', fontsize=7
                    )
            
            ax3.axhline(1.0, color='black', linestyle='--', linewidth=1,
                        label='Sin compresión (ratio = 1.0)')
            ax3.set_ylabel("Tasa compresión (bytes out/longitud original)", fontsize=10)
            ax3.set_xlabel("Código de cadena", fontsize=11)
            ax3.set_title(
                "Figura 3. Tasa de compresión por algoritmo "
                "y tipo de código de cadena\n",
                fontsize=10
            )
            ax3.set_xticks(x)
            ax3.set_xticklabels(nombres_cc)
            ax3.legend(fontsize=9)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.grid(axis='y', linestyle='--', alpha=0.35)
            fig3.tight_layout()
            
            nombre_plot_3 = os.path.join(
                carpeta_destino, f"plot_{nombre_base}_ratio_compresion.png"
            )
            fig3.savefig(nombre_plot_3, dpi=150, bbox_inches='tight')
            print(f"   Guardado: {nombre_plot_3}")
            
            print(f"\nResultados exportados con éxito en la carpeta: {carpeta_destino}\n")

            # Liberar memoria de matplotlib
            plt.close(fig1)
            plt.close(fig2)
            plt.close(fig3)

        finally:
            # Restaurar la salida original para la siguiente iteración
            sys.stdout = logger.terminal
            logger.close()


if __name__ == "__main__":
    main()