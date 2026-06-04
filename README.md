# Procesamiento, Análisis y Compresión de Contornos mediante Códigos de Cadena

## Resumen
Este proyecto implementa un *pipeline* experimental diseñado para la extracción, transformación y compresión sin pérdidas (*lossless*) de contornos bidimensionales en imágenes digitales. El sistema evalúa el rendimiento de múltiples representaciones de fronteras (Códigos de Cadena) en conjunción con transformadas de reordenamiento de datos y algoritmos de compresión entrópica y basada en diccionarios.

## Arquitectura y Metodología del Sistema

El flujo de ejecución se divide en cuatro fases principales:

1. **Preprocesamiento y Binarización:**
   Las imágenes de entrada son procesadas para separar el objeto de interés del fondo mediante un umbral predefinido, aplicando técnicas de *padding* para evitar el truncamiento de fronteras en los bordes de la imagen.

2. **Representación Topológica (Códigos de Cadena):**
   Los contornos extraídos se codifican utilizando cuatro esquemas de representación direccional:
   * **F4 / F8 (Freeman Chain Codes):** Codificación de 4 y 8 conectividades espaciales.
   * **VCC (Vertex Chain Code):** Representación basada en los vértices de las celdas de contorno.
   * **3OT (Three Orthogonal Transitions):** Representación que minimiza el alfabeto utilizando transiciones ortogonales.

3. **Transformadas de Reordenamiento Secuencial:**
   Para optimizar la redundancia de los datos antes de la compresión, se aplican dos transformadas consecutivas:
   * **BWT (Burrows-Wheeler Transform):** Reagrupa secuencias de símbolos repetitivos agrupando caracteres lexicográficamente similares sin alterar la información original.
   * **MTFT (Move-To-Front Transform):** Transforma la salida de la BWT en una distribución sesgada hacia índices menores, reduciendo significativamente la entropía local del mensaje.

4. **Compresión y Evaluación (Benchmarking):**
   Las secuencias transformadas son introducidas a cuatro algoritmos de compresión (*Huffman, Codificación Aritmética, ZIP/Deflate, y PPM*). El sistema evalúa la tasa de compresión y verifica la integridad de los datos (Round-trip validación) asegurando una reconstrucción exacta a la matriz bidimensional original.

## Estructura de Salida de Datos

El algoritmo procesa todas las imágenes alojadas en el directorio `./img/` y genera una carpeta de resultados independiente por cada imagen (`resultado_<nombre_imagen>`). Cada carpeta contiene:

* `output_<nombre>.txt`: Bitácora detallada de la ejecución (longitudes de cadena, iteraciones MTFT, entropía de salida y verificación *lossless*).
* `plot_*_imagen_contornos.png`: Gráfico comparativo que demuestra la reconstrucción de la matriz original a partir de las cadenas comprimidas.
* `plot_*_bytes_comprimidos.png`: Diagrama de barras con los tamaños finales en bytes por cada algoritmo.
* `plot_*_ratio_compresion.png`: Gráfico que expone las tasas de compresión normalizadas en relación con la longitud de la cadena original.

## Dependencias

* `Python 3.8+`
* `numpy`
* `matplotlib`
* Módulos locales: Requiere la estructura de paquetes `src.logic`, `src.transforms` y `src.compression` adecuadamente configurada en el entorno de ejecución (o en el `PYTHONPATH`).

## Ejecución

Para iniciar el flujo de procesamiento, sitúese en el directorio raíz del proyecto y ejecute:

```bash
python main.py
```