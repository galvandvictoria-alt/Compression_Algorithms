"""
    Algoritmos de compresión para chain codes

    Implementamos 4 algoritmos que reciben la cadena ya transformada
    y después la comprimen.
    
    Chain code -> BWT -> MTFT -> COMPRESSION -> BITS COMPRIMIDOS
    Algoritmos:
    1. Huffman - Códigos de longitud variable basados en frecuencia.
    2. Aritmético - Codificación por intervalos (Límite Shannon).
    3. ZIP - Wrapper sobre zlib.
    4. PPM - Predicción de contexto (Modelos de contexto).

"""
import zlib
import heapq
import math
import pickle
from fractions import Fraction
from collections import defaultdict, Counter


def construir_codigos_huffman(frecuencias: dict) -> dict:
    """
    Construye el árbol de Huffman y devuelve el símbolo
    Algoritmo:
    1. Crea un heap con frencuencias, id, nodo para cada símbolo.
    2. Extrae los dos de menor frecuencia y los une en un nodo.
    3. Repite hasta tener un solo nodo de raíz.
    4. Recorre el árbol. Izquierda = 0 , derecha = 1.
    """
    if not frecuencias:
        return{}
    if len(frecuencias) == 1:
        return{next(iter(frecuencias)): "0"}
    
    heap = []
    uid = 0
    for sym, freq in frecuencias.items():
        heapq.heappush(heap, (freq, uid, {"sym": sym, "izq": None, "der": None}))
        uid += 1

    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, uid, {"sym": None, "izq": n1, "der": n2}))
        uid += 1

    codigos = {}

    def recorrer(nodo, prefijo):
        if nodo["sym"] is not None:
            codigos[nodo["sym"]] = prefijo or "0"
            return
        recorrer(nodo["izq"], prefijo + "0")
        recorrer(nodo["der"], prefijo + "1")

    recorrer(heapq.heappop(heap)[2], "")
    return codigos

def comprimir_huffman(cadena: list) -> tuple[bytes, dict]:
    """
    Comprime la cadena con codificación de Huffman.
    
    Args:
        cadena: lista de enteros (chain code transformado).
    Returns:
        datos: bytes comprimidos.
        info: dict con códigos y relleno para descomprimir.    
    """

    if not cadena:
        return b"",{}
    
    frecuencias = Counter(cadena)
    codigos = construir_codigos_huffman(frecuencias)

    bits = "".join(codigos[s] for s in cadena)
    relleno = (8 - len(bits) % 8) % 8
    bits += "0" * relleno
    datos = bytes(int(bits[i:i+8],2) for i in range (0, len(bits), 8))

    info = {"codigos": codigos, "relleno": relleno, "longitud": len(cadena)}
    return datos, info

def descomprimir_huffman(datos: bytes, info: dict) -> list:
    """
    Reconstruye la cadena original desde los bits de Huffman.
    """

    if not datos:
        return []
    
    inverso = {v:k for k,v in info["codigos"].items()}
    relleno = info["relleno"]
    longitud = info["longitud"]

    bits = "".join(f"{byte:08b}" for byte in datos)
    if relleno:
        bits = bits[:-relleno]

    cadena = []
    actual = ""
    for bit in bits:
        actual += bit
        if actual in inverso:
            cadena.append(inverso[actual])
            actual = ""
            if len(cadena) == longitud:
                break
    return cadena

#------------------------------------------------
# Compresión Aritmética.
#-------------------------------------------------

def comprimir_aritmetico(cadena: list) -> tuple[bytes, dict]:
    """
    Comprime la cadena con codificación aritmética exacta.

    Usa fracciones de Python (Fraction) para evitar errores de precisión.
    Principio (Žalik et al. 2014 usa aritmético como codificador final):
        - Probabilidades acumuladas exactas de cada símbolo.
        - Intervalo [low, high) se estrecha con cada símbolo.
        - Se emiten bits mientras el intervalo esté completamente
          en [0, 0.5) o en [0.5, 1).

    Args:
        cadena : lista de enteros.
    Returns:
        datos : bytes comprimidos.
        info  : dict con tablas para descomprimir.
    """
    if not cadena:
        return b"", {}

    conteo  = Counter(cadena)
    total   = len(cadena)
    # Orden estable: mayor frecuencia primero
    simbolos = sorted(conteo.keys(), key=lambda s: (-conteo[s], s))

    # Probabilidades acumuladas exactas
    cum_low:  dict = {}
    cum_high: dict = {}
    acum = Fraction(0)
    for s in simbolos:
        cum_low[s]  = acum
        acum       += Fraction(conteo[s], total)
        cum_high[s] = acum

    # Codificación
    low      = Fraction(0)
    high     = Fraction(1)
    MITAD    = Fraction(1, 2)
    bits_out: list[int] = []

    for s in cadena:
        rango = high - low
        high  = low + rango * cum_high[s]
        low   = low + rango * cum_low[s]

        while True:
            if high <= MITAD:
                bits_out.append(0)
                low  = low  * 2
                high = high * 2
            elif low >= MITAD:
                bits_out.append(1)
                low  = (low  - MITAD) * 2
                high = (high - MITAD) * 2
            else:
                break

    # Bit final
    bits_out.append(0 if low < MITAD else 1)
    bits_out.append(1)  # bit extra para garantizar decodificación

    relleno  = (8 - len(bits_out) % 8) % 8
    n_bits   = len(bits_out)
    bits_out += [0] * relleno
    datos = bytes(
        int("".join(str(b) for b in bits_out[i:i+8]), 2)
        for i in range(0, len(bits_out), 8)
    )

    info = {
        "simbolos": simbolos,
        "cum_low":  {s: (v.numerator, v.denominator) for s, v in cum_low.items()},
        "cum_high": {s: (v.numerator, v.denominator) for s, v in cum_high.items()},
        "longitud": total,
        "relleno":  relleno,
        "n_bits":   n_bits,
    }
    return datos, info


def descomprimir_aritmetico(datos: bytes, info: dict) -> list:
    """
    Reconstruye la cadena desde los bytes del codificador aritmético.

    Args:
        datos : bytes de comprimir_aritmetico.
        info  : dict de comprimir_aritmetico.
    Returns:
        Cadena original reconstruida.
    """
    if not datos:
        return []

    longitud = info["longitud"]
    simbolos = info["simbolos"]
    n_bits   = info["n_bits"]
    cum_low  = {s: Fraction(n, d) for s, (n, d) in info["cum_low"].items()}
    cum_high = {s: Fraction(n, d) for s, (n, d) in info["cum_high"].items()}
    MITAD    = Fraction(1, 2)

    # Bits disponibles
    bits = []
    for byte in datos:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    bits = bits[:n_bits]

    # Valor inicial como fracción binaria
    valor = Fraction(0)
    for i, b in enumerate(bits):
        if b:
            valor += Fraction(1, 2 ** (i + 1))

    low  = Fraction(0)
    high = Fraction(1)
    pos  = len(bits)

    cadena: list = []
    for _ in range(longitud):
        rango    = high - low
        relativo = (valor - low) / rango

        sym = simbolos[-1]
        for s in simbolos:
            if cum_low[s] <= relativo < cum_high[s]:
                sym = s
                break
        cadena.append(sym)

        high = low + rango * cum_high[sym]
        low  = low + rango * cum_low[sym]

        while True:
            if high <= MITAD:
                low   = low  * 2
                high  = high * 2
                b     = bits[pos] if pos < len(bits) else 0
                valor = valor * 2 + Fraction(b, 2)
                pos  += 1
            elif low >= MITAD:
                low   = (low  - MITAD) * 2
                high  = (high - MITAD) * 2
                b     = bits[pos] if pos < len(bits) else 0
                valor = (valor - MITAD) * 2 + Fraction(b, 2)
                pos  += 1
            else:
                break

    return cadena
#------------------------------------------------
# ZIP WRAPPER SOBRE ZLIB Y HUFFMAN
#------------------------------------------------

def comprimir_zip(cadena: list) -> tuple[bytes, dict]:
    """
    Comprime la cadena usando zlib (equivalente a ZIP).

    zlib implementa el algoritmo DEFLATE (LZ77 + Huffman), que es la base
    del formato ZIP estándar.

    Args:
        cadena : lista de enteros.
    Returns:
        datos : bytes comprimidos.
        info  : dict con longitud original.
    """
    if not cadena:
        return b"", {"longitud": 0}

    raw   = pickle.dumps(cadena)
    datos = zlib.compress(raw, level=9)

    info = {"longitud": len(cadena), "raw_size": len(raw)}
    return datos, info


def descomprimir_zip(datos: bytes, info: dict) -> list:
    """
    Reconstruye la cadena desde los bytes comprimidos con zlib.

    Args:
        datos : bytes de comprimir_zip.
        info  : dict de comprimir_zip.
    Returns:
        Cadena original reconstruida.
    """
    if not datos:
        return []

    raw = zlib.decompress(datos)
    return pickle.loads(raw)

#------------------------------------------------------------
# PPM Prediccion parcial, mezcla de contexto.
#--------------------------------------------------------------

def comprimir_ppm(cadena: list, orden: int = 3) -> tuple[bytes, dict]:
    """
    Comprime la cadena con PPM (Prediction by Partial Matching).

    PPM es la "mezcla de contexto" del proyecto. Principio:
        - Para cada símbolo, busca el contexto más largo visto antes
          (hasta "orden" símbolos anteriores).
        - Estima la probabilidad del símbolo según ese contexto.
        - Si el contexto no existe, baja al orden inferior (escape).
        - Orden -1 = distribución uniforme (fallback).

    La implementación guarda las tablas de contexto y la cadena original
    comprimidas con zlib.

    Args:
        cadena : lista de enteros.
        orden  : longitud máxima del contexto (default 3).
    Returns:
        datos : bytes comprimidos.
        info  : dict con metadatos y bits teóricos.
    """
    if not cadena:
        return b"", {"longitud": 0}

    simbolos_unicos = sorted(set(cadena))
    n_sim           = max(len(simbolos_unicos), 1)

    # Construir tablas de frecuencia por contexto
    # contextos[k][ctx_tuple] = Counter de siguientes símbolos
    contextos: dict[int, dict] = {k: defaultdict(Counter) for k in range(orden + 1)}

    for i, sym in enumerate(cadena):
        contextos[0][()][sym] += 1
        for k in range(1, orden + 1):
            if i >= k:
                ctx = tuple(cadena[i - k: i])
                contextos[k][ctx][sym] += 1

    # Calcular bits teóricos usando las probabilidades PPM
    bits_teoricos = 0.0
    for i, sym in enumerate(cadena):
        prob = None
        for k in range(min(i, orden), -1, -1):
            if k == 0:
                tabla = contextos[0][()]
            else:
                ctx   = tuple(cadena[i - k: i])
                tabla = contextos[k].get(ctx)
            if tabla and sym in tabla:
                total = sum(tabla.values())
                prob  = tabla[sym] / total
                break
        if not prob:
            prob = 1.0 / n_sim
        bits_teoricos += -math.log2(prob)

    # Serializar cadena + tablas con zlib
    payload = pickle.dumps({
        "cadena":    cadena,
        "contextos": {k: dict(v) for k, v in contextos.items()},
        "orden":     orden,
    })
    datos = zlib.compress(payload, level=9)

    info = {
        "longitud":      len(cadena),
        "orden":         orden,
        "bits_teoricos": bits_teoricos,
        "simbolos":      simbolos_unicos,
    }
    return datos, info


def descomprimir_ppm(datos: bytes, info: dict) -> list:
    """
    Reconstruye la cadena desde los bytes comprimidos con PPM.

    Args:
        datos : bytes de comprimir_ppm.
        info  : dict de comprimir_ppm.
    Returns:
        Cadena original reconstruida.
    """
    if not datos:
        return []

    payload = zlib.decompress(datos)
    obj     = pickle.loads(payload)
    return obj["cadena"]

#-------------------------------------
#Comparación de los compresores:
#---------------------------------------------
def comparar_compresores(cadena: list, nombre: str = "") -> dict:
    """
    Aplica los 4 compresores y muestra tabla comparativa.

    Args:
        cadena : lista de enteros ya transformada (post BWT+MTFT).
        nombre : etiqueta de la cadena (ej. "F4", "F8").
    Returns:
        dict con resultados por compresor.
    """
    etiqueta = f" {nombre} " if nombre else " "
    print(f"\n{'─'*58}")
    print(f"  Compresores{etiqueta}— {len(cadena)} símbolos")
    print(f"{'─'*58}")

    tam_orig = len(cadena)
    resultados = {}

    for nom_c, fn_c, fn_d in [
        ("Huffman",    comprimir_huffman,    descomprimir_huffman),
        ("Aritmético", comprimir_aritmetico, descomprimir_aritmetico),
        ("ZIP",        comprimir_zip,        descomprimir_zip),
        ("PPM",        comprimir_ppm,        descomprimir_ppm),
    ]:
        datos, info = fn_c(cadena)
        rec         = fn_d(datos, info)
        ok          = "OK" if rec == cadena else "NOT OK"
        resultados[nom_c] = {"bytes": len(datos), "ok": ok, "ratio": len(datos) / tam_orig}

    print(f"  {'Compresor':<14} {'Bytes out':>10}  {'Ratio':>7}  {'Lossless':>8}")
    print(f"  {'─'*14} {'─'*10}  {'─'*7}  {'─'*8}")
    print(f"  {'Original':<14} {tam_orig:>10}  {'1.000':>7}")
    for nom_c, r in resultados.items():
        print(f"  {nom_c:<14} {r['bytes']:>10}  {r['ratio']:>7.3f}  {r['ok']:>8}")

    return resultados
