"""
Move to front (MTFT) e Inverso (IMTFT)
Lógica de implementación:
  1. Se construye la lista L con el alfabeto inicial del chain code.
  2. Se aplica MTFT hasta 4 veces sobre la salida de la iteración anterior.
  3. Se calcula la entropía de Shannon en cada iteración.
  4. Se guarda la iteración n con menor entropía.
  5. El número n se almacena con 2 bits en el archivo comprimido.
"""

import math
from collections import Counter

ALFABETOS = {
    "F4":  [0, 1, 2, 3],
    "F8":  [0, 1, 2, 3, 4, 5, 6, 7],
    "3OT": [0, 1, 2],
    "VCC": [2, 1, 3],
    "NAD": [0, 1, 2, 3],
}


def entropia(cadena: list) -> float:
    """
    Mide qué tanto se puede comprimir la cadena.
    Menor entropía = más comprimible.
    """
    if not cadena:
        return 0.0
    n = len(cadena)
    conteo = Counter(cadena)
    return -sum((c / n) * math.log2(c / n) for c in conteo.values())


def move_to_front(cadena: list, L: list) -> list:
    """
    Lógica del paper. Para cada símbolo s de la cadena:
      1. Busca su posición i en la lista L.
      2. Emite el índice i.
      3. Mueve s al frente de L.

    Args:
        cadena : lista de símbolos a transformar.
        L      : lista ordenada del alfabeto.
    Returns:
        Lista de índices.
    """
    L = list(L)
    salida = []
    for s in cadena:
        i = L.index(s)
        salida.append(i)
        L.pop(i)
        L.insert(0, s)
    return salida


def inverse_move_to_front(indices: list, L: list) -> list:
    """
    Un paso inverso de Move-To-Front Transform.
    Para cada índice i en la lista:
      1. Recupera el símbolo s = L[i].
      2. Emite s.
      3. Mueve s al frente de L.

    Args:
        indices : lista de índices a decodificar.
        L       : misma lista inicial usada en la codificación.
    Returns:
        Lista de símbolos recuperados.
    """
    L = list(L)
    salida = []
    for i in indices:
        s = L[i]
        salida.append(s)
        L.pop(i)
        L.insert(0, s)
    return salida


def move_to_front_transform(cadena: list, chain_type: str = "F4", max_iter: int = 4) -> tuple[list, int, float]:
    """
    Aplica Move-To-Front Transform de 1 a max_iter veces y devuelve
    la iteración con menor entropía de Shannon.

    Cadena de pasos:
        iter 1: salida_1 = MTFT(cadena,   L0)   ← alfabeto del chain code
        iter 2: salida_2 = MTFT(salida_1, Ln)   ← alfabeto de índices
        iter 3: salida_3 = MTFT(salida_2, Ln)
        iter 4: salida_4 = MTFT(salida_3, Ln)

    Args:
        cadena     : lista de enteros del chain code.
        chain_type : tipo de cadena para elegir el alfabeto inicial.
        max_iter   : número máximo de iteraciones a probar (default 4).

    Returns:
        mejor_salida : cadena transformada con menor entropía.
        mejor_n      : número de iteraciones usadas (se guarda en 2 bits).
        mejor_H      : entropía de la mejor salida.
    """
    # Alfabeto inicial según el tipo de chain code
    L0     = list(ALFABETOS.get(chain_type.upper(), sorted(set(cadena))))
    L_size = len(L0)
    Ln     = list(range(L_size))  # alfabeto para pasos 2, 3, 4...

    mejor_salida = list(cadena)
    mejor_n      = 0
    mejor_H      = entropia(cadena)

    actual = list(cadena)
    for n in range(1, max_iter + 1):
        L      = L0 if n == 1 else Ln
        actual = move_to_front(actual, L)
        H      = entropia(actual)
        if H < mejor_H:
            mejor_H      = H
            mejor_salida = list(actual)
            mejor_n      = n

    return mejor_salida, mejor_n, mejor_H  # ← estaba fuera del for, corregido


def inverse_move_to_front_transform(indices: list, n: int, chain_type: str = "F4") -> list:
    """
    Deshace exactamente n pasos de MTFT en orden inverso.

    Espejo del encoder:
        paso n   usó Ln  → recupera salida_{n-1}
        ...
        paso 2   usó Ln  → recupera salida_1
        paso 1   usó L0  → recupera cadena original

    Args:
        indices    : cadena transformada (salida de move_to_front_transform).
        n          : número de iteraciones usadas al codificar.
        chain_type : mismo tipo que se usó al codificar.

    Returns:
        Cadena original reconstruida.
    """
    L0     = list(ALFABETOS.get(chain_type.upper(), sorted(set(indices))))
    L_size = len(L0)
    Ln     = list(range(L_size))

    resultado = list(indices)
    for k in range(n, 0, -1):
        L         = L0 if k == 1 else Ln
        resultado = inverse_move_to_front(resultado, L)
    return resultado