"""
Burrows-Wheeler Transform (BWT) e Inverso (IBWT)
Lógica de implementación: Tomamos la cadena de chain code y reordenamos los símbolos
para que los iguales queden juntos = más fácil de comprimir después.
"""

def bwt(cadena: list) -> tuple[list, int]:
    """
    Aplica la BWT a la cadena del chain code.
    1. Genera todas las rotaciones circulares de la cadena.
    2. Las ordena lexicográficamente.
    3. Toma el último símbolo de cada rotación ordenada.
    4. Guarda en qué fila quedó la cadena original.

    Args:
        cadena (list): Lista de símbolos del chain code.
    Returns:
        B     : cadena transformada (misma longitud).
        index : número entero que se necesita para deshacer el BWT.
    """
    if not cadena:
        return [], 0

    n = len(cadena)

    # Todas las rotaciones circulares
    rotaciones = [tuple(cadena[i:] + cadena[:i]) for i in range(n)]

    # Ordenamos lexicográficamente
    rotaciones_ordenadas = sorted(rotaciones)

    # Última columna de la tabla ordenada
    B = [fila[-1] for fila in rotaciones_ordenadas]

    # Fila donde quedó la cadena original
    index = rotaciones_ordenadas.index(tuple(cadena))

    return B, index


def ibwt(B: list, index: int) -> list:
    """
    Deshace la BWT y reconstruye la cadena original.

    Args:
        B     : cadena transformada por la BWT.
        index : número entero devuelto por bwt().
    Returns:
        La cadena original reconstruida.
    """
    if not B:
        return []

    n = len(B)

    # Primera columna = B ordenado
    F = sorted(B)

    # Para cada posición en B, cuántas veces apareció ese símbolo antes
    rango_B = []
    conteo = {}
    for sym in B:
        rango_B.append(conteo.get(sym, 0))
        conteo[sym] = conteo.get(sym, 0) + 1

    # Primera aparición de cada símbolo en F
    primer_aparicion = {}
    for pos, sym in enumerate(F):
        if sym not in primer_aparicion:
            primer_aparicion[sym] = pos

    # LF-mapping: conecta cada fila de B con su fila en F
    LF = [primer_aparicion[B[i]] + rango_B[i] for i in range(n)]

    # Reconstruye siguiendo el mapa n veces desde `index`
    S = []
    fila = index
    for _ in range(n):
        S.append(B[fila])
        fila = LF[fila]

    S.reverse()
    return S