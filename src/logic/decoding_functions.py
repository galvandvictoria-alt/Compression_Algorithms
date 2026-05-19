
import numpy as np

# =============================
# PURE MATHEMATICAL LOGIC
# =============================

def fill_shape(binary_matrix):
    """
    Fill the interior of a closed contour using Flood Fill algorithm.
    
    Args:
        binary_matrix (np.ndarray): Binary image with contour (0 or 255)
        
    Returns:
        np.ndarray: Filled image (interior marked with 255)
    """
    rows, cols = binary_matrix.shape
    exterior = np.zeros((rows, cols), dtype=np.uint8)
    
    stack = [(0, 0)]
    exterior[0, 0] = 255
    
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while stack:
        r, c = stack.pop()
        
        for dr, dc in movements:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols:
                if binary_matrix[nr, nc] == 0 and exterior[nr, nc] == 0:
                    exterior[nr, nc] = 255
                    stack.append((nr, nc))
    
    filled_img = np.zeros_like(binary_matrix)
    filled_img[exterior == 0] = 255
    
    return filled_img


def af8_to_f8(af8_chain):
    """
    Convert AF8 (relative) chain to F8 (absolute) by testing initial directions.

    The AF8 encoder produces: af8[i] = (f8[i] - f8[i-1]) mod 8  (circular, i=0 uses f8[-1])
    So to decode:  f8[i] = (f8[i-1] + af8[i]) mod 8
    We test all 8 possible values for f8[-1] (= f8[last]) as the seed.
    """
    moves = {0:(0,1), 1:(1,1), 2:(1,0), 3:(1,-1),
            4:(0,-1), 5:(-1,-1), 6:(-1,0), 7:(-1,1)}

    # Probamos los 8 valores posibles de f8[-1] como semilla.
    for last_direction in range(8):
        f8 = []
        previous = last_direction

        for relative_symbol in af8_chain:
            current = (previous + relative_symbol) % 8
            f8.append(current)
            previous = current

        # La semilla es correcta si f8[-1] coincide con last_direction Y la trayectoria cierra
        if f8 and f8[-1] == last_direction:
            x, y = 0, 0
            for direction in f8:
                dx, dy = moves[direction]
                x += dx
                y += dy
            if x == 0 and y == 0:
                return f8

    # Fallback: devolver la mejor aproximacion posible
    f8 = []
    previous = 0
    for relative_symbol in af8_chain:
        current = (previous + relative_symbol) % 8
        f8.append(current)
        previous = current
    return f8


def closes_f4_shape(f4_chain, tolerance=0):
    """
    Verify if an F4 chain generates a closed contour.
    tolerance=0: cierre exacto
    tolerance>0: acepta cadenas que casi cierran (distancia Manhattan <= tolerance)
    """
    x, y = 0, 0
    moves_standard_freeman = {
        0:(1, 0),
        1:(0, 1),
        2:(-1, 0),
        3:(0, -1)
    }
    for direction in f4_chain:
        dx, dy = moves_standard_freeman[direction]
        x += dx
        y += dy

    return (abs(x) + abs(y)) <= tolerance


def vcc_to_f4(vcc_chain, initial_direction=0):
    """
    Convert VCC (variable-length code) chain to F4 (absolute directions).
    
    Args:
        vcc_chain (list): VCC chain with symbols 1, 2, 3
        initial_direction (int): Starting direction (0-3)
        
    Returns:
        list: Absolute F4 directions
    """
    vcc_table = {
        (0, 1): 1, (0, 2): 3, (0, 3): 0,
        (1, 1): 2, (1, 2): 0, (1, 3): 1,
        (2, 1): 3, (2, 2): 1, (2, 3): 2,
        (3, 1): 0, (3, 2): 2, (3, 3): 3
    }

    f4 = []
    previous_direction = initial_direction

    for symbol in vcc_chain:
        if (previous_direction, symbol) in vcc_table:
            new_f4 = vcc_table[(previous_direction, symbol)]
            f4.append(new_f4)
            previous_direction = new_f4
        else:
            f4.append(previous_direction)

    return f4


def c3ot_to_f4(c3ot_chain):
    """
    Convert 3OT chain to F4 by inverting exactly the encoder logic of chain_3ot.

    El simbolo '1' es ambiguo en el encoder: puede significar
    "regresa a referencia" O "cualquier otro giro" (que es +1 o +3 desde previous).
    Por eso probamos las 3 interpretaciones posibles para cada simbolo 1:
      a) current_dir = reference
      b) current_dir = (previous + 1) % 4
      c) current_dir = (previous + 3) % 4
    usando BFS para encontrar la combinacion que cierra la forma.
    """
    if not c3ot_chain:
        return [], False

    def simulate(first_step, first_turn_delta, symbol1_choices):
        """
        Simula el decoder con elecciones especificas para cada simbolo ambiguo.
        symbol1_choices: lista de decisiones (0=ref, 1=+1, 2=+3) para cada simbolo '1'
        """
        f4 = [first_step]
        reference = first_step
        previous = first_step
        direction_changed = False
        choice_idx = 0

        for symbol in c3ot_chain:
            if symbol == 0:
                current_dir = previous

            elif symbol == 2 and not direction_changed:
                current_dir = (previous + first_turn_delta) % 4
                direction_changed = True
                reference = previous

            elif symbol == 1:
                choice = symbol1_choices[choice_idx] if choice_idx < len(symbol1_choices) else 0
                choice_idx += 1
                if choice == 0:
                    current_dir = reference
                elif choice == 1:
                    current_dir = (previous + 1) % 4
                else:
                    current_dir = (previous + 3) % 4
                reference = previous

            elif symbol == 2:
                current_dir = (reference + 2) % 4
                reference = previous

            else:
                current_dir = previous

            f4.append(current_dir)
            previous = current_dir

        return f4

    # Contar cuantos simbolos '1' hay para saber cuantas decisiones tomar
    num_ones = c3ot_chain.count(1)

    # Estrategia 1: probar con todos los simbolos 1 = "regresa a referencia" (choice=0)
    # Esta es la interpretacion mas comun y rapida de probar
    for first_step in range(4):
        for first_turn_delta in [1, 3]:
            choices = [0] * num_ones
            f4 = simulate(first_step, first_turn_delta, choices)
            if closes_f4_shape(f4):
                return f4, True

    # Estrategia 2: todos los simbolos 1 = giro izquierda
    for first_step in range(4):
        for first_turn_delta in [1, 3]:
            choices = [1] * num_ones
            f4 = simulate(first_step, first_turn_delta, choices)
            if closes_f4_shape(f4):
                return f4, True

    # Estrategia 3: todos los simbolos 1 = giro derecha
    for first_step in range(4):
        for first_turn_delta in [1, 3]:
            choices = [2] * num_ones
            f4 = simulate(first_step, first_turn_delta, choices)
            if closes_f4_shape(f4):
                return f4, True

    # Estrategia 4: alternar entre referencia y giro izquierda
    for first_step in range(4):
        for first_turn_delta in [1, 3]:
            choices = [i % 2 for i in range(num_ones)]
            f4 = simulate(first_step, first_turn_delta, choices)
            if closes_f4_shape(f4):
                return f4, True

    # Estrategia 5: alternar entre referencia y giro derecha
    for first_step in range(4):
        for first_turn_delta in [1, 3]:
            choices = [0 if i % 2 == 0 else 2 for i in range(num_ones)]
            f4 = simulate(first_step, first_turn_delta, choices)
            if closes_f4_shape(f4):
                return f4, True

    # Estrategia 6: tolerar cierre aproximado (distancia <= 2)
    for first_step in range(4):
        for first_turn_delta in [1, 3]:
            for choices in [[0]*num_ones, [1]*num_ones, [2]*num_ones,
                            [i % 2 for i in range(num_ones)],
                            [0 if i % 2 == 0 else 2 for i in range(num_ones)]]:
                f4 = simulate(first_step, first_turn_delta, choices)
                if closes_f4_shape(f4, tolerance=2):
                    return f4, True  # Aceptar como valido con tolerancia

    # Fallback: devolver el mejor intento (el que mas cerca cierra)
    best_f4 = None
    best_dist = float('inf')
    moves = {0:(1,0), 1:(0,1), 2:(-1,0), 3:(0,-1)}
    for first_step in range(4):
        for first_turn_delta in [1, 3]:
            for choices in [[0]*num_ones, [1]*num_ones, [2]*num_ones]:
                f4 = simulate(first_step, first_turn_delta, choices)
                x, y = 0, 0
                for d in f4:
                    dx, dy = moves[d]
                    x += dx
                    y += dy
                dist = abs(x) + abs(y)
                if dist < best_dist:
                    best_dist = dist
                    best_f4 = f4

    return best_f4, False


# =============================
# INTELLIGENT DRAWING ENGINES (BOUNDING BOX)
# =============================

def f4_to_matrix(f4_chain, padding=10):
    """
    Generate matrix from F4 chain with automatic bounding box calculation.
    
    The function:
    1. Simulates the path based on F4 directions starting at (0,0)
    2. Finds min/max coordinates
    3. Creates a matrix sized to fit the shape with padding
    4. Draws the contour (value 255)
    
    Args:
        f4_chain (list): F4 directions (0-3)
        padding (int): Padding in pixels around the shape
        
    Returns:
        np.ndarray: Binary matrix with contour drawn
    """
    moves_standard_freeman = {0:(1, 0), 1:(0, 1), 2:(-1, 0), 3:(0, -1)} 
    
    x, y = 0, 0
    coordinates = [(x, y)]
    
    for direction in f4_chain:
        dx, dy = moves_standard_freeman[direction]
        x += dx
        y += dy
        coordinates.append((x, y))
    
    # Bbox calculation:
    all_x = [coord[0] for coord in coordinates]
    all_y = [coord[1] for coord in coordinates]
    
    min_x = min(all_x); max_x = max(all_x)
    min_y = min(all_y); max_y = max(all_y)
    
    width = max_x - min_x + 1  # Corresponde a X
    height = max_y - min_y + 1 # Corresponde a Y
    
    final_height = height + 2 * padding
    final_width = width + 2 * padding
    
    matrix = np.zeros((final_height, final_width), dtype=np.uint8)
    
    # Draw contour:
    for coord_x, coord_y in coordinates:
        adj_x = coord_x - min_x + padding # Coordenada Horizontal -> Columna
        adj_y = coord_y - min_y + padding # Coordenada Vertical -> Fila
        
        # CAMBIO CRÍTICO: Indexación Numpy standard [fila, columna] es [y, x]
        # matrix[fila=Vertical(y), col=Horizontal(x)]
        matrix[adj_y, adj_x] = 255
    
    return matrix


def f8_to_matrix(f8_chain, padding=10):
    """
    Generate matrix from F8 chain with automatic bounding box calculation.
    
    The function:
    1. Simulates the path based on F8 directions starting at (0,0)
    2. Finds min/max coordinates
    3. Creates a matrix sized to fit the shape with padding
    4. Draws the contour (value 255)
    
    Args:
        f8_chain (list): F8 directions (0-7)
        padding (int): Padding in pixels around the shape
        
    Returns:
        np.ndarray: Binary matrix with contour drawn
    """
    moves = {0:(0,1), 1:(1,1), 2:(1,0), 3:(1,-1),
            4:(0,-1), 5:(-1,-1), 6:(-1,0), 7:(-1,1)}
    
    # Simulate path to find bounding box
    x, y = 0, 0
    coordinates = [(x, y)]
    
    for direction in f8_chain:
        dx, dy = moves[direction]
        x += dx
        y += dy
        coordinates.append((x, y))
    
    # Find bounding box
    all_x = [coord[0] for coord in coordinates]
    all_y = [coord[1] for coord in coordinates]
    
    min_x = min(all_x)
    max_x = max(all_x)
    min_y = min(all_y)
    max_y = max(all_y)
    
    # Calculate absolute dimensions
    # x -> columnas (ancho), y -> filas (alto)
    width  = max_x - min_x + 1   #  antes estaba invertido (usaba max_x para height)
    height = max_y - min_y + 1   #  antes estaba invertido (usaba max_y para width)
    
    # Create matrix with padding
    final_height = height + 2 * padding
    final_width  = width  + 2 * padding
    
    matrix = np.zeros((final_height, final_width), dtype=np.uint8)
    
    # Draw contour with adjusted coordinates
    # numpy indexa [fila, columna] = [y, x], no [x, y]
    for coord_x, coord_y in coordinates:
        adj_x = coord_x - min_x + padding   # columna
        adj_y = coord_y - min_y + padding   # fila
        matrix[adj_y, adj_x] = 255
    
    return matrix


# =============================
# DECODER FUNCTIONS (PURE LOGIC)
# =============================

def decode_f4_to_matrix(f4_chain):
    """
    Decode F4 chain to filled matrix.
    
    Args:
        f4_chain (list): F4 directions (0-3)
        
    Returns:
        np.ndarray: Filled binary matrix
    """
    contour = f4_to_matrix(f4_chain)
    filled = fill_shape(contour)
    return filled


def decode_f8_to_matrix(f8_chain):
    """
    Decode F8 chain to filled matrix.
    
    Args:
        f8_chain (list): F8 directions (0-7)
        
    Returns:
        np.ndarray: Filled binary matrix
    """
    contour = f8_to_matrix(f8_chain)
    filled = fill_shape(contour)
    return filled


def decode_af8_to_matrix(af8_chain):
    """
    Decode AF8 chain to filled matrix.
    
    Args:
        af8_chain (list): Relative AF8 directions (0-7)
        
    Returns:
        np.ndarray: Filled binary matrix
    """
    f8_chain = af8_to_f8(af8_chain)
    contour = f8_to_matrix(f8_chain)
    filled = fill_shape(contour)
    return filled


def decode_vcc_to_matrix(vcc_chain):
    """
    Decode VCC chain to filled matrix.
    
    Args:
        vcc_chain (list): VCC chain with symbols 1, 2, 3
        
    Returns:
        np.ndarray: Filled binary matrix
    """
    f4_chain = vcc_to_f4(vcc_chain)
    contour = f4_to_matrix(f4_chain)
    filled = fill_shape(contour)
    return filled


def decode_3ot_to_matrix(c3ot_chain):
    """
    Decode 3OT chain to filled matrix.
    
    Args:
        c3ot_chain (list): 3OT chain with symbols 0=straight, 1=left, 2=right
        
    Returns:
        tuple: (filled_matrix, is_closed) where is_closed indicates if shape properly closes
    """
    f4_chain, is_closed = c3ot_to_f4(c3ot_chain)
    
    if not is_closed:
        contour = f4_to_matrix(f4_chain)
        return contour, False
    
    contour = f4_to_matrix(f4_chain)
    filled = fill_shape(contour)
    return filled, True


# =============================
# UTILITY FUNCTIONS (PURE LOGIC)
# =============================

def get_contour_f4(f4_chain):
    """
    Get only the contour (not filled) from F4 chain.
    
    Args:
        f4_chain (list): F4 directions (0-3)
        
    Returns:
        np.ndarray: Matrix with contour drawn (value 255)
    """
    return f4_to_matrix(f4_chain)


def get_contour_f8(f8_chain):
    """
    Get only the contour (not filled) from F8 chain.
    
    Args:
        f8_chain (list): F8 directions (0-7)
        
    Returns:
        np.ndarray: Matrix with contour drawn (value 255)
    """
    return f8_to_matrix(f8_chain)

