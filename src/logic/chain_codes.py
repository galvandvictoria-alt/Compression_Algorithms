import numpy as np
from src.logic import tools

  
def chain_f4(binary_img):
    """
    Generate 4-directional Freeman chain code by tracing object contour.
    """
    def find_start_point(img):
        """
        Find starting point: first border pixel with background adjacent.
        """
        padded = np.pad(img, 1, 'constant', constant_values=0)

        # Scan image for first valid border pixel
        for y in range(1, padded.shape[0] - 1):
            for x in range(1, padded.shape[1] - 1):
                if padded[y, x] == 255:  # Foreground pixel
                    # Check if background exists above or to the left
                    if padded[y - 1, x] == 0 or padded[y, x - 1] == 0:
                        return x, y
        return None
    
    # Direction vectors: 0=right, 1=down, 2=left, 3=up
    directions = {
        0: (1, 0),   # Right
        1: (0, 1),   # Down
        2: (-1, 0),  # Left
        3: (0, -1)   # Up
    }

    padded = np.pad(binary_img, 1, 'constant', constant_values=0)

    start = find_start_point(binary_img)
    if start is None:
        return []

    start_x, start_y = start
    x, y = start_x, start_y
    current_dir = 0  # Start facing right

    chain = []

    # Trace contour until returning to start point
    for _ in range(10000):
        dx, dy = directions[current_dir]
        x += dx
        y += dy

        chain.append(current_dir)

        # Contour complete when returning to start
        if (x, y) == (start_x, start_y):
            break
        # Try turning left first (counterclockwise)
        current_dir = (current_dir + 3) % 4

        # Find next valid direction
        for _ in range(4):
            dx, dy = directions[current_dir]

            # Select pixel to check based on direction
            if current_dir == 0:      # Right: check current position
                px, py = x, y
            elif current_dir == 1:    # Down: check left neighbor
                px, py = x - 1, y
            elif current_dir == 2:    # Left: check diagonal
                px, py = x - 1, y - 1
            else:                     # Up: check top neighbor
                px, py = x, y - 1

            # If pixel is foreground, move in this direction
            if padded[py, px] == 255:
                break

            # Otherwise, turn right and try again
            current_dir = (current_dir + 1) % 4

    return chain

def chain_f8(binary_img):
    """
    Generate 8-directional Freeman chain code using contour extraction.
    """
    # Get ordered contour points from image using outline detection
    # not a 2D matrix. No reshape needed.
    contour = tools.find_outline(binary_img)
    contorno = contour["contour"]
    if contorno is None or len(contorno) == 0:
        return []
    contorno = np.array(contorno)  # shape (N, 2), each row is (x, y)
    # 8-directional mapping: (dy, dx) -> direction_code
    directions = {
        (0, 1): 0,    # Right
        (1, 1): 1,    # Down-right
        (1, 0): 2,    # Down
        (1, -1): 3,   # Down-left
        (0, -1): 4,   # Left
        (-1, -1): 5,  # Up-left
        (-1, 0): 6,   # Up
        (-1, 1): 7    # Up-right
    }

    chain = []
    
    # Process each contour point and encode direction to next point
    for i in range(len(contorno)):
        current_point = contorno[i]
        next_point = contorno[(i + 1) % len(contorno)]

        # Calculate displacement vector
        #dy = next_point[1] - current_point[1]
        #dx = next_point[0] - current_point[0]
        cx, cy = current_point
        nx, ny = next_point
        dy  = ny - cy
        dx = nx - cx
        

        # Map displacement to direction code
        if (dy, dx) in directions:
            chain.append(directions[(dy, dx)])

    # Rotate chain to align starting point
    if chain:
        chain = chain[-1:] + chain[:-1]

    return chain

def chain_af8(binary_img):
    """
    Convert F8 to AF8 using relative direction differences.
    """
    # Generate F8 chain first
    f8 = chain_f8(binary_img)

    # Lookup table: (previous_direction, current_direction) -> relative_turn
    lookup_table = {
        (0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3, (0, 4): 4, (0, 5): 5, (0, 6): 6, (0, 7): 7,
        (1, 0): 7, (1, 1): 0, (1, 2): 1, (1, 3): 2, (1, 4): 3, (1, 5): 4, (1, 6): 5, (1, 7): 6,
        (2, 0): 6, (2, 1): 7, (2, 2): 0, (2, 3): 1, (2, 4): 2, (2, 5): 3, (2, 6): 4, (2, 7): 5,
        (3, 0): 5, (3, 1): 6, (3, 2): 7, (3, 3): 0, (3, 4): 1, (3, 5): 2, (3, 6): 3, (3, 7): 4,
        (4, 0): 4, (4, 1): 5, (4, 2): 6, (4, 3): 7, (4, 4): 0, (4, 5): 1, (4, 6): 2, (4, 7): 3,
        (5, 0): 3, (5, 1): 4, (5, 2): 5, (5, 3): 6, (5, 4): 7, (5, 5): 0, (5, 6): 1, (5, 7): 2,
        (6, 0): 2, (6, 1): 3, (6, 2): 4, (6, 3): 5, (6, 4): 6, (6, 5): 7, (6, 6): 0, (6, 7): 1,
        (7, 0): 1, (7, 1): 2, (7, 2): 3, (7, 3): 4, (7, 4): 5, (7, 5): 6, (7, 6): 7, (7, 7): 0
    }

    # Compute relative direction at each step using lookup table
    return [lookup_table[(f8[i - 1], f8[i])] for i in range(len(f8))]

def chain_vcc(binary_img):
    """
    Encode vertical and horizontal direction changes in F4 chain.
    """
    # Generate F4 chain first
    f4 = chain_f4(binary_img)
    
    # Lookup table: (previous_dir, current_dir) -> change_type
    lookup_table = {
        (0, 0): 0, (0, 1): 1, (0, 3): 2,  # From right: 0=continue, 1=turn up/down, 2=turn left/opposite
        (1, 0): 2, (1, 1): 0, (1, 2): 1,  # From down: 0=continue, 1=turn left/right, 2=turn up/opposite
        (2, 1): 2, (2, 2): 0, (2, 3): 1,  # From left: 0=continue, 1=turn up/down, 2=turn right/opposite
        (3, 0): 1, (3, 2): 2, (3, 3): 0   # From up: 0=continue, 1=turn left/right, 2=turn down/opposite
    }

    # Classify each direction change using lookup table
    return [lookup_table.get((f4[i - 1], f4[i]), 0) for i in range(len(f4))]

def chain_3ot(binary_img):
    """
    Classify direction changes relative to a reference direction.
    """
    f4 = chain_f4(binary_img)
    if len(f4) < 2:
        return []

    chain = []
    reference = f4[0]
    previous = f4[0]
    direction_changed = False

    # Process each direction transition in the chain
    for i in range(1, len(f4)):
        current_dir = f4[i]

        if current_dir == previous:
            chain.append(0)  # No change: same direction
        else:
            if not direction_changed:
                chain.append(2)  # First direction change
                direction_changed = True
            elif current_dir == reference:
                chain.append(1)  # Return to reference direction
                reference = previous
            elif (current_dir - reference) % 4 == 2:
                chain.append(2)  # Opposite direction
                reference = previous
            else:
                chain.append(1)  # Other turn
                reference = previous

        previous = current_dir

    # Handle circular closure at chain end
    current_dir = f4[0]

    if current_dir == previous:
        chain.append(0)
    elif not direction_changed:
        chain.append(2)
    elif current_dir == reference:
        chain.append(1)
    elif (current_dir - reference) % 4 == 2:
        chain.append(2)
    else:
        chain.append(1)

    return chain


# =========================================
# USAGE EXAMPLES
# =========================================
# binary_img = ...  # Binary image matrix (0 or 255)

# f4_chain = chain_f4(binary_img)
# f8_chain = chain_f8(binary_img)
# af8_chain = chain_af8(binary_img)
# vcc_chain = chain_vcc(binary_img)
# chain_3ot_result = chain_3ot(binary_img, f4_chain)

# print("F4:", ''.join(map(str, f4_chain)))
# print("F8:", ''.join(map(str, f8_chain)))
# print("AF8:", ''.join(map(str, af8_chain)))
# print("VCC:", ''.join(map(str, vcc_chain)))
# print("3OT:", ''.join(map(str, chain_3ot_result)))

