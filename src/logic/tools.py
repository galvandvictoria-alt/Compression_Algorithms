import numpy as np
import os
import math
from collections import Counter
from PIL import Image
import heapq
import matplotlib.pyplot as plt

def reorder_contour(contour):
    """
    Reorder contour points to start from top-left and traverse clockwise.
    """
    if contour is None or contour.size == 0:
        return None

    points = contour.reshape(-1, 2)

    # Find starting point: topmost, then leftmost
    start_idx = min(range(len(points)), key=lambda i: (points[i][1], points[i][0]))

    # Rotate list to start from that point
    points = np.concatenate((points[start_idx:], points[:start_idx]), axis=0)

    # Reverse order to make it clockwise
    points = points[::-1]

    return points.reshape((-1, 1, 2))

def save_matrix_to_csv(matrix_data: np.ndarray, filename: str):
    """
    Save matrix data to CSV file.
    """
    # Remove existing extension if present
    filename = filename.split(".")[0]
    
    # Ensure CSV extension
    if not filename.endswith('.csv'):
        filename = f"{filename}.csv"

    # Write matrix to file with comma delimiter
    np.savetxt(filename, matrix_data, delimiter=",", fmt='%g')
    
    print(f"File saved successfully: {filename}")

def process_and_binarize(filename, threshold=128, padding=2):
    """
    Load image, convert to grayscale, and apply binary threshold.
    Adds black padding around the image so objects touching the border
    are always surrounded by background, ensuring correct contour detection.
    """
    try:
        with Image.open(filename) as img:
            img.seek(0)
            gray_img = img.convert('L')
            np_array = np.array(gray_img)

            # Convert to binary: pixels > threshold become 255, rest 0
            binary_array = ((np_array > threshold) * 255).astype(int)

            # Add black padding so objects never touch the image border
            if padding > 0:
                binary_array = np.pad(
                    binary_array, padding,
                    mode='constant', constant_values=0
                )

            return binary_array
    except Exception as e:
        return e

def connected_components(matrix: np.ndarray, neighbor: int = 4) -> int:
    """
    Count number of connected components in binary matrix.
    """
    rows, cols = matrix.shape
    visited = np.zeros((rows, cols), dtype=bool)

    # Define movement directions based on connectivity type
    if neighbor == 4:
        movements: list[tuple] = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    elif neighbor == 8:
        movements: list[tuple] = [(0, 1), (0, -1), (-1, 0), (1, 0),
                                   (-1, 1), (-1, -1), (1, 1), (1, -1)]
    else:
        return TypeError("Invalid neighborhood type")
    
    num_objects: int = 0
    
    # Scan image for unvisited foreground pixels
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Found new unvisited foreground pixel
            if matrix[i][j] == 1 and not visited[i][j]:
                num_objects += 1

                # Use stack-based flood fill to mark all connected pixels
                stack = [(i, j)]
                visited[i][j] = True

                while stack:
                    current_row, current_col = stack.pop()

                    # Check all neighbors
                    for dr, dc in movements:
                        next_row, next_col = current_row + dr, current_col + dc
                        # Verify bounds and foreground/unvisited condition
                        if 0 <= next_row < rows and 0 <= next_col < cols:
                            if matrix[next_row][next_col] == 1 and not visited[next_row][next_col]:
                                visited[next_row][next_col] = True
                                stack.append((next_row, next_col))

    return num_objects

def find_outline(matrix: np.ndarray) -> dict:
    """
    Detect object outline/edges using neighborhood analysis.
    Returns:
      - "contour": ordered list of (x,y) border points for chain_f8
      - "outline_matrix": 2D matrix marking edge pixels (for display)
      - "perimeter": count of edge pixels
    """
    rows, cols = matrix.shape
    outline_count = 0
    outline = np.zeros((rows, cols), dtype=int)

    # Normalize: accept 0/1 or 0/255 images
    norm = matrix if matrix.max() <= 1 else (matrix > 127).astype(int)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if norm[i][j] == 0:
                continue
            neighborhood_sum = (norm[i, j-1] + norm[i, j+1] +
                                norm[i-1, j] + norm[i+1, j])
            if neighborhood_sum < 4:
                outline[i][j] = 1
                outline_count += 1

    # Build ordered contour point list using Moore neighborhood tracing
    ordered_contour = []
    start = None
    for i in range(rows):
        for j in range(cols):
            if outline[i][j] == 1:
                start = (i, j)
                break
        if start:
            break

    if start:
        neighbors_8 = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
        visited_trace = set()
        cr, cc = start
        prev_dir = 6
        ordered_contour.append((cc, cr))  # (x=col, y=row)
        visited_trace.add((cr, cc))

        for _ in range(outline_count * 2):
            found = False
            start_search = (prev_dir + 5) % 8
            for k in range(8):
                d = (start_search + k) % 8
                nr = cr + neighbors_8[d][0]
                nc = cc + neighbors_8[d][1]
                if (0 <= nr < rows and 0 <= nc < cols
                        and outline[nr][nc] == 1
                        and (nr, nc) not in visited_trace):
                    ordered_contour.append((nc, nr))  # (x, y)
                    visited_trace.add((nr, nc))
                    prev_dir = d
                    cr, cc = nr, nc
                    found = True
                    break
            if not found:
                break

    # Restore outline matrix to original value range
    outline_matrix = (outline * 255).astype(np.uint8) if matrix.max() > 1 else outline

    return {"contour": ordered_contour, "perimeter": outline_count, "outline_matrix": outline_matrix}

def plot_histograms(frequency_dict, probability_dict):
    """
    Generates two stacked, plots.
    """
    if not frequency_dict or not probability_dict:
        print("Error")
        return

    # Extract and sort symbols for the X-axis
    symbols = sorted(list(frequency_dict.keys()))
    
    # Extract frequencies and probabilities matching the sorted symbols
    frequencies = [frequency_dict[sym] for sym in symbols]
    probabilities = [probability_dict[sym] for sym in symbols]

    plt.style.use('default')
    
    # Create two subplots stacked vertically (sharex=True aligns the X-axes)
    fig, (ax_freq, ax_prob) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Custom background color 
    fig.patch.set_facecolor('white') 
    ax_freq.set_facecolor('white')
    ax_prob.set_facecolor('white')

    # Colors
    primary_color = '#4A90E2' # Standard solid blue for academic charts
    text_color = '#000000'    # Pure black for maximum readability
    grid_color = '#E0E0E0'    # Light gray for a subtle grid

    # Frquency
    ax_freq.bar(symbols, frequencies, width=0.8, facecolor=primary_color, alpha=1.0, edgecolor=primary_color, linewidth=0.5)
    
    # Titles and labels 
    ax_freq.set_title("1. Symbol Frequency (Count)", fontsize=14, fontweight='bold', color=text_color, y=1.02)
    ax_freq.set_ylabel("Frequency (Count)", color=text_color, fontsize=13, fontweight='bold')
    ax_freq.tick_params(axis='y', labelcolor=text_color, colors=text_color)
    
    # Configure a minimal grid
    ax_freq.grid(True, linestyle='--', color=grid_color, alpha=0.7)
    # Hide top/right spines
    ax_freq.spines['top'].set_visible(False)
    ax_freq.spines['right'].set_visible(False)

    # Bottom plot
    # Draw line with markers (points) using the same primary color
    ax_prob.plot(symbols, probabilities, color=primary_color, marker='o', markersize=9, linewidth=2.5, markerfacecolor=primary_color, markeredgecolor=primary_color, markeredgewidth=1)
    
    # Titles and labels for Probability plot
    ax_prob.set_title("2. Symbol Probability Distribution", fontsize=14, fontweight='bold', color=text_color, y=1.02)
    ax_prob.set_ylabel("Probability", color=text_color, fontsize=13, fontweight='bold')
    ax_prob.tick_params(axis='y', labelcolor=text_color, colors=text_color)
    
    # Common X-axis label
    ax_prob.set_xlabel("Chain Code Symbol", fontsize=13, fontweight='bold', color=text_color)
    ax_prob.set_xticks(symbols)
    ax_prob.tick_params(axis='x', labelcolor=text_color, colors=text_color)

    # Configure a minimal grid for the bottom plot
    ax_prob.grid(True, linestyle='--', color=grid_color, alpha=0.7)
    # Hide top/right spines
    ax_prob.spines['top'].set_visible(False)
    ax_prob.spines['right'].set_visible(False)
    
    # Ensure the Y-axis for probability starts at 0
    ax_prob.set_ylim(bottom=0)
    plt.suptitle("Chain Code Analysis: Frequency and Probability Dashboard", fontsize=16, fontweight='bold', color=text_color, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96]) 

    plt.suptitle("Chain Code Analysis", fontsize=16, fontweight='bold', color=text_color, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    return fig


def calculate_entropy(chain):
    """
    Calculate the Shannon entropy
    Returns: The Shannon entropy limit
    """
    if not chain:
        return 0.0
    
    total_symbol = len(chain)
    frequency = dict(Counter(chain))
    probability = {sym: freq / total_symbol for sym, freq in frequency.items()}
    entropy = -sum(
        p * math.log2(p)
        for p in probability.values()
        if p > 0 
        )

    return entropy

def lenght_compression_arithmetic(chain, probability_dict):
    """
    Calculate the theoretical average length of the arithmetic compression.
    Step-by-step mathematical implementation (Shannon's Entropy limit).
    Based on G. Langdon and J. Rissanen (1981).
    """
    # If the chain or the dictionary is empty, return 0.0
    if not chain or not probability_dict:
        return 0.0

    average_length = 0.0

    # Iterate through the dictionary symbol by symbol
    for symbol in probability_dict:
        prob = probability_dict[symbol]
        
        # Probability must be greater than 0 for the logarithm to exist
        if prob > 0:
            # Apply the mathematical change of base rule for base 2 logarithm:
            # log2(P) = ln(P) / ln(2)
            log2_prob = math.log(prob) / math.log(2)
            
            # Multiply the probability by its negative logarithm
            symbol_calculation = prob * (-log2_prob)
            
            # Add it to the total (Summation)
            average_length = average_length + symbol_calculation

    return average_length

#me dio dislexia arriba y en otras partes del código porque yo escribo "lenght" en lugar de length y no me acuerdo en donde

def length_huffman_compression(chain, probability_dict):
    """
    Calculate the average length using the Huffman tree.
    Return: average_length, dictionary_with_codes
    """
    if not chain or not probability_dict:
        return 0.0, 0, {}

    #We prepare the heap (tree data structure) probability, symbol and bits
    # An unique id avoid errors in heap when probabilities tie
    heap = []
    counter_id = 0
    for sym, prob in probability_dict.items():
        if prob > 0:
            heapq.heappush(heap, [prob, counter_id, [[sym, ""]]])
            counter_id += 1

    #Tree Huffman development
    while len(heap) > 1:
        low = heapq.heappop(heap)
        up = heapq.heappop(heap)

        #We asign "0" to the left branch and "1" to the right branch
        for pair in low[2]:
            pair[1] = '0' + pair[1]
        for pair in up[2]:
            pair[1] = '1' + pair[1]

        #we join the branches and bring it to the heap
        new_prob = low[0] + up[0]
        combinate_node = low[2] + up[2]

        heapq.heappush(heap, [new_prob, counter_id, combinate_node])
        counter_id += 1

    #Bring final results
    final_node = heapq.heappop(heap)[2]
    mean_length = 0.0
    total_bits = 0
    huffman_code = {}

    total_symbols = len(chain)
    
    for symbol, bits in final_node:
        mean_length += probability_dict[symbol] * len(bits)
        huffman_code[symbol] = bits

        freq = round (probability_dict[symbol]* total_symbols)
        total_bits += freq * len(bits)

    return mean_length, total_bits, huffman_code    

def calculate_perimeter_f4(f4_chain):
    """
    Calculate the perimeter using the length of the F4 chain.
      Each step in F4 equals 1 unit of distance.
    """
    return len(f4_chain)


def calculate_area(binary_matrix):
    """
   Calculate the area by counting the white pixels (255) in the binary image.
    """
    return int(np.sum(binary_matrix == 255))


def calculate_contact_perimeter(binary_matrix):
    """
   Calculate the contact perimeter: number of white pixel - black pixel adjacent pairs in N4 
   (4-neighborhood). That is, for each white pixel, count how many of its 4 neighbors are black.
    """
    rows, cols = binary_matrix.shape
    norm = (binary_matrix == 255).astype(int)
    contact = 0

    for i in range(rows):
        for j in range(cols):
            if norm[i][j] == 1:
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if norm[ni][nj] == 0:
                            contact += 1
                    else:
                        #The edge of de image counts as the background
                        contact += 1
    return contact


def calculate_discrete_compactness(area, perimeter):
    """
    Calculate the discrete compactness. 
    Formula: CD = (n - p/4) / (n - sqrt(n)) 
    where n = area, p = perimeter F4
    """
    if area <= 0:
        return 0.0
    import math
    sqrt_n = math.sqrt(area)
    denominator = area - sqrt_n
    if denominator == 0:
        return 0.0
    numerator = area - (perimeter / 4)
    return numerator / denominator


def calculate_holes(binary_matrix):
    """
    Count the holes in the object: background regions (0) completely enclosed by white pixels. 
    Use N4 for the background.
    Method: flood fill from all edges to mark the exterior background; what remains black and unmarked are the holes.
    """
    rows, cols = binary_matrix.shape
    norm = (binary_matrix == 255).astype(int)

    # Mark outer background with N4 flood fill from the edges
    exterior = np.zeros((rows, cols), dtype=bool)
    stack = []

    # Add all the outer pixels wich belong to background
    for i in range(rows):
        for j in [0, cols - 1]:
            if norm[i][j] == 0 and not exterior[i][j]:
                exterior[i][j] = True
                stack.append((i, j))
    for j in range(cols):
        for i in [0, rows - 1]:
            if norm[i][j] == 0 and not exterior[i][j]:
                exterior[i][j] = True
                stack.append((i, j))

    # Flood fill N4 para marcar todo el fondo exterior
    movements = [(-1,0),(1,0),(0,-1),(0,1)]
    while stack:
        r, c = stack.pop()
        for dr, dc in movements:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if norm[nr][nc] == 0 and not exterior[nr][nc]:
                    exterior[nr][nc] = True
                    stack.append((nr, nc))

    # The unmarked black pixels count as a hole
    # Count conected components of those pixels (N4)
    hole_pixels = (norm == 0) & (~exterior)
    visited = np.zeros((rows, cols), dtype=bool)
    num_holes = 0

    for i in range(rows):
        for j in range(cols):
            if hole_pixels[i][j] and not visited[i][j]:
                num_holes += 1
                stack2 = [(i, j)]
                visited[i][j] = True
                while stack2:
                    cr, cc = stack2.pop()
                    for dr, dc in movements:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if hole_pixels[nr][nc] and not visited[nr][nc]:
                                visited[nr][nc] = True
                                stack2.append((nr, nc))

    return num_holes


def calculate_euler(binary_matrix):
    """
    Calculate Euler characteristic: E = C - H
    C = concected components of the object (N8)
    H = holes (N4)
    """
    norm = (binary_matrix == 255).astype(int)
    C = connected_components(norm, neighbor=8)
    H = calculate_holes(binary_matrix)
    return C - H, C, H