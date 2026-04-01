import numpy as np
import heapq
from numba import njit, types
from numba.typed import Dict

class LeafNode:
    def __init__(self, symbol, weight, node_number):
        self.node_number = node_number
        self.symbol = symbol
        self.weight = weight
        self.left = None
        self.right = None

    def __lt__(self, other):
        if self.weight == other.weight:
            return self.node_number < other.node_number
        return self.weight < other.weight

@njit
def _pack_bits_numba(arr, sym_codes, sym_lengths):
    """
    JIT-compiled bit packer replacing python string manipulation.
    Calculates exact memory requirements and utilizes native bitwise operations.
    """
    total_bits = 0
    for i in range(len(arr)):
        total_bits += sym_lengths[arr[i]]
        
    padding = 8 - (total_bits % 8)
    if padding == 0:
        padding = 8
        
    total_bytes = (total_bits + padding) // 8 + 1 
    out = np.zeros(total_bytes, dtype=np.uint8)
    
    byte_idx = 1
    current_byte = np.uint8(0)
    bits_in_byte = np.int32(0)
    
    for i in range(len(arr)):
        sym = arr[i]
        code = sym_codes[sym]
        length = sym_lengths[sym]
        
        for b in range(length - 1, -1, -1):
            bit = np.uint8((code >> b) & np.uint64(1))
            current_byte = (current_byte << np.uint8(1)) | bit
            bits_in_byte += 1
            
            if bits_in_byte == 8:
                out[byte_idx] = current_byte
                byte_idx += 1
                current_byte = np.uint8(0)
                bits_in_byte = np.int32(0)
                
    if bits_in_byte > 0:
        current_byte = (current_byte << np.uint8(padding))
        out[byte_idx] = current_byte
    else:
        out[byte_idx] = np.uint8(0)
        
    out[0] = np.uint8(padding)
    return out


def build_huffman_tree(arrays):
    """
    Analyzes a joint set of arrays to build a single Huffman tree and 
    generates the Numba-compatible dictionaries required for compression.
    """
    # 1. Flatten arrays to find joint global frequencies
    joint_array = np.concatenate(arrays)
    symbols, counts = np.unique(joint_array, return_counts=True)
    
    queue = []
    node_number = 0
    symbol_list = []
    
    for sym, count in zip(symbols, counts):
        node_number += 1
        node = LeafNode(sym, count, node_number)
        heapq.heappush(queue, node)
        symbol_list.append(sym)
        
    left_nodes = []
    right_nodes = []
    
    # 2. Construct Huffman Tree
    while len(queue) > 1:
        left = heapq.heappop(queue)
        right = heapq.heappop(queue)
        
        node_number += 1
        parent = LeafNode(None, left.weight + right.weight, node_number)
        parent.left = left
        parent.right = right
        heapq.heappush(queue, parent)
        
        left_nodes.append(left.node_number)
        right_nodes.append(right.node_number)
        
    root = heapq.heappop(queue)
    node_max = root.node_number
    
    # 3. Generate encoding dictionary
    encoding_dict = {}
    def generate_codes(node, current_code=""):
        if node.symbol is not None:
            encoding_dict[node.symbol] = current_code
            return
        generate_codes(node.left, current_code + "0")
        generate_codes(node.right, current_code + "1")
        
    if root.left is None and root.right is None:
        generate_codes(root, "0")
    else:
        generate_codes(root)
        
    hufftree = np.concatenate(([node_max], left_nodes, right_nodes)).astype(np.int64, copy=False)
    huffsymb = np.array(symbol_list)
    
    # 4. Optimized translation matrix setup for Numba
    sym_codes = Dict.empty(key_type=types.int64, value_type=types.uint64)
    sym_lengths = Dict.empty(key_type=types.int64, value_type=types.int64)
    
    for sym, code_str in encoding_dict.items():
        sym_codes[np.int64(sym)] = np.uint64(int(code_str, 2))
        sym_lengths[np.int64(sym)] = np.int64(len(code_str))
        
    return hufftree, huffsymb, sym_codes, sym_lengths


def huffman_compress_array(arr, sym_codes, sym_lengths):
    """
    Compresses a single array using pre-calculated Numba typed dictionaries.
    """
    packed_bytes = _pack_bits_numba(arr.astype(np.int64, copy=False), sym_codes, sym_lengths)
    return bytes(packed_bytes)


@njit
def preproc_diff(arr):
    """
    Computes differences in a single pass.
    """
    n = len(arr)
    out = np.empty(n, dtype=np.int64)
    if n > 0:
        out[0] = arr[0]
        for i in range(1, n):
            out[i] = arr[i] - arr[i-1]
    return out

@njit
def preproc_digitize_and_diff(arr, npsi):
    """
    Fuses digitization and pairwise differencing into a single zero-allocation pass.
    Designed specifically for floating point angle arrays.
    """
    n = len(arr)
    out = np.empty(n, dtype=np.int64)
    if n == 0:
        return out
        
    factor = npsi / (2.0 * np.pi)
    
    # Initialize and store the first digitized value
    prev_dig = np.int64(np.round(arr[0] * factor))
    out[0] = prev_dig
    
    for i in range(1, n):
        # Digitize current value
        curr_dig = np.int64(np.round(arr[i] * factor))
        # Store difference
        out[i] = curr_dig - prev_dig
        # Update previous
        prev_dig = curr_dig
        
    return out