#================================================================================
#
# This file is part of Commander3.
#
# Commander3 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Commander3 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Commander3. If not, see <https://www.gnu.org/licenses/>.
#
#================================================================================

import healpy as hp
import numpy as np
import heapq
import os
import numba

@numba.jit(nopython=True, cache=True)
def _numba_decoder(bytarr, left_nodes, right_nodes, symbols, head_node_idx, nsymb):
    padding = bytarr[0]
    
    # Use a list to store the intermediate delta values
    decoded_deltas_list = []
    
    current_node_idx = head_node_idx

    # Decode all symbols into a list. Main loop for all bytes except the last one
    for i in range(1, len(bytarr) - 1):
        byte = bytarr[i]
        for j in range(7, -1, -1):
            bit = (byte >> j) & 1
            current_node_idx = right_nodes[current_node_idx - nsymb - 1] if bit else left_nodes[current_node_idx - nsymb - 1]
            
            if current_node_idx <= nsymb:
                decoded_deltas_list.append(symbols[current_node_idx - 1])
                current_node_idx = head_node_idx

    # Special handling for the last byte
    if len(bytarr) > 1:
        last_byte = bytarr[-1]
        num_bits_in_last_byte = 8 if padding == 0 else (8 - padding)
        for j in range(7, 7 - num_bits_in_last_byte, -1):
            bit = (last_byte >> j) & 1
            current_node_idx = right_nodes[current_node_idx - nsymb - 1] if bit else left_nodes[current_node_idx - nsymb - 1]

            if current_node_idx <= nsymb:
                decoded_deltas_list.append(symbols[current_node_idx - 1])
                current_node_idx = head_node_idx
    
    if not decoded_deltas_list:
        return np.empty(0, dtype=np.int64) # Handle empty case
        
    final_deltas = np.array(decoded_deltas_list, dtype=np.int64)
    
    return np.cumsum(final_deltas)


_node_number = 0

class LeafNode:

    def __init__(self, symbol, weight):
        global _node_number
        _node_number += 1
        self.node_number = _node_number
        self.symbol = symbol
        self.weight = weight
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.weight < other.weight


class Huffman:

    def __init__(self, infile="", nside=256, tree=None, symb=None):
        self.infile = infile
        self.nside = nside
        self.encoding = {}
        self.decoding = {}
        self.queue = []
        self.weight = {}
        self.symbols = []
        self.left_nodes = []
        self.right_nodes = []
        self.node_max = 0
        self.head_node = None

        if(tree is not None):
            self.node_max = tree[0]
            self.left_nodes = tree[1:int((len(tree) -1)/2)+1]
            self.right_nodes = tree[int((len(tree) -1)/2)+1:]

        if(symb is not None):
            self.symbols = np.array(symb)

        if(symb is not None and tree is not None):
            self.BuildTree()
            self.PrintCode(self.head_node)

    #takes the input tree and symbols and generates the tree structure
    #input arrays are structured so that node numbers 1...nsymb correspond to 
    #the symbols array
    #the left and right arrays indicate the left and right children of nodes
    #nsymb+1 ... 2*nsymb-1
    def BuildTree(self):
        nodes = {}
        nsymb = len(self.symbols)

        self.head_node = LeafNode(None, 1)
        self.head_node.node_number = self.node_max
        self.head_node.left = self.left_nodes[self.node_max-nsymb-1]
        self.head_node.right = self.right_nodes[self.node_max-nsymb-1]
        nodes[self.node_max] = self.head_node

        for node in np.append(self.left_nodes, self.right_nodes):
            currNode = LeafNode(None, 1)
            currNode.node_number = node
            if node > nsymb:
                currNode.left = self.left_nodes[node -nsymb-1]
                currNode.right= self.right_nodes[node -nsymb-1]
            else:
                currNode.symbol = self.symbols[node-1]

            nodes[node] = currNode

        for node in nodes.copy().values():
            if node.left is not None:
                node.left = nodes[node.left]

            if node.right is not None:
                node.right = nodes[node.right]

    def PixellizePointing(self, diff=True, write=False):
        angs_pol = np.loadtxt(self.infile)
        pixels = hp.ang2pix(self.nside, angs_pol[:,1], angs_pol[:,0])

        if write :
            fname, fext = os.path.splitext(self.infile)
            np.save(fname+"_n"+str(self.nside).zfill(4)+"_pixels.npy", pixels)

        if diff:
            delta = np.diff(pixels)
            delta = np.insert(delta, 0, pixels[0])
            return delta
        else :
            return pixels

    def Weights(self, array):
        weight = {}
        for d in array:
            if not d in weight:
                    weight[d] = 0
            weight[d] += 1
        return weight

    def PrintCode(self, node, code=""):

        if(node.symbol != None):
            self.encoding[node.symbol] = code
            self.decoding[code] = node.symbol
            #print(node.symbol, node.weight, code)
            return
        self.PrintCode(node.left, code + "0")
        self.PrintCode(node.right, code + "1")


    def byteCode(self, array):
        text_bin = "".join(self.encoding[d] for d in array)
        padding = 8 - len(text_bin) % 8
        text_bin += padding*"0"
        # this tells you how many bits of padding there is apparently
        text_bin = "{0:08b}".format(padding) + text_bin

        b = bytearray()
        for i in range(0, len(text_bin), 8):
            byte = text_bin[i:i+8]
            b.append(int(byte, 2))
        return b


    def GenerateCode(self, array, write=False):
        array = np.array(array).flatten()
        self.weight = self.Weights(array)

        global _node_number

        for d in self.weight:
            node = LeafNode(d, self.weight[d])
            heapq.heappush(self.queue, node)
            self.symbols.append(d)
    
        while(len(self.queue)>1):
            left_child = heapq.heappop(self.queue)
            right_child = heapq.heappop(self.queue)

            collapsed = LeafNode(None, left_child.weight + right_child.weight)
            collapsed.left = left_child
            collapsed.right = right_child
            heapq.heappush(self.queue, collapsed)

            self.left_nodes.append(left_child.node_number)
            self.right_nodes.append(right_child.node_number)

        node = heapq.heappop(self.queue)
        if(node.left == None and node.right == None): #case where there was just one symbol
            self.PrintCode(node, code='0')
        else:
            self.PrintCode(node)

        self.node_max = node.node_number
        _node_number = 0

        b = self.byteCode(array)

        if write :
            fname, fext = os.path.splitext(self.infile)
            np.save(fname+"_n"+str(self.nside).zfill(4)+"_diffpix.npy", array)
            outfile = fname + "_n"+str(self.nside).zfill(4)+"_diffpix.bin"
            with open(outfile, 'wb') as f_out :
                f_out.write(bytes(b))
            return bytes(b), outfile
        else :
            return bytes(b)

    def Decoder(self, bytarr, write=False, numba_decode=True):
        if numba_decode:  # Whether to use the much faster Numba decoder.
            nsymb = len(self.symbols)
            left_nodes = np.array(self.left_nodes, dtype=np.int64)
            right_nodes = np.array(self.right_nodes, dtype=np.int64)

            # Call Numba implementation of Huffman decompression. I think it's about ~100 times faster.
            decoded_arr = _numba_decoder(bytearray(bytarr), left_nodes, right_nodes, self.symbols, self.node_max, nsymb)
        else: 
            bytarr = bytearray(bytarr)
            binary_txt = ''.join(bin(i)[2:].rjust(8,'0') for i in bytarr)
            padding = int(binary_txt[:8], 2)
            binary_txt = binary_txt[8:-1*padding]

            decoded_arr = []
            code = ""

            for b in binary_txt:
                code += b
                if code in self.decoding:
                    d = self.decoding[code]
                    decoded_arr.append(d)
                    code = ""

            decoded_arr = np.cumsum(decoded_arr)

        if write:
            fname, fext = os.path.splitext(self.infile)
            file_out = fname + "_n"+str(self.nside).zfill(4)+"_decoded_pixels.npy"
            np.save(file_out, decoded_arr)
            return decoded_arr, file_out
        else:
            return decoded_arr
