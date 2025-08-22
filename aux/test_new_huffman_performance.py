import numpy as np
import cmdr4_support
import time 
import sys
from tqdm import trange
sys.path.append("/mn/stornext/d23/cmbco/jonas/c4_testing/Commander4/src/python/utils/")
sys.path.append("/mn/stornext/d23/cmbco/jonas/c4_testing/Commander4/")
from commander_tod import commander_tod
from huffman import Huffman

oids = []
pids = []
filenames = []
with open("/mn/stornext/d16/cmbco/bp/mathew/test/filelist_30.txt") as infile:
    infile.readline()
    for line in infile:
        pid, filename, _, _, _ = infile.readline().split()
        pids.append(f"{int(pid):06d}")
        filenames.append(filename[1:-1])
        oids.append(filename.split(".")[0].split("_")[-1])
pids = np.array(pids)
oids = np.array(oids)

psi_encoded_list = []
pix_encoded_list = []
hufftree_list = []
huffsymb_list = []
for i in trange(1000):
    pid = pids[i]
    oid = oids[i]
    com_tod = commander_tod("/mn/stornext/d16/cmbco/bp/mathew/test/", "LFI")
    com_tod.init_file("070", oid)
    psi_encoded = com_tod.load_field(f"/{pid}/18M/psi/")[()]
    pix_encoded = com_tod.load_field(f"/{pid}/18M/pix/")[()]
    hufftree = com_tod.load_field(f"/{pid}/common/hufftree")[()]
    huffsymb = com_tod.load_field(f"/{pid}/common/huffsymb")[()]
    psi_encoded_list.append(psi_encoded)
    pix_encoded_list.append(pix_encoded)
    huffsymb_list.append(huffsymb)
    hufftree_list.append(hufftree)


sizes = []
# Comparison loop
for i in trange(1000):
    huff = Huffman(tree=hufftree_list[i], symb=huffsymb_list[i])
    pix1 = huff.Decoder(pix_encoded_list[i], numba_decode=True)
    psi1 = huff.Decoder(psi_encoded_list[i], numba_decode=True)
    sizes.append(pix1.size)
    pix_encoded = np.frombuffer(pix_encoded_list[i], dtype=np.uint8)
    pix0 = np.empty(sizes[i], dtype=np.int64)
    pix0 = cmdr4_support.utils.huffman_decode(pix_encoded, hufftree_list[i], huffsymb_list[i], pix0)
    pix0 = np.cumsum(pix0)
    psi_encoded = np.frombuffer(psi_encoded_list[i], dtype=np.uint8)
    psi0 = np.empty(sizes[i], dtype=np.int64)
    psi0 = cmdr4_support.utils.huffman_decode(psi_encoded, hufftree_list[i], huffsymb_list[i], psi0)
    psi0 = np.cumsum(psi0)
    if (np.max(np.abs(pix0-pix1))) != 0:
        raise RuntimeError("pix mismatch")
    if (np.max(np.abs(psi0-psi1))) != 0:
        raise RuntimeError("psi mismatch")

# Benchmark Numba version
t0 = time.time()
for i in trange(1000):
    huff = Huffman(tree=hufftree_list[i], symb=huffsymb_list[i])
    pix1 = huff.Decoder(pix_encoded_list[i], numba_decode=True)
    psi1 = huff.Decoder(psi_encoded_list[i], numba_decode=True)
print(f"old version finished in {time.time()-t0:.2f}s.")

# benchmark C++ version
t0 = time.time()
for i in trange(1000):
    pix_encoded = np.frombuffer(pix_encoded_list[i], dtype=np.uint8)
    pix0 = np.empty(sizes[i], dtype=np.int64)
    pix0 = cmdr4_support.utils.huffman_decode(pix_encoded, hufftree_list[i], huffsymb_list[i], pix0)
    pix0 = np.cumsum(pix0)
    psi_encoded = np.frombuffer(psi_encoded_list[i], dtype=np.uint8)
    psi0 = np.empty(sizes[i], dtype=np.int64)
    psi0 = cmdr4_support.utils.huffman_decode(psi_encoded, hufftree_list[i], huffsymb_list[i], psi0)
    psi0 = np.cumsum(psi0)
print(f"cmdr4_support version finished in {time.time()-t0:.2f}s.")
