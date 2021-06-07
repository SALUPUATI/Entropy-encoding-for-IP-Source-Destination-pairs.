# Entropy-encoding-for-IP-Source-Destination-pairs.


The goal of this work is to be able to compress the frame to allow the transit of IP over very low rate underwater acoustic links, using entropy coding.

We considered a star topology, with variable nodes, by doing a Source-Destination IP address pairing.

# Steps to follow:

### HUFFMAN CODING

1. First, we created the source-destination symbol matrix;
2. Next, created a probability distribution of the source-destination symbols in our matrix, and scaled this probability distribution so that we have a sum of 1 at the transmission, considering our star topology; 
3. And further on, we did the probability of occurrence of the different source-destination pairs (symbols) of our scaled probability distribution;
4. And finally we calculated the entropy, the length of the Huffman code, and created the Huffman tree, and our code dictionary, to extract the compressed code.

## Running of the encode.py script

We have three running modes (autonomous network, collaborative network and mixed network)

1. For the autonomous network :

        # python3 encode.py --n_nodes=6 or 8 till 32 --verbose=true --mode=0

2. For the collaborative network :

        # python3 encode.py --n_nodes=6 or 8 till 32 --verbose=true --mode=1

3. For the mixed network :

        # python3 encode.py --n_nodes=6 or 8 till 32 --verbose=true --mode=2

Enjoy
       

