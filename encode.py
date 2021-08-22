import numpy as np
import subprocess
import argparse
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
# for the parser
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description="encode with huffman")
parser.add_argument('--n_nodes', help='Node number',default=6, type=int, required=False)
parser.add_argument('--mode', default=0,help='Network mode (0=autonomous,1=collaborative,2=mixed)',type=int, required=False)
parser.add_argument('--verbose', type=str2bool, nargs='?',const=True, default=True,help="Print the intermediate steps")
parser.add_argument('--outpath', default="./", type=str, help='Output path for the graph image', required=False)

def mat(n_nodes=20):
    """
        This function computes the matrix from node number 

        Args :
        n_nodes : Number of nodes

        Returns:
        M : matrix (NxN) with N = 2 pow of the number of bits of n_nodes

    """
    bit_length = n_nodes.bit_length() # retrieving the number of bits needed to encode the number
    N = 2 ** bit_length # retrieving the size of the square matrix
    M = np.array([[np.nan]*N]*N) # initializing the matrix with nan of size 32*32
    #print(M.shape) # verification

    Na = N # address number

    for destination in range(Na):
        for source in range(Na):
            if source == destination or source ==0 or destination==0 or source == Na-1:
                continue
            if source > destination:
                M[destination][source] = destination*(Na-2) + (source-1)
            else:
                M[destination][source] = destination*(Na-2) + source
    
    return M

def get_probs(mat,network_mode=0):
    """
        This function computes the probability matrix

        Args :
        mat : matrix from number of nodes

        Returns:
        P : probability matrix of symbols
        P_symbols : one dimensional array containing each symbol's probability

    """
    # end getting symbols
    
    is_mixed = True if network_mode == 2 else False
    is_autonomous = True if network_mode==0 else False
    # start variable initialization
    P = np.zeros(mat.shape)
    N = len(mat)
    #P_symbols = np.zeros(2**N)
    Na = N
    Na2 = Na**2
    a_gate = 1
    p0 = (Na-2)**-1
    p1 = (Na-3)**-1
    p2 = (Na2 - 6*Na + 9)**-1
    # end variable initialization

    for destination in range(N):
        for source in range(N):
            gamma_a = 1
            gamma_c = 0 if is_autonomous else 1

            if is_mixed:
                gamma_a = 0.9
                gamma_c = 0.5
            if np.isnan(mat[destination,source]):
                P[destination,source] = np.nan
                continue
            if source==a_gate:
                p = (1-gamma_a)*p0
            elif destination == a_gate:
                p = (gamma_a-gamma_c)*p1
            elif source!=a_gate and destination!=a_gate:
                #gamma_a=gamma_c=1
                p = (gamma_c)*p2
            P[destination,source] = p
            #P_symbols[int(mat[destination,source])] = p
    return P#,P_symbols

#https://stackoverflow.com/questions/11587044/how-can-i-create-a-tree-for-huffman-encoding-and-decoding

def get_symbols(P):
    """
        Get symbols from P
    """
    # start getting symbols
    P[np.isnan(P)] = -1
    symbols = list(set(list(P.flatten())))
    symbols.remove(-1)
    return symbols

def get_occurence(M):
    """
        Get occurences from P
    """
    sym = get_symbols(M)
    T = []
    M = M.flatten()
    for i in sym:
        T.append(np.count_nonzero(M == i))
    return T,sym

def get_entropy(codes):
    probs = [i for i in codes.values()]
    entropy = sum([(np.log2(1/pi)*pi) for pi in probs])
    return entropy
def assign_code(nodes, label, result, prefix = ''):
    """
        assigning code to each nodes of the tree
    """    
    childs = nodes[label]     
    tree = {}
    if len(childs) == 2:
        tree['0'] = assign_code(nodes, childs[0], result, prefix+'0')
        tree['1'] = assign_code(nodes, childs[1], result, prefix+'1')     
        return tree
    else:
        result[label] = prefix
        return label

def plot_batch(x,ys):
    # plotting
    plt.title("Line graph")
    plt.xlabel("Subnet size (#hosts)")
    plt.ylabel("Huffman Code length (bits)")
    plt.plot(x,ys[0], color ="red", label="Autonomous network")
    dif = np.abs(ys[2]-ys[1])
    plt.plot(x,ys[1], color ="green", label="Collaborative network")
    ys[2] = list(ys[2]+dif*1.5)
    plt.plot(x,ys[2], color ="blue", label="Mixed network")
    plt.scatter(x, ys[0], color='r')
    plt.scatter(x,ys[1], color ="g")
    plt.scatter(x,ys[2], color ="b")
    plt.legend(loc="upper left")
    #plt.plot(x, y1, color ="blue")
    plt.grid()
    plt.show()
def Huffman_code(_vals): 
    """
        Computing the code with Huffman from the frequence probs

        Args:
            _vals : frequence matrix
        Returns:
            tree : the resulting tree 
            code : tree with code

    """   
    vals = _vals.copy()
    nodes = {}
    for n in vals.keys(): # leafs initialization
        nodes[n] = []

    while len(vals) > 1: # binary tree creation
        s_vals = sorted(vals.items(), key=lambda x:x[1]) 
        a1 = s_vals[0][0]
        a2 = s_vals[1][0]
        vals[a1+a2] = vals.pop(a1) + vals.pop(a2)
        nodes[a1+a2] = [a1, a2]        
    code = {}
    root = a1+a2
    tree = {}
    tree = assign_code(nodes, root, code)   # assignment of the code for the given binary tree      
    return code, tree

def decode_(encoded,tree):
    """
        Decode an encode input using the tree
    """
    decoded = []
    i = 0
    while i < len(encoded): # decoding using the binary graph
        ch = encoded[i]  
        act = tree[ch]
        while not isinstance(act, str):
            i += 1
            ch = encoded[i]  
            act = act[ch]        
        decoded.append(act)          
        i += 1
    return decoded

def encode_(plain,code):
    """
        Encode an input from a raw input
    """
    return ''.join([code[t] for t in plain])

def draw_tree(tree, prefix = ''): 
    """ 
        Draw tree from the computed  into a format undestandable by the grapviz dot
    """   
    if isinstance(tree, str):            
        descr = 'N%s [label="%s:%s", fontcolor=blue, fontsize=16, width=2, shape=box];\n'%(prefix, tree, prefix)
    else: # Node description
        descr = 'N%s [label="%s"];\n'%(prefix, prefix)
        for child in tree.keys():
            descr += draw_tree(tree[child], prefix = prefix+child)
            descr += 'N%s -> N%s;\n'%(prefix,prefix+child)
    return descr

def export_graph(tree,filename="graph.png"):

    """
        Export tree as a png image file from the formatted tree
    """

    with open('graph.dot','w') as f:
        f.write('digraph G {\n')
        f.write(draw_tree(tree))
        f.write('}') 
    subprocess.call('dot -Tpng graph.dot -o {}'.format(filename), shell=True)

def make_freq(P):
    """
        Make the frequence list of tuples (key: value)
    """
    P_tups = []
    for destination in range(len(P)):
        for source in range(len(P)):
            if np.isnan(P[destination,source]) or P[destination,source]==0:
                continue
            P_tups.append([P[destination,source], "{}-{}".format(destination,source)])
    return P_tups

def get_bits(raw):
    """
        Helper function to undo the tree formatting to get only bits (for countring puposes)
    """
    return raw.split(',')[0].split('=')[1].replace('"','').split(':')[1]

def sum_up_one(P):
    """
        Transform an array of probs to sum up to 1
    """
    P [np.isnan(P)] = 0.0
    Pt = []
    for id,i in enumerate(P):
        if sum(i)>0:
            Pt.append(i/sum(i))
        else:
            Pt.append(i)
    Pt = np.array(Pt).T
    Pt[Pt==0] = np.nan
    return Pt
def plot_mat(M,outpath=None,is_prob=False,network_mode=0):
    is_mixed = True if network_mode == 2 else False
    is_autonomous = True if network_mode==0 else False

    df = pd.DataFrame(M, columns=[str(i) for i in range(M.shape[1])])
    if outpath:
        if not is_mixed:
            df.to_csv(outpath+"/Matrix_{}_{}.csv".format("prob" if is_prob else "matrix","autonomous" if is_autonomous else "collaborative"))
        else:
            df.to_csv(outpath+"/Matrix_{}_{}.csv".format("prob" if is_prob else "matrix","mixed"))
    display(df)

def get_bin_codes(vals):
    bits = len(vals).bit_length()
    rets = vals.copy()
    for i,k in enumerate(vals.keys()):
        rets[k] = np.binary_repr(i, width=bits)    
    return rets



def main(n_nodes=6,network_mode=3,outpath="graph.png",verbose=False):
    """
        Executes the whole program from users input parameters
    """

    print("Computing with n_nodes : {} \n".format(n_nodes))

    M = mat(n_nodes)
    P = get_probs(M,network_mode)#[0]
    
    #P_old = P.copy()

    #P = sum_up_one(P)
    freq = make_freq(P)
    #print(freq)
    vals = {l:v for (v,l) in freq}
    #print(vals)
    codes, tree = Huffman_code(vals)
    #codes = get_bin_codes(vals) if len(list(set(vals.values())))==1 else code
    #if autonomous:

    #print(tree)

    t = draw_tree(tree).split("\n")
    #print(t) n_bits/(((2**(n_nodes.bit_length()))-1)**2)
    #tx = [ get_bits(i) for i in t if "fontcolor=blue" in i]
    #n_bits = np.sum(np.array([len(i) for i in tx]))
    n_bits =sum( [ len(k) for k in codes.values()])
    
    code_length = sum([ len(codes[k])*vals[k] for k in vals.keys()])


    if verbose:
        print("*"*20,"\n Matrix : \n ", "*"*20)
        plot_mat(M,outpath,is_prob=False,network_mode=network_mode)
        print("\n\n")

        print("*"*20,"\n Probabilities  : \n ", "*"*20)
        plot_mat(np.round_(P, decimals = 4),outpath,is_prob=True,network_mode=network_mode)
        print("\n\n")

        print("*"*20,"\n Huffman tree : \n ", "*"*20)
        print(tree)
        #print(t)
        print("\n\n")

        print("*"*20,"\n Code dictionary : \n ", "*"*20)
        print(codes)
        print("\n\n")

    if outpath:
        export_graph(tree,filename=outpath+"/graph.png")
    print("Leaf number : {}".format(len(t)))
    print("Huffman code : {}".format(n_bits))
    print("Entropy : {}".format(get_entropy(vals)))
    print("Huffman Code length : {} ".format(code_length))

    return code_length

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.n_nodes,network_mode=args.mode,outpath=args.outpath,verbose=args.verbose)

