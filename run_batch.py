from encode import main, plot_batch
import pandas as pd
from IPython.display import display
import numpy as np



START, END = 4,64

n_nodes = START

dts = []
print(dts)

while n_nodes <= END:
    dt = [n_nodes]
    for network_mode in range(3):
        dt.append(main(n_nodes=n_nodes, network_mode=network_mode))
    dts.append(dt)
    print(len(dt))
    n_nodes = n_nodes * 2
#print(dts)


df = pd.DataFrame(dts, columns=["n_nodes","autonomous", "collaborative", "mixed"])


x = df.values[:,0]
x = list(x)
ys = []
ys.append(df.values[:,1])
ys.append(df.values[:,2])
ys.append(df.values[:,3])

plot_batch(x,ys)







