#!/usr/bin/env python3

import numpy as np
from persistent_path_homology import *
import matplotlib.pyplot as plt

DATA_SIZE = 11
PPH_DIM   = 3

##### First Test
def euclidean_distance( x, y ):
    #x, y numpy arrays

    return sum( (x-y)**2 )

def proj0( take_a_list ):
    return np.array( [x[0] for x in take_a_list] )

def proj1( take_a_list ):
    return np.array( [x[1] for x in take_a_list] )

np.random.seed(500)
network_set = np.random.uniform( 0,1, (DATA_SIZE, 2) )

##### symmetric case
# network_weight = np.zeros( (DATA_SIZE, DATA_SIZE) )
# for i in range( DATA_SIZE ):
#     for j in range( DATA_SIZE ):
#         network_weight[i,j] = euclidean_distance( network_set[i], \


##### network_weight with random weights
network_weight = np.random.uniform(0,1,  (DATA_SIZE, DATA_SIZE) )
for i in range( DATA_SIZE ):
    network_weight[i,i] = 0

test1 = PPH( network_set, network_weight, PPH_DIM )

test1.ComputePPH()

fig, ax = plt.subplots(1,2)
const_max = max( proj0( test1.Pers[0] ).tolist() +  proj0( test1.Pers[1] ).tolist() + [1] )
plot_diag = np.linspace(0, const_max, 2)


ax[0].plot( network_set[:,0], network_set[:,1], 'P', color='blue' )
ax[0].set_title('Data')

ax[1].plot( proj0( test1.Pers[0] ), proj1( test1.Pers[0] ), 'X', color='red', label='dim = 0' )
ax[1].plot( proj0( test1.Pers[1] ), proj1( test1.Pers[1] ), 'X', color='green', label = 'dim = 1' )
ax[1].plot( plot_diag, plot_diag, color='black' )

ax[1].set_title('PPH diagrams')
ax[1].legend(loc='lower right')

plt.show()
