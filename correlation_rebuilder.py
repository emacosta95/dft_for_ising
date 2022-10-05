#%%
import numpy as np
import matplotlib.pyplot as plt
data=np.load('data/correlation_1nn/train_1nn_correlation_map_h_4.5_n_150000_l_16_pbc_j_1.0.npz')

xx=data['correlation']
z=data['density']


# we save it as a C_i i+j term
r=np.arange(z.shape[-1])
#j=np.arange(z.shape[-1])
for j in range(1,z.shape[-1]):
    
    m=xx[:,r,(r+j)%z.shape[-1]]
    if j==1:
        ms=m.reshape(-1,1,z.shape[-1])
    else:
        ms=np.append(ms,m.reshape(-1,1,z.shape[-1]),axis=-2)

plt.imshow(xx[0])
plt.colorbar()
plt.show()
for i in range(z.shape[-1]-1):
    plt.plot(ms[0,i,:])
    plt.show()

print(ms.shape)
np.savez('data/correlation_1nn_rebuilt/train_1nn_correlation_map_h_4.5_150000_l_16_pbc_j_1.0.npz',density=z,correlation=ms)        
        
# %%
