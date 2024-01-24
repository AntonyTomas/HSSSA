import numpy as np
data = np.load(r'F:\YDQ\SAMHSA\2006_2015_51X51_THSUV.npy')
data = np.array(data,dtype=np.float32)
data = np.reshape(data, [10*365,5,51,51])
data = data[:,0:1,-48:,-48:]

data_nan = data[0,0,:,:]
nan_mask = np.ones([48,48],dtype=np.float32)
for i in range(48):
    for j in range(48):
        if data_nan[i,j]<-500:
            nan_mask[i,j] = 0
np.save(r'F:\YDQ\SAMHSA\nan_mask.npy',nan_mask)
data = data*nan_mask
np.save(r'F:\YDQ\SAMHSA\data_use.npy',data)