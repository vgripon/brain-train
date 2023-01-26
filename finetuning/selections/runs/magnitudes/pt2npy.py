import torch
import numpy as np

k = 50

for dataset in ['cub', 'aircraft', 'dtd',  'fungi', 'omniglot', 'traffic_signs', 'vgg_flower', 'mscoco',]:
    a = torch.load('magnitude_{}.pt'.format(dataset))['ord']
    b = np.ones(a.shape,dtype=np.bool)
    for i in range(a.shape[0]):
        b[i,a[i,:k].cpu()]=0
        #print(b[i])

    np.save('binary_{}.npy'.format(dataset), b)