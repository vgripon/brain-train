import os
import torch
from time import sleep
os.system('ps aux | grep test_subdomains.py')
N=50   # number of subdomains 

torch.save({'info' : info}, 'results.pt')
for i in range(N):
    print("CLUSTER NUMBER  :", i )
    os.system('python main.py  --epochs 1 --dataset-path "/users2/libre/datasets/"  --load-backbone /users2/libre/clusters/clusters22 --test-dataset "["metadataset_imagenet_test"]" --backbone resnet18_large --few-shot-queries 0 --few-shot-way 0 --few-shot-shots 0 --few-shot --few-shot-runs 600 --batch-size 128')
    sleep(3)  # time to exit process