import os
from time import sleep
os.system('ps aux | grep test_subdomains.py')
N=50   # number of subdomains 

for i in range(N):
    print("CLUSTER NUMBER  :", i )
    os.system("python main.py --test-features /users2/libre/models_imagenet/clusters"+str(i)+"metadataset_aircraft_test_features.pt --test-dataset metadataset_aircraft_test --dataset-path /users2/libre/datasets/")
    sleep(3)  # time to exit process