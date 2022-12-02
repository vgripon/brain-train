import subprocess
from itertools import permutations
import numpy as np
from functools import reduce

threshold = 42
values_strides = [2,2,2,2,2,3,3,3]

list_strides = []
for i in range(2,6): 
    result = permutations(values_strides, i) 
    for k in result  : 
        if reduce(lambda x, y: x*y, k) < threshold: 
            list_strides.append(k)
list_strides = np.unique(np.array(list_strides))
print(list_strides)

architectures = []
for k in range(len(list_strides)) :
    multiplier = 1
    blocks = [[2,1,1]]
    for l in list_strides[k]:
        multiplier = multiplier * l
        blocks.append([2,l,multiplier])
    architectures.append(blocks)

print(architectures)


for archi in architectures : 
    print(str(archi))

    subprocess.run(['python3',f"main.py", "--dataset" , "metadataset_imagenet",
                    "--dataset-path", "/users2/libre/datasets/",
                    "--steps", "[['manifold mixup'],['rotations','lr']]",
                    "--backbone", "custom_resnet",
                    "--blocks", str(archi), 
                    "--leaky",
                    "--batch-size", "256",
                    "--dataset-size", "38400",
                    "--milestones", "100",
                    "--epochs", "200",
                    "--cosine",
                    "--lr", "0.1",
                    "--gamma", "0.6",
                    "--csv", str(f"trains_reda/train_reda_{''.join([str(item[1]) for item in archi])}.csv"),
                    "--few-shot",
                    "--few-shot-ways", "0", 
                    "--few-shot-shots", "0",
                    "--few-shot-queries", "0",
                    "--few-shot-runs", "600",
                    "--deterministic",
                    "--wd", "1e-4",
                    "--feature-maps", "64",
                    "--feature-processing", "E",
                    "--skip-epochs", "180",
                    "--training-image-size", "84",
                    "--test-image-size", "84",
                    "--stride_normalize",str(archi[-1][2])])

    