import os
suffix = '--backbone resnet12  --wandb raflaf --wandb-dir wandb --wandbProjectName brain-train --wd 0.0001'

sl12 = False

if sl12:
    os.system('python main.py --dataset-path /users2/libre/datasets/  --training-dataset metadataset_imagenet_train  --freeze-backbone --force-train --epoch 20 --lr 0.05 --load-backbone /users2/libre/raphael/resnet12_metadataset_imagenet_64.pt --save-classifier /users2/libre/raphael/classifier_metadataset_imagenet_64.pt --backbone resnet12 '+ suffix)
    for dataset in ['cub', 'aircraft', 'dtd', 'mscoco', 'fungi', 'omniglot', 'traffic_signs', 'vgg_flower']:
        os.system('python main.py --dataset-path /users2/libre/datasets/  --validation-dataset metadataset_{0}_validation  --freeze-backbone --freeze-classifier --load-classifier /users2/libre/raphael/classifier_metadataset_imagenet_64.pt --load-backbone /users2/libre/raphael/resnet12_metadataset_imagenet_64.pt --save-logits /users2/libre/raphael/logits_{0}_val.pt --backbone resnet12'.format(dataset))
else:
    os.system('python create_dataset_files.py --dataset-path /home/datasets/')
    #os.system('python main.py --dataset-path /home/datasets/  --training-dataset metadataset_imagenet_train  --freeze-backbone --force-train --epoch 20 --lr 0.05 --load-backbone /home/raphael/Documents/models/resnet12_metadataset_imagenet_64.pt --save-classifier /home/raphael/Documents/models/classifier_metadataset_imagenet_64.pt --backbone resnet12 '+ suffix)
    for dataset in ['cub', 'aircraft', 'dtd', 'mscoco', 'fungi', 'omniglot', 'traffic_signs', 'vgg_flower']:
        os.system('python main.py --dataset-path /home/datasets/ --epoch 1 --training-dataset metadataset_imagenet_train --validation-dataset metadataset_{0}_validation  --freeze-backbone --freeze-classifier --load-classifier /home/raphael/Documents/models/classifier_metadataset_imagenet_64.pt --load-backbone /home/raphael/Documents/models/resnet12_metadataset_imagenet_64.pt --save-logits /home/raphael/Documents/models/logits_{0}_val.pt --backbone resnet12'.format(dataset))