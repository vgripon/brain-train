import os

list_test_sets = ['metadataset_imagenet_test', 'metadataset_vgg_flower_test', 'metadataset_cub_test', 'metadataset_aircraft_test', 'metadataset_fungi_test', 'metadataset_dtd_test', 'metadataset_omniglot_test', 'metadataset_mscoco_test', 'metadataset_traffic_signs_test','metadataset_quickdraw_test']

#list_test_sets = [ 'metadataset_vgg_flower_test', 'metadataset_cub_test', 'metadataset_aircraft_test', 'metadataset_fungi_test', 'metadataset_dtd_test', 'metadataset_omniglot_test', 'metadataset_mscoco_test', 'metadataset_traffic_signs_test','metadataset_quickdraw_test']
list_test_sets = [ 'metadataset_quickdraw_test']
list_test_sets = [ 'metadataset_mscoco_test']


N=50   # number of subdomains 
for test_set in list_test_sets:
    os.system('python main.py  --epochs 1 --dataset-path "/users2/libre/datasets/" --feature-processing "ME" --load-backbone /users2/local/meta_imagenet2.pt --test-dataset ' + test_set + ' --backbone resnet18_large --few-shot-queries 0 --few-shot-way 0 --few-shot-shots 0 --few-shot --few-shot-runs 600 --batch-size 128 --save-features-prefix /users2/libre/features/baseline')
