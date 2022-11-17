python id_backbone.py --target-dataset "['cub','aircraft','vgg_flower','dtd', 'fungi','traffic_signs','omniglot','mscoco']" --proxy fake_acc --seed 1 --few-shot-shots 0 --few-shot-ways 0 --few-shot-queries 0 --few-shot-runs 10000 --save-features-prefix /users2/local/features/ --dataset-path /users2/local/datasets/ --num-clusters 10 --max-queries;
echo fake_acc is_done;

python id_backbone.py --target-dataset "['cub','aircraft','vgg_flower','dtd', 'fungi','traffic_signs','omniglot','mscoco']" --proxy fake_acc --seed 1 --few-shot-shots 0 --few-shot-ways 0 --few-shot-queries 0 --few-shot-runs 10000 --save-features-prefix /users2/local/features/ --dataset-path /users2/local/datasets/ --num-clusters 10 --max-queries --QR;
echo fake_acc QR is_done;

python id_backbone.py --target-dataset "['cub','aircraft','vgg_flower','dtd', 'fungi','traffic_signs','omniglot','mscoco']" --proxy fake_acc --seed 1 --few-shot-shots 0 --few-shot-ways 0 --few-shot-queries 0 --few-shot-runs 10000 --save-features-prefix /users2/local/features/ --dataset-path /users2/local/datasets/ --num-clusters 10 --max-queries --QR --isotropic;
echo fake_acc QR isotropic is_done;


python id_backbone.py --target-dataset "['cub','aircraft','vgg_flower','dtd', 'fungi','traffic_signs','omniglot','mscoco']" --proxy soft --seed 1 --few-shot-shots 0 --few-shot-ways 0 --few-shot-queries 0 --few-shot-runs 10000 --save-features-prefix /users2/local/features/ --dataset-path /users2/local/datasets/ --num-clusters 10 --max-queries;
echo soft is_done;
python id_backbone.py --target-dataset "['cub','aircraft','vgg_flower','dtd', 'fungi','traffic_signs','omniglot','mscoco']" --proxy hard --seed 1 --few-shot-shots 0 --few-shot-ways 0 --few-shot-queries 0 --few-shot-runs 10000 --save-features-prefix /users2/local/features/ --dataset-path /users2/local/datasets/ --num-clusters 10 --max-queries;
echo hard is_done;

python id_backbone.py --target-dataset "['cub','aircraft','vgg_flower','dtd', 'fungi','traffic_signs','omniglot','mscoco']" --proxy loo --seed 1 --few-shot-shots 0 --few-shot-ways 0 --few-shot-queries 0 --few-shot-runs 10000 --save-features-prefix /users2/local/features/ --dataset-path /users2/local/datasets/ --num-clusters 10 --max-queries;
echo loo is_done;
python id_backbone.py --target-dataset "['cub','aircraft','vgg_flower','dtd', 'fungi','traffic_signs','omniglot','mscoco']" --proxy snr --seed 1 --few-shot-shots 0 --few-shot-ways 0 --few-shot-queries 0 --few-shot-runs 10000 --save-features-prefix /users2/local/features/ --dataset-path /users2/local/datasets/ --num-clusters 10 --max-queries;
echo snr is_done;

python id_backbone.py --target-dataset "['cub','aircraft','vgg_flower','dtd', 'fungi','traffic_signs','omniglot','mscoco']" --proxy rankme --centroids --seed 1 --few-shot-shots 0 --few-shot-ways 0 --few-shot-queries 0 --few-shot-runs 10000 --save-features-prefix /users2/local/features/ --dataset-path /users2/local/datasets/ --num-clusters 10 --max-queries;
echo rankme is_done;
python id_backbone.py --target-dataset "['cub','aircraft','vgg_flower','dtd', 'fungi','traffic_signs','omniglot','mscoco']" --proxy rankme --seed 1 --few-shot-shots 0 --few-shot-ways 0 --few-shot-queries 0 --few-shot-runs 10000 --save-features-prefix /users2/local/features/ --dataset-path /users2/local/datasets/ --num-clusters 10 --max-queries;
echo rankme is_done centroids;

python id_backbone.py --target-dataset "['cub','aircraft','vgg_flower','dtd', 'fungi','traffic_signs','omniglot','mscoco']" --proxy rankme_t --seed 1 --few-shot-shots 0 --few-shot-ways 0 --few-shot-queries 0 --few-shot-runs 10000 --save-features-prefix /users2/local/features/ --dataset-path /users2/local/datasets/ --num-clusters 10 --max-queries;
echo rankme transductive is_done;
