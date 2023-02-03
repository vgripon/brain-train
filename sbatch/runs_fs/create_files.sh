
for i in $(seq 1 7); do
  cp 0classifier_runs_fs.sh ${i}classifier_runs_fs.sh 
  cp 0finetune_runs_fs.sh ${i}finetune_runs_fs.sh
  cp 0gen_feat_runs_fs.sh ${i}gen_feat_runs_fs.sh 
done