






cd runs_fs
for i in $(seq 3 7); do
  sbatch   ${i}classifier_runs_fs.sh 
  sbatch ${i}finetune_runs_fs.sh 
  sbatch  ${i}gen_feat_runs_fs.sh 
done
cd ..
sbatch  id_backbone_episode.sh 
