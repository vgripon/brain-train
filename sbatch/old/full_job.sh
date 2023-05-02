


sbatch gen_feat_baseline.sh

sem=$(sbatch classifier_sem.sh | awk '{print $4}')
vis=$(sbatch classifier_vis.sh | awk '{print $4}')
random=$(sbatch classifier_random.sh | awk '{print $4}')


fsem=$(sbatch --dependency=afterany:$sem finetune_sem.sh | awk '{print $4}')
fvis=$(sbatch --dependency=afterany:$vis finetune_vis.sh | awk '{print $4}')
frandom=$(sbatch --dependency=afterany:$random finetune_random.sh | awk '{print $4}')

gensem=$(sbatch --dependency=afterany:$fsem gen_feat_sem.sh | awk '{print $4}')
genvis=$(sbatch --dependency=afterany:$fvis gen_feat_vis.sh | awk '{print $4}')
genrandom=$(sbatch --dependency=afterany:$frandom gen_feat_random.sh | awk '{print $4}')


id_backboneMD=$(sbatch --dependency=afterany:$genrandom id_backboneMD.sh | awk '{print $4}')

gen_fs=$id_backboneMD

cd runs_fs
for i in $(seq 0 7); do
  runs_fs=$(sbatch  --dependency=afterany:$gen_fs ${i}classifier_runs_fs.sh | awk '{print $4}')
  finetune_fs=$(sbatch --dependency=afterany:$runs_fs ${i}finetune_runs_fs.sh | awk '{print $4}')
  gen_fs=$(sbatch --dependency=afterany:$finetune_fs ${i}gen_feat_runs_fs.sh | awk '{print $4}')
done
cd ..
final=$(sbatch --dependency=afterany:$gen_fs id_backbone_episode.sh | awk '{print $4}')
