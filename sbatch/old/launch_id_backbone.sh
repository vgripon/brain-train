
cd ..
dat=$@
valtest="validation"
dirvis="working_dirs/work_dir/vis/features/${dat}/"
dirsem="working_dirs/work_dir/sem/features/${dat}/"
dirrandom="working_dirs/work_dir/random/features/${dat}/"
directories=($dirvis $dirsem $dirrandom)
result="["
count=0

for dir in "${directories[@]}"; do
  echo $dir
  files=$(find "$dir" -type f -name "*$valtest*") 
  for file in $files; do
    result="$result'$file',"
    count=$((count+1))
  done
done
result="${result%,}]"
result="\"$result\""

python id_backbone.py --dataset-path /home/datasets/ --target-dataset $@ --proxy hnm --valtest validation --num-clusters $count --competing-features $result --few-shot-shots 0 --few-shot-queries 0  --few-shot-runs 10000 --out-file working_dirs/result.pt