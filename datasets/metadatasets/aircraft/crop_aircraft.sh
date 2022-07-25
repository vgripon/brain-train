#!/bin/bash
helpFunction()
{
   echo "RUN THIS SCRIPT TO GET THE CROP IMAGES FROM AIRCRAFT"
   echo "Usage: $0 -a dataset_path "
   echo -e "\t-a dataset_path"
   exit 1 # Exit script after printing help
}

while getopts "a:" opt
do
   case "$opt" in
      a ) dataset_path="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$dataset_path" ] 
then
   echo "the parameters are empty";
   helpFunction
fi

# Begin script in case all parameters are correct
mkdir $dataset_path/fgvc-aircraft-2013b/data/images_cropped/
echo folder $dataset_path/fgvc-aircraft-2013b/data/images_cropped/ created ! 

python crop_aircraft.py $dataset_path