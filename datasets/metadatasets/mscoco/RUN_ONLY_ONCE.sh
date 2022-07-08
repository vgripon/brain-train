#!/bin/bash
echo "RUN THIS SCRIPT TO GET THE CROPPED IMAGES FROM MSCOCO AND THE JSON FILE ASSOCIATED WITH IT (at the image level not the class level) \n It's very long run it only once"
helpFunction()
{
   echo "RUN THIS SCRIPT TO GET THE CROPPED IMAGES FROM MSCOCO AND THE JSON FILE ASSOCIATED WITH IT"
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
mkdir $dataset_path/metadatasets/mscoco/cropped_imgs
echo folder $dataset_path/metadatasets/mscoco/cropped_imgs created ! 

python crop_mscoco.py $dataset_path



