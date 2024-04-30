#!/bin/bash

# switch to path of script
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

mkdir '../data/supervised/multiview'
mkdir '../data/supervised/multiview/train'
mkdir '../data/supervised/multiview/test'
mkdir '../data/supervised/multiview/val'

IFS=$'\n' 
SUBJECT_DIRS=($(find /data/deep_tee/raw_data/train_test/*/ -maxdepth 1))
unset IFS

echo "There are ${#SUBJECT_DIRS[@]} subject directories"

g=2
for((i=0; i < ${#SUBJECT_DIRS[@]}; i+=g))
do
  part=( "${SUBJECT_DIRS[@]:i:g}" )
  python -c "import categorize_standard_dicoms_multiview; categorize_standard_dicoms_multiview.run('${part[*]}')" &
done

wait

