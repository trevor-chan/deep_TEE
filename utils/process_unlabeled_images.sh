#!/bin/bash

# switch to path of script
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

IFS=$'\n'
SUBJECT_DIRS=($(find /data/deep_tee/first_transfer -maxdepth 1))
unset IFS

echo "There are ${#SUBJECT_DIRS[@]} subject directories"

for((i=0; i < ${#SUBJECT_DIRS[@]}; i+=1))
do
  echo "starting part ${i}"
  part=( "${SUBJECT_DIRS[@]:i:1}" )
  python -c "import process_unlabeled_images; process_unlabeled_images.run('${part[*]}')"
done



# g=1000
# for((i=0; i < ${#SUBJECT_DIRS[@]}; i+=g))
# do
#   echo "starting part ${i}"
#   part=( "${SUBJECT_DIRS[@]:i:g}" )
#   python -c "import process_unlabeled_images; process_unlabeled_images.run('${part[*]}')" &
# done