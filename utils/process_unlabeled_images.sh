#!/bin/bash

# switch to path of script
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

mkdir '../data/multiview'
mkdir '../data/multiview/train'
mkdir '../data/multiview/test'
mkdir '../data/multiview/val'

IFS=$'\n'
SUBJECT_DIRS=($(find ../data/first_transfer/8 -maxdepth 1))
unset IFS

echo "There are ${#SUBJECT_DIRS[@]} subject directories"

g=25
for((i=0; i < ${#SUBJECT_DIRS[@]}; i+=g))
do
  part=( "${SUBJECT_DIRS[@]:i:g}" )
  python -c "import process_unlabeled_images; process_unlabeled_images.run('${part[*]}')" &
done

#count = 20

#for i in $(seq 0 $(( ${#SUBJECT_DIRS[@]}/$count - 1 )));do
#	echo "######## Set" $i "########";
#	echo "${SUBJECT_DIRS[@]:$(( i * $count )):$count }" ;
#	for img in "${SUBJECT_DIRS[@]:$(( i * $count )):$count }"; do
#		part=( "${SUBJECT_DIRS[@]:i:g}" )
#  		python -c "import categorize_standard_dicoms_multiview; categorize_standard_dicoms_multiview.run('${part[*]}')" &
#	done
#done
