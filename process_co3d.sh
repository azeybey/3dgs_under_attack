#!/bin/bash

base_path="data"

if [ -d "$base_path" ]; then
  # Iterate over each directory in the base path
  for folder in "$base_path"/*; do
    if [ -d "$folder" ]; then
      echo "Processing folder: $folder"

      object_name=$(basename "$folder")

      echo "Object name: $object_name"

      python image_util.py --directory_path "$base_path/$object_name"
      
      python gaussian-splatting/convert.py -s $base_path/$object_name

      python gaussian-splatting/train.py -s $base_path/$object_name --object_name ${object_name}_plain

      python gaussian-splatting/render.py -m output/${object_name}_plain

      python attack.py --dataset_name $object_name --true_label_name $object_name

      mv $base_path/$object_name/images $base_path/$object_name/images_plain

      mv $base_path/$object_name/images_noisy $base_path/$object_name/images

      python gaussian-splatting/train.py -s $base_path/$object_name --object_name ${object_name}_noisy_eval --eval

      mv $base_path/$object_name/images $base_path/$object_name/images_noisy

      mv $base_path/$object_name/images_plain $base_path/$object_name/images

      python gaussian-splatting/render.py -m output/${object_name}_noisy_eval

      python evaluate.py --dataset_name $object_name --true_label_name $object_name --plain_folder output/${object_name}_noisy_eval/train/ours_30000/gt --noisy_folder output/${object_name}_noisy_eval/train/ours_30000/renders

      python evaluate.py --dataset_name $object_name --true_label_name $object_name --plain_folder output/${object_name}_noisy_eval/test/ours_30000/gt --noisy_folder output/${object_name}_noisy_eval/test/ours_30000/renders

    fi
  done
else
  echo "Base path $base_path does not exist."
fi

