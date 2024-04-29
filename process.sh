#!/bin/bash

object_name="apple"

#mkdir data/$object_name/input

#ffmpeg -i data/$object_name/1.mp4 -qscale:v 1 -qmin 1 -vf fps=2 data/$object_name/input/%04d.jpg

#python image_util.py --directory_path "data/$object_name/input"

#python gaussian-splatting/convert.py -s data/$object_name

#python gaussian-splatting/train.py -s data/$object_name --object_name ${object_name}_plain

#python gaussian-splatting/render.py -m output/${object_name}_plain

##python attack.py --dataset_name $object_name --true_label_name $object_name

#mv data/$object_name/images data/$object_name/images_plain

#mv data/$object_name/images_noisy data/$object_name/images

#python gaussian-splatting/train.py -s data/$object_name --object_name ${object_name}_noisy

python gaussian-splatting/render.py -m output/${object_name}_noisy

python evaluate.py --dataset_name $object_name --true_label_name $object_name --plain_folder output/${object_name}_plain/train/ours_30000/renders --noisy_folder output/${object_name}_noisy/train/ours_30000/renders