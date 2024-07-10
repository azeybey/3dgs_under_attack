#!/bin/bash

directory="data/house"

mkdir $directory/images_raw

# Check if the directory exists
if [ -d "$directory" ]; then
    # Loop through each file in the directory
    counter=0
    for file in "$directory"/*
    do
        ((counter++))
        # Check if it's a file (not a directory)
        if [ -f "$file" ]; then
            # Print the file name
            echo "Processing file: $file"
            
            ffmpeg -i $file -qscale:v 1 -qmin 1 -vf "fps=1,scale=1600:-1" $directory/input/$counter%04d.jpg

        fi
    done
else
    echo "Directory not found: $directory"
fi
