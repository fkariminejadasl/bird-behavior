#!/bin/bash

meta_file="$HOME/meta.txt"

# save datetime
date >> "$meta_file"

# Save hash value
cd "$HOME/dev/bird-behavior"
echo "$(pwd)"
git log -1 --pretty=%h >> "$meta_file"
echo $(git log -1 --pretty=%h)

# Run the program
#python "$HOME/dev/bird-behavior/scripts/train.py"

# Save configs
#cat "$HOME/dev/bird-behavior/configs/test.yaml" >> "$meta_file"
#cat "$HOME/dev/bird-behavior/scripts/train.py" >> "$meta_file"

