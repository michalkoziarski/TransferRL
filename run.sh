#!/bin/bash

worlds=("goal" "coin" "water" "fire" "wall" "portal" "door")

for i in {1..3}; do
    for world in ${worlds[@]}; do
        model_name="${world}_${i}"
        world_path="world_${world}.json"

        sbatch batch.sh $model_name $world_path
    done
done
