#!/bin/bash

worlds=("goal" "coin" "water" "fire" "wall" "portal" "door")

for i in {1..3}; do
    for j in {0..6}; do
        world=${worlds[j]}
        model_name="final_${world}_${i}"
        world_path="world_${world}.json"

        sbatch batch.sh $model_name $world_path
    done
done
