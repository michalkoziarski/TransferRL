#!/bin/bash

worlds=("goal" "coin" "water" "fire" "wall" "portal" "door")

for i in {1..3}; do
    for j in {1..6}; do
        world=${worlds[j]}
        curriculum=${worlds[j-1]}
        model_name="curriculum_${world}_${i}"
        curriculum_name="final_${curriculum}_${i}"
        world_path="world_${world}.json"

        sbatch curriculum.sh $model_name $world_path $curriculum_name
    done
done
