#!/bin/bash

export TRAIN_CONFIG="phase_1_train_config.yml"
export MODEL_CONFIG="phase_1_model_vanillaccn.yml"
export DATASET_CONFIG="single-distance-up-dataset.yml"
export SCRIPT="experiments/scene-ccn/spatial-transformer-ccn/train/phase_1_train_ccn.py"

OBJECTS=(002_master_chef_can 003_cracker_box 004_sugar_box 005_tomato_soup_can 006_mustard_bottle)

for OBJECT in ${OBJECTS[@]}; do
        echo "python $SCRIPT --train_config_path $TRAIN_CONFIG --model_config_path $MODEL_CONFIG --dataset_config_path $DATASET_CONFIG --object_name $OBJECT"
done
