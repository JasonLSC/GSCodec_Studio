# ----------------- Training Setting-------------- #
SCENE_DIR="data/neural_3d"
SCENE_LIST="coffee_martini cook_spinach cut_roasted_beef flame_salmon_1 flame_steak sear_steak" # cook_spinach
# SCENE_LIST="coffee_martini cook_spinach"

RESULT_DIR="results/stg_neu3d_bs4"
# ----------------- Training Setting-------------- #

# ----------------- Main Job --------------------- #
run_single_scene() {
    local GPU_ID=$1
    local SCENE=$2

    echo "Running $SCENE"

    CUDA_VISIBLE_DEVICES=$GPU_ID python simple_trainer_STG.py default \
        --model_path $RESULT_DIR/$SCENE/ \
        --data_dir $SCENE_DIR/$SCENE/colmap_0 \
        --result_dir $RESULT_DIR/$SCENE/ \
        --batch_size 4

}

# ----------------- Main Job --------------------- #

# ----------------- Experiment Loop -------------- #
GPU_LIST=(0 1 2 4 6 7)
GPU_COUNT=${#GPU_LIST[@]}

SCENE_IDX=-1

for SCENE in $SCENE_LIST;
do
    SCENE_IDX=$((SCENE_IDX + 1))
    {
        run_single_scene ${GPU_LIST[$SCENE_IDX]} $SCENE
    } &

done

# ----------------- Experiment Loop -------------- #

# Wait for finishing the jobs across all scenes 
wait

echo "All scenes finished."
