# ----------------- Training Setting-------------- #
SCENE_DIR="data/neural_3d"
SCENE_LIST="cook_spinach " 
#  flame_salmon_1 flame_steak sear_steak
# SCENE_LIST="coffee_martini cook_spinach"

RESULT_DIR="results/stg_neu3d"
# ----------------- Training Setting-------------- #

# ----------------- Main Job --------------------- #
run_single_scene() {
    local GPU_ID=$1
    local SCENE=$2

    echo "Running $SCENE"

    CUDA_VISIBLE_DEVICES=$GPU_ID python simple_trainer_STG.py default \
        --model_path $RESULT_DIR/$SCENE/ \
        --data_dir $SCENE_DIR/$SCENE/colmap_0 \
        --result_dir $RESULT_DIR/$SCENE/

}

# ----------------- Main Job --------------------- #

# ----------------- Experiment Loop -------------- #
GPU_LIST=(3 4 5 6 7)
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
