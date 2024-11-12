# ----------------- Training Setting-------------- #
SCENE_DIR="data/neural_3d"
SCENE_LIST="coffee_martini cook_spinach cut_roasted_beef flame_salmon_1 flame_steak sear_steak" # coffee_martini cook_spinach cut_roasted_beef 

RESULT_DIR="results/stg_neu3d_GOP50"

NUM_FRAME=50

START_FRAME_LIST="150 200 250" # 0 50 100 150 200 250

# ----------------- Training Setting-------------- #

# ----------------- Main Job --------------------- #
run_single_GOP() {
    local GPU_ID=$1
    local SCENE=$2
    local START_FRAME=$3

    echo "Running $SCENE"

    CUDA_VISIBLE_DEVICES=$GPU_ID python simple_trainer_STG.py compression_sim \
        --model_path $RESULT_DIR/$SCENE/ \
        --data_dir $SCENE_DIR/$SCENE/colmap_$START_FRAME \
        --result_dir $RESULT_DIR/$SCENE/$START_FRAME/ \
        --duration $NUM_FRAME \
        --entropy_model_opt \
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python simple_trainer_STG.py default \
        --model_path $RESULT_DIR/$SCENE/ \
        --data_dir $SCENE_DIR/$SCENE/colmap_$START_FRAME \
        --result_dir $RESULT_DIR/$SCENE/$START_FRAME/ \
        --duration $NUM_FRAME \
        --lpips_net vgg \
        --compression stg \
        --ckpt $RESULT_DIR/$SCENE/ckpts/ckpt_best_rank0.pt

}

# ----------------- Main Job --------------------- #

# ----------------- Experiment Loop -------------- #
GPU_LIST=(0 1 2 4 6 7)
GPU_COUNT=${#GPU_LIST[@]}

SCENE_IDX=-1

for SCENE in $SCENE_LIST;
do
    GPU_LIST=(5 6 7)
    SCENE_IDX=-1

    for START_FRAME in $START_FRAME_LIST;
    do
        SCENE_IDX=$((SCENE_IDX + 1))
        run_single_GOP ${GPU_LIST[$SCENE_IDX]} $SCENE $START_FRAME &
    done 
    wait 
    echo "$SCENE is Done."
done

# run_single_GOP 0 coffee_martini 0

# ----------------- Experiment Loop -------------- #

# Wait for finishing the jobs across all scenes 
wait

echo "All scenes finished."
