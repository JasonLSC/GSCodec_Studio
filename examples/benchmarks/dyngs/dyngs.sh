SCENE_DIR="data/GSC"
SCENE_LIST="Bartender" # CBA Bartender Cinema

declare -A TEST_VIEWS
TEST_VIEWS=(
    ["CBA"]="7 22"
    ["Bartender"]="9 11"
    ["Cinema"]="9 11"
)

declare -A START_FRAMES
START_FRAMES=(
    ["CBA"]=0
    ["Bartender"]=50
    ["Cinema"]=235
)

RESULT_DIR="results/dyngs"

NUM_FRAME=65

run_single_scene() {
    local GPU_ID=$1
    local SCENE=$2
    local TEST_VIEW_IDS=${TEST_VIEWS[$SCENE]}
    local START_FRAME=${START_FRAMES[$SCENE]}

    echo "Running $SCENE START_FRAME @ ${START_FRAME}"

    # execute training 
    CUDA_VISIBLE_DEVICES=$GPU_ID python simple_trainer_dyngs.py compression_sim \
        --model_path $RESULT_DIR/$SCENE/ \
        --data_dir $SCENE_DIR/$SCENE/colmap/colmap_${START_FRAME} \
        --result_dir $RESULT_DIR/$SCENE/ \
        --downscale_factor 1 \
        --duration $NUM_FRAME \
        --batch_size 2 \
        --max_steps 50_000 \
        --refine_start_iter 1_500 \
        --refine_stop_iter 45_000 \
        --refine_every 100 \
        --reset_every 5_000 \
        --pause_refine_after_reset 2_000 \
        --strategy Modified_STG_Strategy \
        --test_view_id $TEST_VIEW_IDS 
    
    # execute evaluation and compression 
    CUDA_VISIBLE_DEVICES=$GPU_ID python simple_trainer_dyngs.py default \
        --model_path $RESULT_DIR/$SCENE/ \
        --data_dir $SCENE_DIR/$SCENE/colmap/colmap_${START_FRAME} \
        --result_dir $RESULT_DIR/$SCENE/ \
        --downscale_factor 1 \
        --duration $NUM_FRAME \
        --lpips_net vgg \
        --compression stg \
        --ckpt $RESULT_DIR/$SCENE/ckpts/ckpt_best_rank0.pt \
        --test_view_id $TEST_VIEW_IDS 
}

GPU_LIST=(0)
GPU_COUNT=${#GPU_LIST[@]}

SCENE_IDX=-1

for SCENE in $SCENE_LIST;
do
    SCENE_IDX=$((SCENE_IDX + 1))
    {
        run_single_scene ${GPU_LIST[$SCENE_IDX]} $SCENE
    } #&

done