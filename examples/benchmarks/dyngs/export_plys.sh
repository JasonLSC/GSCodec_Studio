SCENE_DIR="data/GSC"
SCENE_LIST="Bartender" # CBA Bartender

declare -A TEST_VIEWS
TEST_VIEWS=(
    ["CBA"]="7 22"
    ["Bartender"]="8 10 12"
)

RESULT_DIR="results/dyngs"

NUM_FRAME=65

run_single_scene() {
    local GPU_ID=$1
    local SCENE=$2
    local TEST_VIEW_IDS=${TEST_VIEWS[$SCENE]}

    echo "Running $SCENE"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python simple_trainer_dyngs.py default \
        --model_path $RESULT_DIR/$SCENE/ \
        --data_dir $SCENE_DIR/$SCENE/colmap/colmap_50 \
        --result_dir $RESULT_DIR/$SCENE/ \
        --downscale_factor 1 \
        --duration $NUM_FRAME \
        --lpips_net vgg \
        --compression stg \
        --ckpt $RESULT_DIR/$SCENE/ckpts/ckpt_best_rank0.pt \
        --test_view_id $TEST_VIEW_IDS \
        --enable_dyn_splats_export
}

GPU_LIST=(7)
GPU_COUNT=${#GPU_LIST[@]}

SCENE_IDX=-1

for SCENE in $SCENE_LIST;
do
    SCENE_IDX=$((SCENE_IDX + 1))
    {
        run_single_scene ${GPU_LIST[$SCENE_IDX]} $SCENE
    } #&

done