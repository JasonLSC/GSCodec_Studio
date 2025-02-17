# Option for pruning
PRUNE_SPLATS=False
echo "PRUNE_SPLATS is set to: $PRUNE_SPLATS"

# Datadir and dataset
SCENE_DIR="data/GSC"
SCENE_LIST="CBA" # CBA Bartender

declare -A TEST_VIEWS
TEST_VIEWS=(
    ["CBA"]="7 22"
    ["Bartender"]="8 10 12"
)

declare -A START_FRAMES
START_FRAMES=(
    ["CBA"]=0
    ["Bartender"]=50
)

RESULT_DIR="results/dyngs"
NUM_FRAME=65

run_single_scene() {
    local GPU_ID=$1
    local SCENE=$2
    local TEST_VIEW_IDS=${TEST_VIEWS[$SCENE]}
    local START_FRAME=${START_FRAMES[$SCENE]}

    echo "Running $SCENE"
    
    CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python simple_trainer_dyngs.py default \
        --model_path $RESULT_DIR/$SCENE/ \
        --data_dir $SCENE_DIR/$SCENE/colmap/colmap_${START_FRAME} \
        --result_dir $RESULT_DIR/$SCENE/ \
        --downscale_factor 1 \
        --duration $NUM_FRAME \
        --lpips_net vgg \
        --ckpt $RESULT_DIR/$SCENE/ckpts/ckpt_best_rank0.pt \
        --test_view_id $TEST_VIEW_IDS \
        --enable_dyn_splats_export"
    
    if [ "$PRUNE_SPLATS" = "True" ]; then
        CMD="$CMD --temp_opa_vis_pruning"
        echo "Pruning splat is $PRUNE_SPLATS"
    fi

    eval "$CMD"
}

GPU_LIST=(7)
GPU_COUNT=${#GPU_LIST[@]}

SCENE_IDX=-1

for SCENE in $SCENE_LIST;
do
    SCENE_IDX=$((SCENE_IDX + 1))
    {   
        # export plys
        run_single_scene ${GPU_LIST[$SCENE_IDX]} $SCENE
    } #&

    # pack up all splats in ply dir
    if [ "$PRUNE_SPLATS" = "True" ]; then
        echo "Zip pruned splats to $RESULT_DIR/$SCENE/${SCENE}_pruned_splats.zip"
        zip -r $RESULT_DIR/$SCENE/${SCENE}_pruned_splats.zip $RESULT_DIR/$SCENE/plys
    else 
        echo "Zip splats to $RESULT_DIR/$SCENE/${SCENE}_splats.zip"
        zip -r $RESULT_DIR/$SCENE/${SCENE}_splats.zip $RESULT_DIR/$SCENE/plys
    fi

done

