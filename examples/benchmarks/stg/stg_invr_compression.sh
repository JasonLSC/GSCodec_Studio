SCENE_DIR="data/INVR"
SCENE_LIST="CBA" # Bartender
#  

RESULT_DIR="results/stg_invr_compression"

NUM_FRAME=65

run_single_scene() {
    local GPU_ID=$1
    local SCENE=$2

    echo "Running $SCENE"

    CUDA_VISIBLE_DEVICES=$GPU_ID python simple_trainer_STG.py compression_sim \
        --model_path $RESULT_DIR/$SCENE/ \
        --data_dir $SCENE_DIR/$SCENE/colmap_0 \
        --result_dir $RESULT_DIR/$SCENE/ \
        --duration $NUM_FRAME \
        --entropy_model_opt \
        # --rd_lambda 0.02
        # --profiler_enabled
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python simple_trainer_STG.py default \
        --model_path $RESULT_DIR/$SCENE/ \
        --data_dir $SCENE_DIR/$SCENE/colmap_0 \
        --result_dir $RESULT_DIR/$SCENE/ \
        --duration $NUM_FRAME \
        --lpips_net vgg \
        --compression stg \
        --ckpt $RESULT_DIR/$SCENE/ckpts/ckpt_best_rank0.pt
}

GPU_LIST=(0 1 2 3 4 5)
GPU_COUNT=${#GPU_LIST[@]}

SCENE_IDX=-1

for SCENE in $SCENE_LIST;
do
    SCENE_IDX=$((SCENE_IDX + 1))
    {
        run_single_scene ${GPU_LIST[$SCENE_IDX]} $SCENE
    } #&

done

wait

# Zip the compressed files and summarize the stats
if command -v zip &> /dev/null
then
    echo "Zipping results"
    python benchmarks/stg/summarize_stats.py --results_dir $RESULT_DIR --scenes $SCENE_LIST --num_frame $NUM_FRAME
else
    echo "zip command not found, skipping zipping"
fi