# ----------------- Training Setting-------------- #
SCENE_DIR="data/GSC"
SCENE_LIST="Bartender" 

# 0.16M GSs
RESULT_DIR="results/mpeg_basic"
CAP_MAX=160000

RD_LAMBDA=0.01

# ----------------- Training Setting-------------- #

# ----------------- Args ------------------------- #

if [ ! -z "$1" ]; then
    RD_LAMBDA="$1"
    RESULT_DIR="results/Ours_TT_rd_lambda_${RD_LAMBDA}"
fi

# ----------------- Args ------------------------- #

# ----------------- Main Job --------------------- #
run_single_scene() {
    local GPU_ID=$1
    local SCENE=$2

    echo "Running $SCENE on GPU: $GPU_ID"

    # train without eval
    CUDA_VISIBLE_DEVICES=$GPU_ID python simple_trainer.py default --eval_steps -1 --disable_viewer --data_factor 1 \
        --scene_type GSC \
        --test_view_id 8 10 12 \
        --data_dir $SCENE_DIR/$SCENE/colmap/colmap_50 \
        --result_dir $RESULT_DIR/$SCENE/ \
        --lpips_net vgg \
        --compression png


    # eval: use vgg for lpips to align with other benchmarks
    CUDA_VISIBLE_DEVICES=$GPU_ID python simple_trainer.py default --disable_viewer --data_factor 1 \
        --scene_type GSC \
        --test_view_id 8 10 12 \
        --data_dir $SCENE_DIR/$SCENE/colmap/colmap_50 \
        --result_dir $RESULT_DIR/$SCENE/ \
        --lpips_net vgg \
        --ckpt $RESULT_DIR/$SCENE/ckpts/ckpt_29999_rank0.pt \
        --compression png
    
}
# ----------------- Main Job --------------------- #



# ----------------- Experiment Loop -------------- #
GPU_LIST=(1)
GPU_COUNT=${#GPU_LIST[@]}

SCENE_IDX=-1

for SCENE in $SCENE_LIST;
do
    SCENE_IDX=$((SCENE_IDX + 1))
    {
        run_single_scene ${GPU_LIST[$SCENE_IDX]} $SCENE
    } #&

done

# ----------------- Experiment Loop -------------- #

# Wait for finishing the jobs across all scenes 
wait
echo "All scenes finished."

# Zip the compressed files and summarize the stats
if command -v zip &> /dev/null
then
    echo "Zipping results"
    python benchmarks/compression/summarize_stats.py --results_dir $RESULT_DIR --scenes $SCENE_LIST
else
    echo "zip command not found, skipping zipping"
fi