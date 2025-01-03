SCENE_DIR="data/INVR_colortune"
SCENE_LIST="CBA" # Bartender
#  

RESULT_DIR="results/stg_invr_colortune_full_res_65frame_noprune"

NUM_FRAME=65

run_single_scene() {
    local GPU_ID=$1
    local SCENE=$2

    echo "Running $SCENE"

    CUDA_VISIBLE_DEVICES=$GPU_ID python simple_trainer_STG.py compression_sim \
        --model_path $RESULT_DIR/$SCENE/ \
        --data_dir $SCENE_DIR/$SCENE/colmap_0 \
        --result_dir $RESULT_DIR/$SCENE/ \
        --downscale_factor 1 \
        --duration $NUM_FRAME \
        --batch_size 2 \
        --max_steps 60_000 \
        --refine_start_iter 3_000 \
        --refine_stop_iter 30_000 \
        --refine_every 100 \
        --reset_every 6_000 \
        --pause_refine_after_reset 500 \
        --strategy Modified_STG_Strategy \
        --test_view_id 7 22 \
        # --save_steps 10_000 20_000 30_000 40_000 50_000 60_000 \
        # --eval_steps 10_000 20_000 30_000 40_000 50_000 60_000 \
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python simple_trainer_STG.py default \
        --model_path $RESULT_DIR/$SCENE/ \
        --data_dir $SCENE_DIR/$SCENE/colmap_0 \
        --result_dir $RESULT_DIR/$SCENE/ \
        --downscale_factor 1 \
        --duration $NUM_FRAME \
        --lpips_net vgg \
        --compression stg \
        --ckpt $RESULT_DIR/$SCENE/ckpts/ckpt_best_rank0.pt \
        --test_view_id 7 22
}

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

wait

# Zip the compressed files and summarize the stats
if command -v zip &> /dev/null
then
    echo "Zipping results"
    python benchmarks/stg/summarize_stats.py --results_dir $RESULT_DIR --scenes $SCENE_LIST --num_frame $NUM_FRAME
else
    echo "zip command not found, skipping zipping"
fi