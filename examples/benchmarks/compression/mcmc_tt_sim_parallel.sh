# ----------------- Training Setting-------------- #
SCENE_DIR="data/tandt"
# eval all 9 scenes for benchmarking
SCENE_LIST="train truck" #  truck
# SCENE_LIST="garden bicycle stump bonsai counter kitchen room treehill flowers"

# # 0.36M GSs
# RESULT_DIR="results/benchmark_tt_mcmc_0_36M_png_compression"
# CAP_MAX=360000

# # 0.49M GSs
# RESULT_DIR="results/benchmark_tt_mcmc_tt_0_49M_png_compression"
# CAP_MAX=490000

# 1M GSs
RESULT_DIR="results/benchmark_tt_mcmc_1M_png_compression_pipe_wo_adamask"
CAP_MAX=1000000

# # 4M GSs
# RESULT_DIR="results/benchmark_tt_mcmc_4M_png_compression"
# CAP_MAX=4000000

# ----------------- Training Setting-------------- #



# ----------------- Main Job --------------------- #
run_single_scene() {
    local GPU_ID=$1
    local SCENE=$2

    echo "Running $SCENE on GPU: $GPU_ID"

    # train without eval
    # CUDA_VISIBLE_DEVICES=$GPU_ID python simple_trainer.py mcmc --eval_steps -1 --disable_viewer --data_factor 1 \
    #     --strategy.cap-max $CAP_MAX \
    #     --data_dir $SCENE_DIR/$SCENE/ \
    #     --result_dir $RESULT_DIR/$SCENE/ \
    #     --compression_sim \
    #     --entropy_model_opt \
    #     # --shN_ada_mask_opt
    #     # --compression png


    # eval: use vgg for lpips to align with other benchmarks
    CUDA_VISIBLE_DEVICES=$GPU_ID python simple_trainer.py mcmc --disable_viewer --data_factor 1 \
        --strategy.cap-max $CAP_MAX \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/ \
        --lpips_net vgg \
        --compression png \
        --ckpt $RESULT_DIR/$SCENE/ckpts/ckpt_29999_rank0.pt
    
}
# ----------------- Main Job --------------------- #



# ----------------- Experiment Loop -------------- #
GPU_LIST=(4 5)
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

# Zip the compressed files and summarize the stats
if command -v zip &> /dev/null
then
    echo "Zipping results"
    python benchmarks/compression/summarize_stats.py --results_dir $RESULT_DIR --scenes $SCENE_LIST
else
    echo "zip command not found, skipping zipping"
fi