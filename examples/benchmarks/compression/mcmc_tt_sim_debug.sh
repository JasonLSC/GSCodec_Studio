SCENE_DIR="data/tandt"
# eval all 2 scenes for benchmarking
SCENE_LIST="train " #  truck

# # 0.36M GSs
# RESULT_DIR="results/benchmark_tt_mcmc_0_36M_png_compression"
# CAP_MAX=360000

# # 0.49M GSs
# RESULT_DIR="results/benchmark_tt_mcmc_tt_0_49M_png_compression"
# CAP_MAX=490000

# 1M GSs
RESULT_DIR="results/mcmc_compression_entropy_debug"
CAP_MAX=1000000
# examples/results/mcmc_compression_entropy_debug

# # 4M GSs
# RESULT_DIR="results/benchmark_tt_mcmc_4M_png_compression"
# CAP_MAX=4000000

for SCENE in $SCENE_LIST;
do
    {
        echo "Running $SCENE"

        # train without eval
        # CUDA_VISIBLE_DEVICES=3 python simple_trainer.py mcmc --eval_steps -1 --disable_viewer --data_factor 1 \
        #     --strategy.cap-max $CAP_MAX \
        #     --data_dir $SCENE_DIR/$SCENE/ \
        #     --result_dir $RESULT_DIR/$SCENE/ \
        #     --compression_sim \
        #     --entropy_model_opt \
        #     --shN_ada_mask_opt \
        #     --entropy_model_type "gaussian_model"
            # --compression png


        # eval: use vgg for lpips to align with other benchmarks
        CUDA_VISIBLE_DEVICES=3 python simple_trainer.py mcmc --disable_viewer --data_factor 1 \
            --strategy.cap-max $CAP_MAX \
            --data_dir $SCENE_DIR/$SCENE/ \
            --result_dir $RESULT_DIR/$SCENE/ \
            --lpips_net vgg \
            --compression png \
            --ckpt $RESULT_DIR/$SCENE/ckpts/ckpt_29999_rank0.pt
    } 
done

# Wait for finishing the jobs across all scenes 
wait

# Zip the compressed files and summarize the stats
if command -v zip &> /dev/null
then
    echo "Zipping results"
    python benchmarks/compression/summarize_stats.py --results_dir $RESULT_DIR --scenes $SCENE_LIST
else
    echo "zip command not found, skipping zipping"
fi