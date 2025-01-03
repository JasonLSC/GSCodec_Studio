# ------------------------- Settings ------------------------- #
SCENE_DIR="data/360_v2"
# eval all 9 scenes for benchmarking
SCENE_LIST="bicycle bonsai counter flowers garden kitchen room stump treehill"
# SCENE_LIST="flowers treehill"

# # 0.36M GSs
# RESULT_DIR="results/benchmark_mcmc_0_36M_png_compression"
# CAP_MAX=360000

# # 0.49M GSs
# RESULT_DIR="results/benchmark_mcmc_0_49M_png_compression"
# CAP_MAX=490000

# 1M GSs
RESULT_DIR="results/benchmark_mcmc_1M_png_compression"
CAP_MAX=1000000

# # 4M GSs
# RESULT_DIR="results/benchmark_mcmc_4M_png_compression"
# CAP_MAX=4000000

GPU_LIST="1 2 3 4"
# ------------------------- Settings ------------------------- #

# ------------------------- Experiment ----------------------- #

run_scene() {
    SCENE=$1
    GPU_ID=$2

    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi

    echo "Running $SCENE on GPU $GPU_ID"

    # train without eval
    CUDA_VISIBLE_DEVICES=$GPU_ID python simple_trainer.py mcmc --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR \
        --strategy.cap-max $CAP_MAX \
        --data_dir data/360_v2/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

    # eval
    CUDA_VISIBLE_DEVICES=$GPU_ID python simple_trainer.py mcmc --disable_viewer --data_factor $DATA_FACTOR \
        --strategy.cap-max $CAP_MAX \
        --data_dir data/360_v2/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/ \
        --lpips_net vgg \
        --compression png \
        --ckpt $RESULT_DIR/$SCENE/ckpts/ckpt_29999_rank0.pt
}

# ------------------------- Experiment ----------------------- #

# # ------------------------- Main ----------------------------- #

# # 将场景列表和GPU列表转换为数组
# IFS=' ' read -r -a SCENE_ARRAY <<< "$SCENE_LIST"
# IFS=' ' read -r -a GPU_ARRAY <<< "$GPU_LIST"
# NUM_GPUS=${#GPU_ARRAY[@]}

# # 追踪已经处理的场景索引
# current_scene_index=0
# total_scenes=${#SCENE_ARRAY[@]}

# # 追踪每个GPU上运行的进程和场景
# declare -A gpu_pids
# declare -A gpu_scenes

# # 初始分配场景给GPU
# for ((i=0; i<NUM_GPUS && i<total_scenes; i++)); do
#     GPU_ID=${GPU_ARRAY[$i]}
#     SCENE=${SCENE_ARRAY[$i]}
#     run_scene $SCENE $GPU_ID &
#     gpu_pids[$GPU_ID]=$!
#     gpu_scenes[$GPU_ID]=$SCENE
#     current_scene_index=$((i + 1))
#     echo "Started $SCENE on GPU $GPU_ID"
# done

# # 持续检查并分配新的场景
# while [ $current_scene_index -lt $total_scenes ]; do
#     for GPU_ID in "${GPU_ARRAY[@]}"; do
#         # 检查这个GPU上的进程是否完成
#         if [ -n "${gpu_pids[$GPU_ID]}" ] && ! kill -0 ${gpu_pids[$GPU_ID]} 2>/dev/null; then
#             echo "Finished ${gpu_scenes[$GPU_ID]} on GPU $GPU_ID"
            
#             # 分配新的场景给这个GPU
#             SCENE=${SCENE_ARRAY[$current_scene_index]}
#             run_scene $SCENE $GPU_ID &
#             gpu_pids[$GPU_ID]=$!
#             gpu_scenes[$GPU_ID]=$SCENE
#             echo "Started $SCENE on GPU $GPU_ID"
            
#             current_scene_index=$((current_scene_index + 1))
            
#             # 如果所有场景都已分配，退出循环
#             if [ $current_scene_index -eq $total_scenes ]; then
#                 break 2
#             fi
#         fi
#     done
#     sleep 10  # 检查间隔
# done

# # 等待所有进程完成
# wait

# echo "All scenes completed!"
# ------------------------- Main ----------------------------- #

# ------------------------- Compression ---------------------- #

Zip the compressed files and summarize the stats
if command -v zip &> /dev/null
then
    echo "Zipping results"
    python benchmarks/compression/summarize_stats.py --results_dir $RESULT_DIR --scenes $SCENE_LIST
else
    echo "zip command not found, skipping zipping"
fi
# ------------------------- Compression ---------------------- #