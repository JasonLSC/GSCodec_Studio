# post-processing compression
SCENE_DIR="data/neural_3d"
SCENE_LIST="cook_spinach cut_roasted_beef flame_salmon_1 flame_steak sear_steak"
# coffee_martini

RESULT_DIR="results/stg_neu3d_post_compression"

NUM_FRAME=300

for SCENE in $SCENE_LIST;
do
    echo "Running $SCENE"

    CUDA_VISIBLE_DEVICES=1 python simple_trainer_STG.py default \
        --model_path $RESULT_DIR/$SCENE/ \
        --data_dir $SCENE_DIR/$SCENE/colmap_0 \
        --result_dir $RESULT_DIR/$SCENE/ \
        --duration $NUM_FRAME
    
    CUDA_VISIBLE_DEVICES=1 python simple_trainer_STG.py default \
        --model_path $RESULT_DIR/$SCENE/ \
        --data_dir $SCENE_DIR/$SCENE/colmap_0 \
        --result_dir $RESULT_DIR/$SCENE/ \
        --duration $NUM_FRAME \
        --lpips_net vgg \
        --compression stg \
        --ckpt $RESULT_DIR/$SCENE/ckpts/ckpt_best_rank0.pt
done 

# Zip the compressed files and summarize the stats
# if command -v zip &> /dev/null
# then
#     echo "Zipping results"
#     python benchmarks/stg/summarize_stats.py --results_dir $RESULT_DIR --scenes $SCENE_LIST --num_frame $NUM_FRAME
# else
#     echo "zip command not found, skipping zipping"
# fi