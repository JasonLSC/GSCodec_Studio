SCENE_DIR=data/GSC
SCENE_LIST="Bartender"

BASE_DIR="data/GSC/Splats/zju_internal/gsc_benchmark_data" # /bartender_colmap/colmap_50


PLY_TYPE_LIST="basic_gsplat gscodec_studio"

RESULT_DIR=results/gsc_compression_benchmark

for PLY_TYPE in $PLY_TYPE_LIST;
do
    # echo "Running evaluation and compression on $PLY_TYPE"
    CUDA_VISIBLE_DEVICES=0 python ply_loader_renderer.py \
        png_compression \
        --disable_viewer --data_factor 1 \
        --scene_type GSC \
        --test_view_id 8 10 12 \
        --data_dir $BASE_DIR/bartender_colmap/colmap_50 \
        --result_dir $RESULT_DIR/$PLY_TYPE \
        --lpips_net vgg \
        --ply_path $BASE_DIR/$PLY_TYPE/splats.ply
    
    echo "================"
    echo "R-D Results"
    zip -q -r $RESULT_DIR/$PLY_TYPE/compression.zip $RESULT_DIR/$PLY_TYPE/compression/
    du -b $RESULT_DIR/$PLY_TYPE/compression.zip | awk '{printf "%.2f MB\n", $1/1024/1024}'
    echo
    cat $RESULT_DIR/$PLY_TYPE/stats/val_step-001.json
    echo
    cat $RESULT_DIR/$PLY_TYPE/stats/compress_step-001.json    

done
# Zip the compressed files and summarize the stats
# if command -v zip &> /dev/null
# then
#     echo "Zipping results"
#     python benchmarks/compression/summarize_stats.py --results_dir $RESULT_DIR --scenes $SCENE_LIST
# else
#     echo "zip command not found, skipping zipping"
# fi