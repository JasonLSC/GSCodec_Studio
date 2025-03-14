SCENE_DIR=data/GSC
SCENE_LIST="Bartender"

BASE_DIR="data/GSC/Splats/zju_internal/gsc_benchmark_data" # /bartender_colmap/colmap_50

PLY_TYPE_LIST="basic_gsplat gscodec_studio"

RP_LIST="0 1 2 3"

RESULT_DIR=results/gsc_compression_benchmark

for PLY_TYPE in $PLY_TYPE_LIST;
do
    for RP in $RP_LIST;
    do 
        SAVE_DIR="$RESULT_DIR/$PLY_TYPE/rp$RP"
        echo "Running evaluation and compression on $PLY_TYPE and RP:$RP"

        CUDA_VISIBLE_DEVICES=0 python ply_loader_renderer.py \
            x265_compression_rp$RP \
            --disable_viewer --data_factor 1 \
            --scene_type GSC \
            --test_view_id 8 10 12 \
            --data_dir $BASE_DIR/bartender_colmap/colmap_50 \
            --result_dir $SAVE_DIR \
            --lpips_net vgg \
            --ply_path $BASE_DIR/$PLY_TYPE/splats.ply   
    done

    python benchmarks/mpeg/zip_and_summarize_stats.py --results_dir $RESULT_DIR/$PLY_TYPE --rps rp0 rp1 rp2 rp3

done
