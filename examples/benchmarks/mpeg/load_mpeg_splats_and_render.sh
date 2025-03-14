SCENE_LIST="Bartender"
SCENE_PATH=data/GSC/Splats/Bartender_250303/frame000

RESULT_DIR=results/mpeg_splats

PLY_FILE=/work/Users/lisicheng/Dataset/GSC/Splats/Bartender_250303/0000.ply

for SCENE in $SCENE_LIST;
do
    CUDA_VISIBLE_DEVICES=0 python ply_loader_renderer.py mcmc --disable_viewer --data_factor 1 \
        --data_dir $SCENE_PATH \
        --result_dir $RESULT_DIR/$SCENE/ \
        --lpips_net vgg \
        --compression png \
        --ply_path $PLY_FILE
done

# Zip the compressed files and summarize the stats
if command -v zip &> /dev/null
then
    echo "Zipping results"
    python benchmarks/compression/summarize_stats.py --results_dir $RESULT_DIR --scenes $SCENE_LIST
else
    echo "zip command not found, skipping zipping"
fi