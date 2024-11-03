SCENE_DIR="data/neural_3d"
# SCENE_LIST="coffee_martini cook_spinach cut_roasted_beef flame_salmon_1 flame_steak sear_steak"
SCENE_LIST="cook_spinach"

RESULT_DIR="results/stg_neu3d_png"


for SCENE in $SCENE_LIST;
do
    echo "Running $SCENE"

    CUDA_VISIBLE_DEVICES=1 python simple_trainer_STG.py default \
        --model_path $RESULT_DIR/$SCENE/ \
        --data_dir $SCENE_DIR/$SCENE/colmap_0 \
        --result_dir $RESULT_DIR/$SCENE/
    
    CUDA_VISIBLE_DEVICES=1 python simple_trainer_STG.py default \
        --model_path $RESULT_DIR/$SCENE/ \
        --data_dir $SCENE_DIR/$SCENE/colmap_0 \
        --result_dir $RESULT_DIR/$SCENE/ \
        --lpips_net vgg \
        --compression png \
        --ckpt $RESULT_DIR/$SCENE/ckpts/ckpt_29999_rank0.pt
done 