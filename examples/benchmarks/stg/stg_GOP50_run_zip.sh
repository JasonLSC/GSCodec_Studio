# ----------------- Training Setting-------------- #
SCENE_DIR="data/neural_3d"
SCENE_LIST="coffee_martini cook_spinach cut_roasted_beef flame_salmon_1 flame_steak sear_steak" 

RESULT_DIR="results/stg_neu3d_GOP50"

NUM_FRAME=50

START_FRAME_LIST="0 50 100 150 200 250" # 0 50 100 150 200 250

if command -v zip &> /dev/null
then
    echo "Zipping results"
    python benchmarks/stg/summarize_GOP50_stats.py --results_dir $RESULT_DIR --scenes $SCENE_LIST --num_frame $NUM_FRAME
else
    echo "zip command not found, skipping zipping"
fi

