python mpeg_gsc_utils/multiview_video_preprocess/video_preprocess.py \
    --scene Cinema 

python mpeg_gsc_utils/multiview_video_preprocess/gen_poses_bds_file.py \
    --scene Cinema 

python mpeg_gsc_utils/multiview_video_preprocess/run_per_frame_colmap.py \
    --scene Cinema 