# Preparation for dynamic splats

## Dataset file structure
**MPEG Multiview Video Dataset**
Please organize the data structure as follows:

data/
├── GSC
│   ├── scene1/
│   │   ├── colmap
│   │   ├── mp4
│   │   ├── png
│   │   ├── yuv
│   │   │   ├── v00_texture_1920x1080_yuv420p10le.yuv
│   │   │   ├── v01_texture_1920x1080_yuv420p10le.yuv
│   │   │   ├── ...
│   ├── scene1/
│   │   ├── colmap
│   │   ├── mp4
│   │   ├── png
│   │   ├── yuv
│   │   │   ├── v00_texture_1920x1080_yuv420p10le.yuv
│   │   │   ├── v01_texture_1920x1080_yuv420p10le.yuv
│   │   │   ├── ...

## Operations
1. Convert multiview videos in "yuv" format to "mp4" format and "png" format
```bash
python mpeg_gsc_utils/multiview_video_preprocess/video_preprocess.py
```
2. Obtain the camera intrinsic and extrinsic parameters and save as "poses_bds.npy" file
```bash
python mpeg_gsc_utils/multiview_video_preprocess/gen_poses_bds.py
```
3. Run colmap frame by frame to get per-frame SfM point clouds 
```bash
python mpeg_gsc_utils/multiview_video_preprocess/run_per_frame_colmap.py
```