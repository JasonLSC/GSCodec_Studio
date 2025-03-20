# Preparation for dynamic splats

## Dataset file structure
**MPEG Multiview Video Dataset**
Please organize the data structure as follows:

```
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
│   ├── scene2/
│   │   ├── colmap
│   │   ├── mp4
│   │   ├── png
│   │   ├── yuv
│   │   │   ├── v00_texture_1920x1080_yuv420p10le.yuv
│   │   │   ├── v01_texture_1920x1080_yuv420p10le.yuv
│   │   │   ├── ...
```

## Preliminaries
The following series of preprocessing operations for MPEG datasets requires **Colmap** and **ffmpeg** to be installed in advance.

## Operations
1. Before you start, please open ``mpeg_gsc_utils/multiview_video_preprocess/scene_info.py`` and fill in some necessary metadata for the Scene you are going to convert.
For now, we have already prepared the metadata for Bartender and Cinema.

1. Convert multiview videos in "yuv" format to "mp4" format and "png" format
```bash
python mpeg_gsc_utils/multiview_video_preprocess/video_preprocess.py \
    --scene Cinema 
``` 

For the script mentioned above, you must specify the scene name. Additionally, if you haven't placed your data under ``examples/data/GSC/{scene}``, you can manually specify your base_dir by passing ``--base_dir``.

2. Obtain the camera intrinsic and extrinsic parameters and save as "poses_bds.npy" file
```bash
python mpeg_gsc_utils/multiview_video_preprocess/gen_poses_bds_file.py \
    --scene Cinema 
```

For the script mentioned above, you must specify the scene name. Additionally, if you haven't placed your data under ``examples/data/GSC/{scene}``, you can manually specify your base_dir by passing ``--base_dir``.You can also select a specific number of frames by passing ``--frame_num``.

3. Run colmap frame by frame to get per-frame SfM point clouds 
```bash
python mpeg_gsc_utils/multiview_video_preprocess/run_per_frame_colmap.py \
    --scene Cinema 
```

For the script mentioned above, you must specify the scene name. Additionally, if you haven't placed your data under ``examples/data/GSC/{scene}``, you can manually specify your base_dir by passing ``--base_dir``.You can also select a specific number of frames by passing ``--frame_num``.