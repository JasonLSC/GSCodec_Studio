import glob
import os
from pathlib import Path
import subprocess
from tqdm import tqdm

from scene_info import DATASET_INFOS

def convert_yuv_to_mp4(yuv_file, mp4_file, resolution):
    cmd = [
        '/usr/bin/ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-f', 'rawvideo', '-pixel_format', 'yuv420p10le', '-s', resolution,
        '-colorspace', 'bt709',
        '-color_range', 'pc',
        '-r', '30',
        '-i', yuv_file,
        '-vf', 'scale=in_range=pc:in_color_matrix=bt709:out_range=pc',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-qp', '0',
        '-color_range', 'pc',
        '-colorspace', 'bt709',
        '-color_primaries', 'bt709', 
        '-color_trc', 'bt709',
        '-sws_flags', 'lanczos+bitexact+full_chroma_int+full_chroma_inp',
        '-r', '30',
        mp4_file
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully converted {yuv_file} to {mp4_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")

def convert_yuv_to_png_sequence(input_yuv, output_path, resolution):
    os.makedirs(output_path, exist_ok=True)
    
    base_name = os.path.basename(input_yuv).split('_')[0]
    
    output_pattern = os.path.join(output_path, f'{base_name}_frame%03d.png')
    
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-f', 'rawvideo', '-pixel_format', 'yuv420p10le', '-s', resolution,
        '-colorspace', 'bt709',
        '-color_range', 'pc',
        '-i', input_yuv,
        '-vf', 'scale=in_range=pc:in_color_matrix=bt709:out_range=pc',
        '-pix_fmt', 'rgb24',
        '-color_range', 'pc',
        '-sws_flags', 'lanczos+bitexact+full_chroma_int+full_chroma_inp',
        '-compression_level', '0',
        '-pred', 'none',
        output_pattern
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully converted {input_yuv} to PNG sequence in {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")


if __name__ == "__main__":
    SCENE = "Bartender"
    BASE_DIR = f"/work/Users/lisicheng/Dataset/GSC/{SCENE}"

    yuv_file_list = sorted(glob.glob(BASE_DIR+"/yuv/*.yuv"))

    # YUV to mp4
    for yuv_file in tqdm(yuv_file_list):
        yuv_file = Path(yuv_file)
        mp4_file = yuv_file.parents[1] / "mp4" / yuv_file.with_suffix('.mp4').name
        
        convert_yuv_to_mp4(yuv_file, mp4_file, DATASET_INFOS[SCENE]['resolution'])

    # YUV to PNG
    png_dirpath = BASE_DIR+"/png"

    for yuv_file in tqdm(yuv_file_list):
        yuv_file = Path(yuv_file)
        # mp4_file = yuv_file.parents[1] / "mp4" / yuv_file.with_suffix('.mp4').name
        # print(mp4_file)
        convert_yuv_to_png_sequence(yuv_file, png_dirpath, DATASET_INFOS[SCENE]['resolution'])