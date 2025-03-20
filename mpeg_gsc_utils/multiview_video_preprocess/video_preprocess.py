import glob
import os
from pathlib import Path
import subprocess
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional

import tyro

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


@dataclass
class ConversionConfig:
    """Configuration for YUV conversion"""
    scene: str
    """Scene name (e.g., Bartender)"""
    
    base_dir: Optional[str] = None
    """Base directory path. If not provided, defaults to examples/data/GSC/{scene}"""
    
    only_mp4: bool = False
    """Only convert YUV to MP4"""
    
    only_png: bool = False
    """Only convert YUV to PNG"""

def main(config: ConversionConfig):
    # Process parameters
    SCENE = config.scene
    BASE_DIR = config.base_dir if config.base_dir else f"examples/data/GSC/{SCENE}"
    
    # Get resolution from DATASET_INFOS
    resolution = DATASET_INFOS[SCENE]['resolution']
    
    yuv_file_list = sorted(glob.glob(BASE_DIR+"/yuv/*.yuv"))
    
    # By default, perform both conversions unless specified otherwise
    do_mp4 = not config.only_png
    do_png = not config.only_mp4
    
    # If both only_mp4 and only_png are specified, perform both conversions
    if config.only_mp4 and config.only_png:
        do_mp4 = do_png = True
    
    # YUV to mp4
    if do_mp4:
        print("Converting YUV to MP4...")
        for yuv_file in tqdm(yuv_file_list):
            yuv_file = Path(yuv_file)
            mp4_file = Path(BASE_DIR) / "mp4" / yuv_file.with_suffix('.mp4').name
            # Ensure target directory exists
            mp4_file.parent.mkdir(parents=True, exist_ok=True)
            convert_yuv_to_mp4(yuv_file, mp4_file, resolution)
    
    # YUV to PNG
    if do_png:
        print("Converting YUV to PNG sequences...")
        png_dirpath = BASE_DIR+"/png"
        # Ensure PNG directory exists
        Path(png_dirpath).mkdir(parents=True, exist_ok=True)
        
        for yuv_file in tqdm(yuv_file_list):
            yuv_file = Path(yuv_file)
            convert_yuv_to_png_sequence(yuv_file, png_dirpath, resolution)

    print("Conversion completed!")

if __name__ == "__main__":
    config = tyro.cli(ConversionConfig)
    main(config)