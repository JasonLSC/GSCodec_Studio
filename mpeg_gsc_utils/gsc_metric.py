import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple

def run_QMIV_metric(render_YUV_filename: Path, ref_YUV_filename: Path, 
                    render_start_frame: int, ref_start_frame: int,
                    saved_log_file: Optional[Path] = None, 
                    resolution: str = "2048x1088", pix_fmt: str = "yuv420p10le"):
    """
    Run QMIV quality metrics on YUV files in both YUV and RGB domains.
    
    This function executes QMIV tool twice:
    1. First in YUV domain to calculate YUV-based metrics
    2. Then in RGB domain with BT.709 color space conversion
    
    Args:
        render_YUV_filename (Path): Path to the rendered YUV file to be evaluated
        ref_YUV_filename (Path): Path to the reference YUV file
        render_start_frame (int): Index of the start frame of rendered video, usually as 0.
        ref_start_frame (int): Index of the start frame of reference video, not always as 0.
        saved_log_file (Optional[Path], optional): Path where the QMIV log will be saved. 
            If None, uses render file's stem + ".txt". Defaults to None.
        resolution (str, optional): Resolution of input videos in format "widthxheight". 
            Defaults to "2048x1088".
        pix_fmt (str, optional): Pixel format of input videos. 
            Defaults to "yuv420p10le".
    
    Returns:
        dict: Dictionary containing the following metrics:
            - "RGB_PSNR": PSNR value in RGB domain
            - "YUV_PSNR": PSNR value in YUV domain
            - "YUV_SSIM": SSIM value in YUV domain
            - "YUV_IVSSIM": IVSSIM value
    
    Note:
        - Requires QMIV executable in the current directory
        - Processes 65 frames as specified in the -nf parameter
        - Uses BT.709 color space for RGB conversion
        - Automatically overwrites existing log file if it exists
    """
        
    if saved_log_file is None:
        saved_log_file = render_YUV_filename.stem + ".txt"

    if os.path.exists(saved_log_file):
        os.remove(saved_log_file)
    
    # YUV domain
    QMIV_cmd = [
        "./QMIV",
        "-i0", render_YUV_filename,
        "-i1", ref_YUV_filename,
        "-s0", f"{render_start_frame}",
        "-s1", f"{ref_start_frame}",
        "-ps", resolution,
        "-pf", pix_fmt,
        "-nf", "65",
        "-r", saved_log_file,
        "-ml", "All"
    ]

    subprocess.run(QMIV_cmd, capture_output=True, text=True)
    # TODO:specify start frame of ref video
    # RGB domain
    QMIV_cmd = [
        "./QMIV",
        "-i0", render_YUV_filename,
        "-i1", ref_YUV_filename,
        "-s0", f"{render_start_frame}",
        "-s1", f"{ref_start_frame}",
        "-ps", resolution,
        "-pf", pix_fmt,
        "-nf", "65",
        "-csi", "YCbCr_BT709", "-csm", "RGB", "-cwa", "1:1:1:0", "-cws", "1:1:1:0",
        "-r", saved_log_file,
        "-ml", "All"
    ]

    subprocess.run(QMIV_cmd, capture_output=True, text=True)

    with open(saved_log_file, 'r') as f:
        content = f.read()

    yuv_psnr = float(re.search(r'PSNR\s+-YCbCr\s+(\d+\.\d+)', content).group(1))
    yuv_ssim = float(re.search(r'SSIM\s+-YCbCr\s+(\d+\.\d+)', content).group(1))
    yuv_ivssim = float(re.search(r'IVSSIM\s+(\d+\.\d+)', content).group(1))
    rgb_psnr = float(re.search(r'PSNR\s+-RGB\s+(\d+\.\d+)', content).group(1))

    ret_dict = {
        "RGB_PSNR": rgb_psnr,
        "YUV_PSNR": yuv_psnr,
        "YUV_SSIM": yuv_ssim,
        "YUV_IVSSIM": yuv_ivssim,
    }

    return ret_dict

def convert_mp4_to_yuv(input_mp4: Path) -> Tuple[bool, Path | None]:
    """
    Convert MP4 to YUV using ffmpeg
    
    Args:
        input_mp4: Input MP4 filename
        
    Returns:
        bool: True if conversion successful, False otherwise
    """
    # Check if input file exists
    if not os.path.exists(input_mp4):
        print(f"Error: Input file {input_mp4} does not exist")
        return False, None

    # Generate output filename by replacing extension
    output_yuv = input_mp4.with_suffix('.yuv')

    # Check if output file exists
    if os.path.exists(output_yuv):
        os.remove(output_yuv)

    # Construct ffmpeg command
    cmd = [
        'ffmpeg', '-i', str(input_mp4),
        '-vf', 'scale=in_range=pc:in_color_matrix=bt709:out_range=pc:out_color_matrix=bt709',
        '-pix_fmt', 'yuv420p10le',
        '-colorspace', 'bt709',
        '-color_primaries', 'bt709', 
        '-color_trc', 'bt709',
        '-color_range', 'pc',
        '-sws_flags', 'lanczos+bitexact+full_chroma_int+full_chroma_inp',
        '-f', 'rawvideo',
        str(output_yuv)
    ]
    
    try:
        # Execute ffmpeg command
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Successfully converted {input_mp4} to {output_yuv}")
        return True, output_yuv
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e.stderr.decode()}")
        return False, None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False, None

DATASET_INFOS = {
    "CBA": {
        "start_frame": 0,
        "resolution": "2048x1088"
    },
    "Bartender": {
        "start_frame": 50,
        "resolution": "1920x1080"
    },
    "Choreo_Dark": {
        "start_frame": 30,
    },
    "Cinema": {
        "start_frame": 235,
    },
}

if __name__ == "__main__":
    SCENE = "CBA"

    success, output_path = convert_mp4_to_yuv(Path('../temp/val_step29999_testv0.mp4'))
    if success:
        results_dict = run_QMIV_metric(output_path,
                            Path("../temp/v07_texture_2048x1088_yuv420p10le.yuv"),
                            render_start_frame=0,
                            ref_start_frame=DATASET_INFOS[SCENE]["start_frame"],
                            resolution=DATASET_INFOS[SCENE]["resolution"]
                        )
    print(results_dict)


