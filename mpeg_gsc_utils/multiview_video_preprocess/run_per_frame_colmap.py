import os
import shutil
import numpy as np
from pathlib import Path
from tqdm import trange
import tyro
from dataclasses import dataclass
from typing import Optional

from scene_info import DATASET_INFOS
from mpeg_gsc_utils.pre_colmap import COLMAPDatabase

def posetow2c_matrcs(poses):
    tmp = inversestep4(inversestep3(inversestep2(inversestep1(poses))))
    N = tmp.shape[0]
    ret = []
    for i in range(N):
        ret.append(tmp[i])
    return ret

def inversestep4(c2w_mats):
    return np.linalg.inv(c2w_mats)
def inversestep3(newposes):
    tmp = newposes.transpose([2, 0, 1]) # 20, 3, 4 
    N, _, __ = tmp.shape
    zeros = np.zeros((N, 1, 4))
    zeros[:, 0, 3] = 1
    c2w_mats = np.concatenate([tmp, zeros], axis=1)
    return c2w_mats

def inversestep2(newposes):
    return newposes[:,0:4, :]
def inversestep1(newposes):
    poses = np.concatenate([newposes[:, 1:2, :], newposes[:, 0:1, :], -newposes[:, 2:3, :],  newposes[:, 3:4, :],  newposes[:, 4:5, :]], axis=1)
    return poses

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def write_colmap(path, cameras, offset=0):
    projectfolder = path / f"colmap_{offset}"
    manualfolder = projectfolder / "manual"
    manualfolder.mkdir(exist_ok=True)

    savetxt = manualfolder / "images.txt"
    savecamera = manualfolder / "cameras.txt"
    savepoints = manualfolder / "points3D.txt"

    imagetxtlist = []
    cameratxtlist = []

    db_file = projectfolder / "input.db"
    if db_file.exists():
        db_file.unlink()

    db = COLMAPDatabase.connect(db_file)

    db.create_tables()


    for cam in cameras:
        id = cam['id']
        filename = cam['filename']

        # intrinsics
        w = cam['w']
        h = cam['h']
        fx = cam['fx']
        fy = cam['fy']
        cx = cam['cx']
        cy = cam['cy']

        # extrinsics
        colmapQ = cam['q']
        T = cam['t']

        # check that cx is almost w /2, idem for cy
        # assert abs(cx - w / 2) / cx < 0.10, f"cx is not close to w/2: {cx}, w: {w}"
        # assert abs(cy - h / 2) / cy < 0.10, f"cy is not close to h/2: {cy}, h: {h}"

        line = f"{id} " + " ".join(map(str, colmapQ)) + " " + " ".join(map(str, T)) + f" {id} {filename}\n"
        imagetxtlist.append(line)
        imagetxtlist.append("\n")

        params = np.array((fx , fy, cx, cy,))

        camera_id = db.add_camera(1, w, h, params)
        cameraline = f"{id} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n"
        cameratxtlist.append(cameraline)
        image_id = db.add_image(filename, camera_id,  prior_q=colmapQ, prior_t=T, image_id=id)
        db.commit()
    db.close()

    savetxt.write_text("".join(imagetxtlist))
    savecamera.write_text("".join(cameratxtlist))
    savepoints.write_text("")  # Creating an empty points3D.txt file

def convertdynerftocolmapdb(path, offset=0, downscale=1):
    originnumpy = path / "poses_bounds.npy"
    # video_paths = sorted(path.glob('cam*.mp4'))
    # if not video_paths:
    #     video_paths = sorted(path.glob('v*.mp4'))
    colmap_images_dir = path / f"colmap_{offset}" / "input"
    image_paths = sorted(colmap_images_dir.glob("v*.png"))

    with open(originnumpy, 'rb') as numpy_file:
        poses_bounds = np.load(numpy_file)
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)

        llffposes = poses.copy().transpose(1, 2, 0)
        w2c_matriclist = posetow2c_matrcs(llffposes)
        assert (type(w2c_matriclist) == list)

        cameras = []
        
        for i in range(len(poses)):
            cameraname = image_paths[i].stem
            m = w2c_matriclist[i]
            colmapR = m[:3, :3]
            T = m[:3, 3]

            H, W, focal = poses[i, :, -1] / downscale

            colmapQ = rotmat2qvec(colmapR)

            camera = {
                'id': i + 1,
                'filename': f"{cameraname}.png",
                'w': W,
                'h': H,
                'fx': focal,
                'fy': focal,
                'cx': W // 2,
                'cy': H // 2,
                'q': colmapQ,
                't': T,
            }
            cameras.append(camera)

    write_colmap(path, cameras, offset)

def getcolmapsinglen3d(folder, offset):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor --database_path " + dbfile+ " --image_path " + inputimagefolder

    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)


    featurematcher = "colmap exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)

   # threshold is from   https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/scripts/local_colmap_and_resize.sh#L62
    triandmap = "colmap point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)


    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel  + " --output_path " + folder  \
    + " --output_type COLMAP" 
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    removeinput = "rm -r " + inputimagefolder
    exit_code = os.system(removeinput)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)

@dataclass
class ColmapProcessConfig:
    """Configuration for processing frames with COLMAP"""
    scene: str
    """Scene name (e.g., Bartender)"""
    
    base_dir: Optional[str] = None
    """Base directory path. If not provided, defaults to examples/data/GSC/{scene}"""
    
    frame_num: int = 65
    """Number of frames to process"""

def main(config: ColmapProcessConfig):
    # Process parameters
    SCENE = config.scene
    BASE_DIR = config.base_dir if config.base_dir else f"examples/data/GSC/{SCENE}"
    COLMAP_DIR = BASE_DIR + "/colmap"
    FRAME_NUM = config.frame_num
    START_FRAME = DATASET_INFOS[SCENE]["start_frame"]

    print(f"Processing scene {SCENE} with {FRAME_NUM} frames starting from {START_FRAME}")
    print(f"Base directory: {BASE_DIR}")
    print(f"COLMAP directory: {COLMAP_DIR}")

    # Make sure every frame share the same camera extrinsic and intrinsic
    print("Converting DyNeRF format to COLMAP database for each frame...")
    for frame in trange(START_FRAME, START_FRAME+FRAME_NUM):
        convertdynerftocolmapdb(Path(COLMAP_DIR), frame)
    
    # Run COLMAP for each frame to obtain initial point clouds
    print("Running COLMAP for each frame to obtain initial point clouds...")
    for frame in trange(START_FRAME, START_FRAME+FRAME_NUM):
        getcolmapsinglen3d(Path(COLMAP_DIR), frame)
    
    print("COLMAP processing completed!")

if __name__ == "__main__":
    config = tyro.cli(ColmapProcessConfig)
    main(config)