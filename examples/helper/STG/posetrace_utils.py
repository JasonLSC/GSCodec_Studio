import numpy as np

def slerp(q0, q1, t):
    """Spherical linear interpolation (SLERP) between two quaternions."""
    dot = np.dot(q0, q1)
    
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = q0 + t * (q1 - q0)
        result /= np.linalg.norm(result)
        return result
    
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return (s0 * q0) + (s1 * q1)

def interpolate_camera_poses(qvec0, tvec0, qvec1, tvec1, t):
    qvec0 = np.array(qvec0)
    qvec1 = np.array(qvec1)
    tvec0 = np.array(tvec0)
    tvec1 = np.array(tvec1)
    
    q_interpolated = []
    t_interpolated = []
    

    q_interp = slerp(qvec0, qvec1, t)
    t_interp = (1 - t) * tvec0 + t * tvec1
    
    return q_interp, t_interp

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

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def interpolate_camera_poses2(r0, tvec0, r1, tvec1, t):
    r0 = np.array(r0)
    r1 = np.array(r1)
    qvec0 = rotmat2qvec(r0)
    qvec1 = rotmat2qvec(r1)
    tvec0 = np.array(tvec0)
    tvec1 = np.array(tvec1)
    
    q_interpolated = []
    t_interpolated = []
    
    t1 = (np.sin(2 * np.pi * t - 0.5 * np.pi) + 1) / 2
    q_interp = slerp(qvec0, qvec1, t1)
    t_interp = (1 - t1) * tvec0 + t1 * tvec1 + np.array([0,0.3,0]) * np.sin(2 * np.pi * t)
    
    return q_interp, t_interp