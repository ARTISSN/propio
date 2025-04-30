import math
import numpy as np
from mathutils import Vector
from scipy.special import sph_harm
import json
from pathlib import Path

# ——— USER INPUT ———

K = 5       # number of suns to create
N = 5000    # sampling resolution (more = more accuracy, slower)
energy_scale = 1e-7  # tweak so your scene brightness looks correct

# ——— INTERNALS ———
# 1) Build list of (l,m) pairs
lm = [(0, 0)] \
   + [(1, m) for m in (-1, 0, 1)] \
   + [(2, m) for m in (-2, -1, 0, 1, 2)] \
   + [(3, m) for m in (-3, -2, -1, 0, 1, 2, 3)]

def fibonacci_sphere(n):
    i = np.arange(n)
    phi = np.arccos(1 - 2*(i + 0.5)/n)           # polar angle [0, π]
    theta = 2 * np.pi * ((i + 0.5)/((1 + 5**0.5)/2))
    return phi, theta % (2*np.pi)

def compute_lobes(coeffs, K, N):
    phi, theta = fibonacci_sphere(N)
    L = np.zeros(N)
    for c, (l, m) in zip(coeffs, lm):
        Y = sph_harm(m, l, theta, phi)
        L += c * Y.real
    # pick top K intensities
    idx = np.argsort(L)[-K:][::-1]
    dirs, ints = [], []
    for i in idx:
        th, ph = phi[i], theta[i]
        val = abs(L[i])
        x = math.sin(th)*math.cos(ph)
        y = math.sin(th)*math.sin(ph)
        z = math.cos(th)
        dirs.append(Vector((x, y, z)))
        ints.append(val)
    return dirs, np.clip(ints/np.max(ints), 0, 1)

def direction_to_euler(dir_vec):
    # point local -Z to dir_vec
    return Vector((0,0,-1)).rotation_difference(dir_vec).to_euler()

def extract_coeff_sets(metadata_path):
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    return data

def average_direction(dirs, ints):
    # Weighted average by intensity
    arr = np.array([d[:] for d in dirs])
    weights = np.array(ints)
    avg = np.average(arr, axis=0, weights=weights)
    norm = np.linalg.norm(avg)
    if norm == 0:
        return [0.0, 0.0, 1.0]
    return (avg / norm).tolist()

def compute_sun_direction(coeffs, K=5, N=5000):
    phi, theta = fibonacci_sphere(N)
    L = np.zeros(N)
    for c, (l, m) in zip(coeffs, lm):
        Y = sph_harm(m, l, theta, phi)
        L += c * Y.real
    idx = np.argsort(L)[-K:][::-1]
    dirs, ints = [], []
    for i in idx:
        th, ph = phi[i], theta[i]
        val = abs(L[i])
        x = math.sin(th)*math.cos(ph)
        y = math.sin(th)*math.sin(ph)
        z = math.cos(th)
        dirs.append([x, y, z])
        ints.append(val)
    arr = np.array(dirs)
    weights = np.array(ints)
    avg = np.average(arr, axis=0, weights=weights)
    norm = np.linalg.norm(avg)
    if norm == 0:
        return [0.0, 0.0, 1.0]
    return (avg / norm).tolist()

def compute_top_suns(coeffs, K=3, N=5000):
    phi, theta = fibonacci_sphere(N)
    L = np.zeros(N)
    for c, (l, m) in zip(coeffs, lm):
        Y = sph_harm(m, l, theta, phi)
        L += c * Y.real
    idx = np.argsort(L)[-K:][::-1]
    suns = []
    for i in idx:
        th, ph = phi[i], theta[i]
        intensity = abs(L[i])
        x = math.sin(th)*math.cos(ph)
        y = math.sin(th)*math.sin(ph)
        z = math.cos(th)
        suns.append({"direction": [x, y, z], "intensity": float(intensity)})
    return suns

def main(metadata_path):
    data = extract_coeff_sets(metadata_path)
    changed = False
    for frame_name, frame_data in data.get("frames", {}).items():
        lighting = frame_data.get("lighting")
        if lighting and "coefficients" in lighting:
            coeffs = lighting["coefficients"]
            dirs, ints = compute_lobes(coeffs, K, N)
            avg_dir = average_direction(dirs, ints)
            # Save as a new key in the lighting dict
            lighting["sun"] = avg_dir
            changed = True
            print(f"Frame: {frame_name} | Avg sun direction: {avg_dir}")
    if changed:
        # Save back to file (overwrite)
        with open(metadata_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Updated {metadata_path} with sun directions.")

# ——— MAIN ———
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py path/to/metadata.json")
    else:
        main(sys.argv[1])
