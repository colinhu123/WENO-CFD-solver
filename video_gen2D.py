import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# =========== 用户可修改的参数 ===========
folder =  "data/2026-02-16_20-10-28"  # 包含 0.npy..699.npy 和 time.npy 的文件夹
slice_index = 4                  # 取 [:,:,4]
outfile = "slice_animation_with_time.mp4"  # 输出视频名
fps = 20                         # 输出视频帧率
interval_ms = 50                 # 界面播放时每帧间隔（ms）
sample_max = 200                 # 用于估算 vmin/vmax 的抽样帧数（<= n_frames）
compute_exact_vrange = False     # True 则对所有帧做精确扫描以求 vmin/vmax（I/O 多）
bitrate = 4000                   # 输出比特率（可调）
time_filename = "time.npy"       # 时刻文件名
# ========================================

# 辅助排序（0.npy, 1.npy, ...）
def numeric_key(name):
    base = os.path.splitext(name)[0]
    try:
        return int(base)
    except ValueError:
        return base

# 1) 文件列表
files = [f for f in os.listdir(folder) if f.endswith(".npy") and f != time_filename]
if not files:
    raise RuntimeError(f"No .npy files found in folder: {folder}")
files = sorted(files, key=numeric_key)
n_frames = len(files)
print(f"Found {n_frames} frame files (excluding {time_filename}).")

# 2) 读 time.npy
time_path = os.path.join(folder, time_filename)
if not os.path.exists(time_path):
    raise RuntimeError(f"Could not find {time_filename} in {folder}")
t_data = np.load(time_path)
if t_data.ndim != 1:
    raise RuntimeError(f"{time_filename} should be a 1D array of times, but has shape {t_data.shape}")
if len(t_data) < n_frames:
    print(f"Warning: time.npy length ({len(t_data)}) < number of frame files ({n_frames}). Will use min length.")
# 确保索引安全：实际帧数用 min
actual_n_frames = min(n_frames, len(t_data))
if actual_n_frames != n_frames:
    print(f"Using {actual_n_frames} frames (limited by time.npy). Extra files will be ignored.")

# 3) 选择用于估算 vmin/vmax 的样本索引
if compute_exact_vrange:
    sample_indices = list(range(actual_n_frames))
else:
    sample_count = min(sample_max, actual_n_frames)
    if sample_count <= 0:
        sample_indices = [0]
    else:
        step = actual_n_frames / sample_count
        sample_indices = [int(math.floor(i * step)) for i in range(sample_count)]
        if sample_indices[-1] != actual_n_frames - 1:
            sample_indices[-1] = actual_n_frames - 1

print(f"Sampling {len(sample_indices)} frames to estimate vmin/vmax.")

# 4) 估算 vmin/vmax（使用 mmap，不会一次性读入内存）
vmin = np.inf
vmax = -np.inf
for idx in sample_indices:
    fname = files[idx]
    path = os.path.join(folder, fname)
    try:
        arr = np.load(path, mmap_mode='r')
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}; skipping.")
        continue
    if arr.ndim < 3 or slice_index >= arr.shape[2]:
        raise RuntimeError(f"File {fname} has shape {arr.shape}; can't take [:,:,{slice_index}]")
    sl = arr[:, :, slice_index]
    cur_min = float(np.nanmin(sl))
    cur_max = float(np.nanmax(sl))
    if cur_min < vmin: vmin = cur_min
    if cur_max > vmax: vmax = cur_max

if not np.isfinite(vmin) or not np.isfinite(vmax):
    raise RuntimeError("Computed non-finite vmin/vmax. Check your data.")
print(f"Estimated vmin={vmin}, vmax={vmax}")

# 5) 初始化画布（用第一帧）
first_path = os.path.join(folder, files[0])
first_arr = np.load(first_path, mmap_mode='r')
if first_arr.ndim < 3 or slice_index >= first_arr.shape[2]:
    raise RuntimeError(f"First file {files[0]} shape {first_arr.shape} incompatible with slice_index {slice_index}.")
img0 = first_arr[:, :, slice_index]

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(img0, vmin=vmin, vmax=vmax, origin='lower', aspect='auto',cmap='coolwarm')
cbar = fig.colorbar(im, ax=ax)
# 标题模板：Riem. Prob. 1024^{2} WENO HLLC t = ???s
# 使用 mathtext 渲染 1024^2
initial_t = float(t_data[0]) if len(t_data) > 0 else 0.0
ax.set_title(rf"Riem. Prob. $1024^2$ WENO HLLC t = {initial_t:.4f}s")
ax.set_xlabel("X index")
ax.set_ylabel("Y index")

# 6) 更新函数（按帧 mmap 加载）
def update(frame_idx):
    # frame_idx ranges 0..actual_n_frames-1
    fname = files[frame_idx]
    path = os.path.join(folder, fname)
    arr = np.load(path, mmap_mode='r')
    sl = arr[:, :, slice_index]
    im.set_data(sl)
    t = float(t_data[frame_idx])
    ax.set_title(rf"Riem. Prob. $1024^2$ WENO HLLC t = {t:.4f}s")
    return [im]

# 7) 创建并保存动画（blit=False 更可靠）
ani = FuncAnimation(fig, update, frames=actual_n_frames, interval=interval_ms, blit=False)

print("Saving animation — this will stream frames to ffmpeg and should not use large RAM.")
writer = FFMpegWriter(fps=fps, metadata=dict(artist='Auto'), bitrate=bitrate)
ani.save(outfile, writer=writer)
print(f"Saved animation to {outfile}")
