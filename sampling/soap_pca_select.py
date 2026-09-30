# Auto-exported from 数据转换和PCA分析以及训练集测试集选择.ipynb (code cells only; paths are the authors' original ones)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
本脚本用于遍历 root_dir 下所有子目录，
尝试使用 dpdata 读取 DeepMD (npy) 格式的数据，并转换为 ASE 的 Atoms 对象。
在每个子目录下生成一个 extxyz 文件（output.extxyz），
并将所有帧合并到 root_dir 目录下的 total_mace.xyz 文件中。
"""

import os
import dpdata as dp
from dpdata.plugins.ase import ASEStructureFormat
from ase.io import write

# ======================== 变量定义区 ========================
# 根目录：脚本会递归搜索此目录及其所有子目录
root_dir = "./pbe"

# 原子类型映射：如果 type.raw 或 npy 中是整型 (0,1,2,3)，
# 则需要在此列表中按顺序指定对应的元素符号。
my_type_map = ["Li", "La", "Zr", "O","Ta","Nb"]

# 每个子目录写出的 extxyz 文件名
per_dir_extxyz_name = "output.extxyz"

# 合并后在根目录下写出的 extxyz 文件名
merged_extxyz_name = "total_mace.xyz"
# ======================== 变量定义区 ========================

# 用于累积所有转换得到的 Atoms 帧
all_atoms = []

# 遍历 root_dir 及所有子目录
for dirpath, dirnames, filenames in os.walk(root_dir):
    try:
        # 1) 尝试将当前目录作为 deepmd label 数据读取 (npy 格式)
        # 如果目录里混合 npy 和 raw，可以改用 fmt="deepmd/npy/mixed"
        ls = dp.LabeledSystem(dirpath, fmt="deepmd/npy", type_map=my_type_map)
        
        # 2) 转换为 ASE 的 Atoms 对象列表
        atoms_list = ASEStructureFormat().to_system(ls.data)
        
        # 3) 如果有能量、力和 virial，则手动添加到 Atoms 对象
        energies = ls.data.get("energies", None)
        forces = ls.data.get("forces", None)
        virials = ls.data.get("virials", None)  # 有些版本可能键名为 "virial"
        
        if energies is not None:
            for i, atoms in enumerate(atoms_list):
                atoms.info["energy"] = float(energies[i])
        if forces is not None:
            for i, atoms in enumerate(atoms_list):
                atoms.arrays["forces"] = forces[i]
        if virials is not None:
            for i, atoms in enumerate(atoms_list):
                atoms.info["virial"] = virials[i]
        
        # 4) 统计帧数，写出单目录的 extxyz 文件
        num_frames = len(atoms_list)
        output_path = os.path.join(dirpath, per_dir_extxyz_name)
        
        print(f"正在转换目录: {dirpath}")
        print(f"该目录转换的 LabeledSystem 帧数: {num_frames}")
        print(f"输出文件将保存至: {output_path}")
        
        write(output_path, atoms_list, format="extxyz")
        print(f"目录 {dirpath} 转换完成。\n")
        
        # 5) 将该目录所有帧累积到全局列表中
        all_atoms.extend(atoms_list)
        
    except Exception as e:
        # 如果该目录不能作为合法的 deepmd label 数据读取，则跳过
        print(f"跳过目录 {dirpath}: {e}")
        continue

# 6) 合并所有帧并写出到根目录下的 total_mace.xyz 文件
merged_file = os.path.join(root_dir, merged_extxyz_name)
write(merged_file, all_atoms, format="extxyz")

print(f"\n合并完成，共 {len(all_atoms)} 帧数据，写出到 {merged_file}")


# %% ---- cell ----

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from sklearn.cluster import KMeans

# ====== Configurations ======
input_xyz_path = "pbe/total_mace.xyz"
output_xyz_path = "representative_2000_frames.xyz"
cutoff = 6.0
num_bins = 30
sample_atoms_per_frame = 5
n_selected_frames = 2000

# ====== Read and parse XYZ file ======
def parse_xyz_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    frames = []
    i = 0
    while i < len(lines):
        try:
            num_atoms = int(lines[i])
        except ValueError:
            i += 1
            continue
        comment = lines[i+1]
        atoms = lines[i+2:i+2+num_atoms]
        frames.append((num_atoms, comment, atoms))
        i += 2 + num_atoms
    return frames

frames = parse_xyz_file(input_xyz_path)

# ====== Compute local descriptors ======
bin_edges = np.linspace(0, cutoff, num_bins + 1)

def compute_frame_descriptor(frame):
    positions = np.array([list(map(float, line.split()[1:4])) for line in frame[2]])
    n_atoms = len(positions)
    selected = np.random.choice(n_atoms, min(sample_atoms_per_frame, n_atoms), replace=False)
    descriptors = []
    for i in selected:
        dists = distance.cdist([positions[i]], positions, 'euclidean')[0]
        mask = (dists > 1e-6) & (dists < cutoff)
        hist, _ = np.histogram(dists[mask], bins=bin_edges)
        descriptors.append(hist)
    return np.mean(descriptors, axis=0)

all_descriptors = np.array([compute_frame_descriptor(f) for f in frames])

# ====== PCA analysis ======
scaler = StandardScaler()
X_scaled = scaler.fit_transform(all_descriptors)
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# ====== Clustering to select representative frames ======
kmeans = KMeans(n_clusters=n_selected_frames, random_state=42)
kmeans.fit(X_pca)
_, idxs = np.unique(kmeans.labels_, return_index=True)
selected_frames = [frames[i] for i in idxs]

# ====== Write output XYZ ======
with open(output_xyz_path, "w") as f:
    for num_atoms, comment, atoms in selected_frames:
        f.write(f"{num_atoms}\n")
        f.write(comment)
        f.writelines(atoms)

print(f"Saved {len(selected_frames)} representative frames to: {output_xyz_path}")


# %% ---- cell ----
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial import distance

# === 配置参数 ===
input_xyz_path = "pbe/total_mace.xyz"
output_xyz_path = "representative_2000_frames.xyz"
output_plot_path = "pca_selected_2000_frames.png"

cutoff = 6.0
num_bins = 30
sample_atoms_per_frame = 5
n_selected_frames = 2000

# === 读取和解析 XYZ 文件 ===
def parse_xyz_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    frames = []
    i = 0
    while i < len(lines):
        try:
            num_atoms = int(lines[i])
        except ValueError:
            i += 1
            continue
        comment = lines[i+1]
        atoms = lines[i+2:i+2+num_atoms]
        frames.append((num_atoms, comment, atoms))
        i += 2 + num_atoms
    return frames

frames = parse_xyz_file(input_xyz_path)

# === 计算局部结构描述符（Radial Histogram） ===
bin_edges = np.linspace(0, cutoff, num_bins + 1)

def compute_frame_descriptor(frame):
    positions = np.array([list(map(float, line.split()[1:4])) for line in frame[2]])
    n_atoms = len(positions)
    selected = np.random.choice(n_atoms, min(sample_atoms_per_frame, n_atoms), replace=False)
    descriptors = []
    for i in selected:
        dists = distance.cdist([positions[i]], positions, 'euclidean')[0]
        mask = (dists > 1e-6) & (dists < cutoff)
        hist, _ = np.histogram(dists[mask], bins=bin_edges)
        descriptors.append(hist)
    return np.mean(descriptors, axis=0)

desc_all = np.array([compute_frame_descriptor(f) for f in frames])

# === PCA 降维 ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(desc_all)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# === KMeans 聚类选代表帧 ===
kmeans = KMeans(n_clusters=n_selected_frames, random_state=42)
kmeans.fit(X_pca)
_, selected_idxs = np.unique(kmeans.labels_, return_index=True)
selected_frames = [frames[i] for i in selected_idxs]

# === 保存选中的帧为新 XYZ 文件 ===
with open(output_xyz_path, "w") as f:
    for num_atoms, comment, atoms in selected_frames:
        f.write(f"{num_atoms}\n")
        f.write(comment)
        f.writelines(atoms)

print(f"✅ 代表性帧已保存：{output_xyz_path}")

# === 绘制 PCA 分布图 ===
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X_pca[:, 0], X_pca[:, 1], s=1, c='gray', alpha=0.3, label="All frames")
ax.scatter(X_pca[selected_idxs, 0], X_pca[selected_idxs, 1], s=10, c='red', label="Selected 2000")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_title("PCA Projection of Local Environments")
ax.legend()
plt.tight_layout()
plt.savefig(output_plot_path, dpi=300)
print(f"📊 PCA图已保存：{output_plot_path}")

# %% ---- cell ----

import numpy as np
from ase.io import read, write
from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== Configuration ==========
input_xyz = "./pbe/total_mace.xyz"
output_xyz = "soap_selected_2000_frames.xyz"
output_plot = "soap_pca_2000.png"
n_selected_frames = 2000
soap_params = {
    "r_cut": 6.0,
    "n_max": 8,
    "l_max": 6,
    "sigma": 0.5,
    "species":["Li", "La", "Zr", "O","Ta","Nb"],  # adjust to match your data
    "periodic": False,
    "sparse": False
}
sample_atoms_per_frame = 5

# ========== Load Data ==========
print("Loading XYZ frames...")
frames = read(input_xyz, index=":")
print(f"Loaded {len(frames)} frames")

# ========== Initialize SOAP ==========
print("Generating SOAP descriptors...")
soap = SOAP(**soap_params)

# ========== Compute per-frame descriptor ==========
desc_list = []
for atoms in tqdm(frames):
    natoms = len(atoms)
    selected_idx = np.random.choice(natoms, min(sample_atoms_per_frame, natoms), replace=False)
    descs = soap.create(atoms,selected_idx)
    desc_list.append(np.mean(descs, axis=0))

X = np.array(desc_list)

# ========== PCA ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ========== KMeans Selection ==========
print("Clustering with KMeans...")
kmeans = KMeans(n_clusters=n_selected_frames, random_state=42)
kmeans.fit(X_pca)
_, selected_idxs = np.unique(kmeans.labels_, return_index=True)
selected_frames = [frames[i] for i in selected_idxs]

# ========== Save Selected Frames ==========
write(output_xyz, selected_frames)
print(f"✅ Saved {len(selected_frames)} frames to {output_xyz}")

# ========== Plot ==========
plt.figure(figsize=(6, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=1, c="gray", label="All")
plt.scatter(X_pca[selected_idxs, 0], X_pca[selected_idxs, 1], s=10, c="red", label="Selected 2000")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("SOAP PCA of Local Atomic Environments")
plt.legend()
plt.tight_layout()
plt.savefig(output_plot, dpi=300)
print(f"📊 PCA plot saved to {output_plot}")

# %% ---- cell ----
import os
import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list
from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------- Configuration ----------
input_xyz    = "./pbe/total_mace.xyz"
output_root  = "."                                   # 创建子目录的根路径
target_sizes = [500, 1000, 2000, 4000, 8000]         # 需要的代表帧数量
soap_params  = {
    "r_cut":   6.0,
    "n_max":   8,
    "l_max":   6,
    "sigma":   0.5,
    "species": ["Li", "La", "Zr", "O","Ta","Nb"],
    "periodic": False,
    "sparse":   False,
}
sample_atoms_per_frame = 20                          # 每帧抽多少原子
neighbor_cutoff        = 4.0                         # “表面原子”配位半径
random_state           = 42

# ---------- Helper: pick low-coordination (“surface”) atoms ----------
def get_surface_atoms(atoms, num_select=20, cutoff=4.0):
    """返回 num_select 个低配位原子下标；不足时全部返回。"""
    i_list, _ = neighbor_list("ij", atoms, cutoff)
    coord = np.bincount(i_list, minlength=len(atoms))
    order = np.argsort(coord)              # 配位数从低到高
    return order[:min(num_select, len(order))]

# ---------- Load structures ----------
print("📥 Loading XYZ frames …")
frames = read(input_xyz, index=":")
n_total = len(frames)
print(f"✅ Loaded {n_total} frames")

# ---------- Pre-compute SOAP on all frames ----------
print("🔍 Calculating SOAP descriptors (surface-prioritized) …")
soap = SOAP(**soap_params)
X, valid_frames = [], []
for atoms in tqdm(frames, desc="SOAP"):
    try:
        sel_idx = get_surface_atoms(atoms,
                                    num_select=sample_atoms_per_frame,
                                    cutoff=neighbor_cutoff)
        desc = soap.create(atoms, sel_idx)
        X.append(np.mean(desc, axis=0))
        valid_frames.append(atoms)
    except Exception as e:
        print(f"⚠️  Skip a frame: {e}")

X = np.asarray(X)
print(f"✅ SOAP done on {len(valid_frames)} frames")

# ---------- PCA once ----------
print("📊 Running PCA …")
X_scaled = StandardScaler().fit_transform(X)
X_pca    = PCA(n_components=2, random_state=random_state).fit_transform(X_scaled)

# ---------- Core routine for one target size ----------
def export_subset(n_select: int):
    if len(valid_frames) < n_select:
        print(f"[WARN] only {len(valid_frames)} frames available; skip {n_select}.")
        return

    print(f"\n[ {n_select} ] KMeans clustering …")
    kmeans = KMeans(n_clusters=n_select, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(X_pca)
    _, unique_idx = np.unique(labels, return_index=True)
    sel_idx = np.sort(unique_idx)
    subset  = [valid_frames[i] for i in sel_idx]

    # make directory
    d = f"{n_select}-fra"
    os.makedirs(d, exist_ok=True)

    # save xyz
    xyzf = os.path.join(d, f"soap_selected_{n_select}_frames.xyz")
    write(xyzf, subset)
    print(f"    ✅ XYZ  -> {xyzf}")

    # save png
    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=1, c="grey", alpha=0.3, label="All")
    plt.scatter(X_pca[sel_idx, 0], X_pca[sel_idx, 1],
                s=10, c="red", label=f"Selected {n_select}")
    plt.xlabel("PCA 1");  plt.ylabel("PCA 2")
    plt.title(f"SOAP-PCA (n = {n_select})")
    plt.legend(); plt.tight_layout()
    pngf = os.path.join(d, f"soap_pca_{n_select}.png")
    plt.savefig(pngf, dpi=300); plt.close()
    print(f"    📈 PNG  -> {pngf}")

    # save dat
    datf = os.path.join(d, f"soap_pca_{n_select}.dat")
    with open(datf, "w") as f:
        f.write("# idx  PC1  PC2  selected(0/1)\n")
        for i, (x, y) in enumerate(X_pca):
            flag = 1 if i in sel_idx else 0
            f.write(f"{i} {x:.6f} {y:.6f} {flag}\n")
    print(f"    📄 DAT  -> {datf}")

# ---------- Loop over required sizes ----------
for n in target_sizes:
    export_subset(n)

print("\n🎉  All tasks finished.")

# %% ---- cell ----

import os, glob
import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list
from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------- Global settings ----------
folders        = ["500-fra", "1000-fra", "2000-fra", "4000-fra", "8000-fra"]
test_ratio     = 0.10                     # 10 % 测试集
random_state   = 42
soap_params    = {
    "r_cut":   6.0,
    "n_max":   8,
    "l_max":   6,
    "sigma":   0.5,
    "species": ["Li", "La", "Zr", "O","Ta","Nb"],
    "periodic": False,
    "sparse":   False,
}
sample_atoms_per_frame = 20
neighbor_cutoff        = 4.0              # 配位判定半径

# ---------- Helper: pick low-coordination atoms ----------
def get_surface_atoms(atoms, num_select=20, cutoff=4.0):
    i_list, _ = neighbor_list("ij", atoms, cutoff)
    coord = np.bincount(i_list, minlength=len(atoms))
    return np.argsort(coord)[:min(num_select, len(atoms))]

# ---------- Main loop ----------
for d in folders:
    xyz_list = glob.glob(os.path.join(d, "soap_selected_*_frames.xyz"))
    if not xyz_list:
        print(f"[SKIP] {d}: xyz file not found.")
        continue

    xyz_path = xyz_list[0]
    frames   = read(xyz_path, index=":")
    n_total  = len(frames)
    if n_total == 0:
        print(f"[WARN] {d}: empty xyz.")
        continue

    print(f"\n📂 {d}: total {n_total} frames")

    # ----- SOAP descriptors -----
    soap = SOAP(**soap_params)
    X = []
    for atoms in tqdm(frames, desc=f"SOAP {d}"):
        sel = get_surface_atoms(atoms,
                                num_select=sample_atoms_per_frame,
                                cutoff=neighbor_cutoff)
        X.append(np.mean(soap.create(atoms, sel), axis=0))
    X = np.asarray(X)

    # ----- PCA (2 components) -----
    X_pca = PCA(n_components=2,
                random_state=random_state).fit_transform(
                StandardScaler().fit_transform(X))

    # ----- KMeans for test-set diversity -----
    n_test = max(1, int(round(test_ratio * n_total)))
    kmeans = KMeans(n_clusters=n_test, random_state=random_state, n_init="auto")
    kmeans.fit(X_pca)
    _, test_idx = np.unique(kmeans.labels_, return_index=True)
    train_idx   = sorted(set(range(n_total)) - set(test_idx))

    train_frames = [frames[i] for i in train_idx]
    test_frames  = [frames[i] for i in test_idx]

    # ----- Save xyz -----
    train_xyz = os.path.join(d, f"train_{len(train_frames)}_frames.xyz")
    test_xyz  = os.path.join(d, f"test_{len(test_frames)}_frames.xyz")
    write(train_xyz, train_frames)
    write(test_xyz,  test_frames)
    print(f"   ✅ Train -> {train_xyz}")
    print(f"   ✅ Test  -> {test_xyz}")

    # ----- PCA plot (train vs test) -----
    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[train_idx, 0], X_pca[train_idx, 1],
                s=4, c="red", alpha=0.4, label=f"Train {len(train_frames)}")
    plt.scatter(X_pca[test_idx, 0],  X_pca[test_idx, 1],
                s=10, c="blue", label=f"Test {len(test_frames)}")
    plt.xlabel("PCA Component 1");  plt.ylabel("PCA Component 2")
    plt.title(f"{d} – SOAP-PCA Split (Train vs Test)")
    plt.legend(); plt.tight_layout()
    png_path = os.path.join(d, "soap_pca_split_train_test.png")
    plt.savefig(png_path, dpi=300); plt.close()
    print(f"   📈 Plot -> {png_path}")

    # ----- Save .dat -----
    dat_path = os.path.join(d, "soap_pca_split_train_test.dat")
    with open(dat_path, "w") as f:
        f.write("# idx  PC1  PC2  split(0=train,1=test)\n")
        for i, (pc1, pc2) in enumerate(X_pca):
            flag = 0 if i in train_idx else 1
            f.write(f"{i} {pc1:.6f} {pc2:.6f} {flag}\n")
    print(f"   📄 Dat  -> {dat_path}")

print("\n🎉  All datasets split & visualized.")


# %% ---- cell ----

import numpy as np
from ase.io import read, write
from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== 配置 ==========
input_xyz = "soap_selected_2000_frames.xyz"
train_output = "soap_train_1800.xyz"
test_output = "soap_test_200.xyz"
plot_output = "soap_pca_split_train_test.png"
soap_params = {
    "r_cut": 6.0,
    "n_max": 8,
    "l_max": 6,
    "sigma": 0.5,
    "species": ["Li", "La", "Zr", "O","Ta","Nb"],
    "periodic": False,
    "sparse": False
}
sample_atoms_per_frame = 20
n_test = 200

# ========== 加载结构 ==========
frames = read(input_xyz, index=":")
print(f"Loaded {len(frames)} frames")

soap = SOAP(**soap_params)

desc_list = []
for atoms in tqdm(frames):
    natoms = len(atoms)
    selected = np.random.choice(natoms, min(sample_atoms_per_frame, natoms), replace=False)
    descs = soap.create(atoms, selected)
    desc_list.append(np.mean(descs, axis=0))
X = np.array(desc_list)

# ========== PCA + KMeans ==========
X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(X_scaled)

kmeans = KMeans(n_clusters=n_test, random_state=42)
kmeans.fit(X_pca)
_, test_indices = np.unique(kmeans.labels_, return_index=True)

# ========== 生成训练/测试集 ==========
test_set = [frames[i] for i in test_indices]
train_indices = sorted(list(set(range(len(frames))) - set(test_indices)))
train_set = [frames[i] for i in train_indices]

write(train_output, train_set)
write(test_output, test_set)
print(f"✅ Saved {len(train_set)} train frames → {train_output}")
print(f"✅ Saved {len(test_set)} test frames → {test_output}")

# ========== 绘图 ==========
plt.figure(figsize=(6, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=1, c='gray', alpha=0.3, label="All 2000")
plt.scatter(X_pca[test_indices, 0], X_pca[test_indices, 1], s=10, c='blue', label="Test 200")
plt.scatter(X_pca[train_indices, 0], X_pca[train_indices, 1], s=10, c='red', label="Train 1800")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("SOAP PCA split: Train vs Test")
plt.legend()
plt.tight_layout()
plt.savefig(plot_output, dpi=300)
print(f"📊 PCA plot saved to {plot_output}")


# %% ---- cell ----
pwd

# %% ---- cell ----
#!/usr/bin/env python3
# crossval_500fra_generator.py
import os, shutil, glob, random
import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list
from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== 用户可配置区域 ==========
TOTAL_XYZ          = "./pbe/total_mace.xyz"   # 整个数据集
SEEDS              = [0, 1, 2, 3, 4]                   # 五折交叉验证
N_FRAMES_SUBSET    = 500                               # 每折 500 帧
TEST_RATIO         = 0.10                              # 10% 测试集
LOW_COORD_ATOMS    = 20                                # 每帧抽 20 个低配位原子
NEIGHBOR_CUTOFF    = 4.0                               # 判定配位数的半径
SOAP_PARAMS = dict(                                   # 与原脚本完全一致
    r_cut   = 6.0, n_max = 8, l_max = 6, sigma = 0.5,
    species = ["Li", "La", "Zr", "O", "Ta", "Nb"],
    periodic=False, sparse=False,
)
# ========== END 用户可配置区域 ==========

# ---------- 帮助函数 ----------
def get_surface_atoms(atoms, k=20, cutoff=4.0):
    i_list, _ = neighbor_list("ij", atoms, cutoff)
    coord = np.bincount(i_list, minlength=len(atoms))
    return np.argsort(coord)[:min(k, len(atoms))]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# ---------- 读取总数据集 ----------
frames_total = read(TOTAL_XYZ, index=":")
n_total      = len(frames_total)
assert N_FRAMES_SUBSET <= n_total, "子集大小超过总帧数！"

print(f"📦  总数据集: {TOTAL_XYZ}  (共 {n_total} 帧)")
soap = SOAP(**SOAP_PARAMS)              # 只实例化一次

# ---------- 对每个随机种子生成子集 ----------
for seed in SEEDS:
    subset_dir = f"500-fra_seed{seed}"
    ensure_dir(subset_dir)

    #############################################
    # 1) 采样 500 帧（保证各子集互不重复）       #
    #############################################
    rng = np.random.default_rng(seed)
    subset_idx = rng.choice(n_total, size=N_FRAMES_SUBSET, replace=False)
    subset_frames = [frames_total[i] for i in subset_idx]
    subset_xyz = os.path.join(subset_dir,
                              f"soap_selected_{N_FRAMES_SUBSET}_frames_seed{seed}.xyz")
    write(subset_xyz, subset_frames)
    print(f"\n[Seed {seed}] 500-帧子集写入 ➜ {subset_xyz}")

    #############################################
    # 2) 计算 SOAP 描述符                         #
    #############################################
    X_vecs = []
    for atoms in tqdm(subset_frames, desc=f"SOAP(seed={seed})"):
        sel = get_surface_atoms(atoms, LOW_COORD_ATOMS, NEIGHBOR_CUTOFF)
        X_vecs.append(np.mean(soap.create(atoms, sel), axis=0))
    X_vecs = np.asarray(X_vecs)

    #############################################
    # 3) PCA ↓ 2D                               #
    #############################################
    scaler = StandardScaler().fit(X_vecs)
    X_pca  = PCA(n_components=2, random_state=seed)\
                .fit_transform(scaler.transform(X_vecs))

    #############################################
    # 4) KMeans 选择多样化测试集                 #
    #############################################
    n_test  = max(1, int(round(TEST_RATIO * N_FRAMES_SUBSET)))
    kmeans  = KMeans(n_clusters=n_test, random_state=seed, n_init="auto")
    kmeans.fit(X_pca)
    _, test_idx = np.unique(kmeans.labels_, return_index=True)
    train_idx   = sorted(set(range(N_FRAMES_SUBSET)) - set(test_idx))

    train_xyz = os.path.join(subset_dir,
                             f"train_seed{seed}_{len(train_idx)}.xyz")
    test_xyz  = os.path.join(subset_dir,
                             f"test_seed{seed}_{len(test_idx)}.xyz")
    write(train_xyz, [subset_frames[i] for i in train_idx])
    write(test_xyz,  [subset_frames[i] for i in test_idx])
    print(f"   ✅ Train ➜ {train_xyz}")
    print(f"   ✅ Test  ➜ {test_xyz}")

    #############################################
    # 5) 可视化                                 #
    #############################################
    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[train_idx, 0], X_pca[train_idx, 1],
                s=4, c="red", alpha=0.4, label=f"Train {len(train_idx)}")
    plt.scatter(X_pca[test_idx, 0],  X_pca[test_idx, 1],
                s=10, c="blue", label=f"Test {len(test_idx)}")
    plt.xlabel("PCA 1"); plt.ylabel("PCA 2")
    plt.title(f"500-fra (seed {seed}) – Train vs Test")
    plt.legend(); plt.tight_layout()
    png_path = os.path.join(subset_dir, f"soap_pca_split_seed{seed}.png")
    plt.savefig(png_path, dpi=300); plt.close()
    print(f"   📈 Plot  ➜ {png_path}")

    #############################################
    # 6) 保存 .dat                              #
    #############################################
    dat_path = os.path.join(subset_dir, f"soap_pca_split_seed{seed}.dat")
    with open(dat_path, "w") as f:
        f.write("# idx  PC1  PC2  split(0=train,1=test)\n")
        for i, (pc1, pc2) in enumerate(X_pca):
            flag = 0 if i in train_idx else 1
            f.write(f"{i} {pc1:.6f} {pc2:.6f} {flag}\n")
    print(f"   📄 Dat   ➜ {dat_path}")

print("\n🎉  五折 500-fra 子集 + Train/Test 划分全部完成！")


# %% ---- cell ----
import os, glob, numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list
from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===== 用户可配 =====
TOTAL_XYZ          = "./pbe/total_mace.xyz"   # 总轨迹
SEEDS              = [0, 1, 2, 3, 4]                   # 五折交叉验证
SUBSET_FRAMES      = 500                               # 每折采样 500 帧
TEST_RATIO         = 0.10                              # 10 % 测试集
LOW_COORD_ATOMS    = 20                                # 每帧取 20 个低配位原子
NEIGHBOR_CUTOFF    = 4.0                               # 判定配位数半径
SOAP_PARAMS = dict(
    r_cut=6.0, n_max=8, l_max=6, sigma=0.5,
    species=["Li", "La", "Zr", "O", "Ta", "Nb"],
    periodic=False, sparse=False,
)
# ====================

def get_surface_atoms(atoms, k=20, cutoff=4.0):
    i_list, _ = neighbor_list("ij", atoms, cutoff)
    coord = np.bincount(i_list, minlength=len(atoms))
    return np.argsort(coord)[:min(k, len(atoms))]

# ---------- 1) 读取总数据集 ----------
frames_all = read(TOTAL_XYZ, index=":")
n_total    = len(frames_all)
assert SUBSET_FRAMES <= n_total, "500 > 总帧数！"
print(f"📦 总数据集: {n_total} 帧  ➜  计算全局 SOAP …")

# ---------- 2) 全局 SOAP 描述符 ----------
soap = SOAP(**SOAP_PARAMS)
X_all = []
for atoms in tqdm(frames_all, desc="全局 SOAP"):
    sel = get_surface_atoms(atoms, LOW_COORD_ATOMS, NEIGHBOR_CUTOFF)
    X_all.append(np.mean(soap.create(atoms, sel), axis=0))
X_all = np.asarray(X_all)
scaler_global = StandardScaler().fit(X_all)
X_all_scaled  = scaler_global.transform(X_all)

# ---------- 3) 针对每个随机种子生成 500-fra ----------
for seed in SEEDS:
    subset_dir = f"500-fra_seed{seed}"
    os.makedirs(subset_dir, exist_ok=True)

    # —— 3-a  KMeans(n_clusters=500) → 选帧代表 —— #
    print(f"\n[Seed {seed}]  K-Means 采样 500 帧 …")
    km_500 = KMeans(n_clusters=SUBSET_FRAMES,
                    random_state=seed, n_init="auto")
    km_500.fit(X_all_scaled)
    centers = km_500.cluster_centers_

    # 找到每个 center 最近的原始帧
    subset_idx = []
    for c in centers:
        dist = np.linalg.norm(X_all_scaled - c, axis=1)
        subset_idx.append(int(dist.argmin()))
    subset_idx = np.unique(subset_idx)         # 理论上应正好 500
    assert len(subset_idx) == SUBSET_FRAMES, "簇中心不唯一？"

    subset_frames = [frames_all[i] for i in subset_idx]
    subset_xyz = os.path.join(subset_dir,
                              f"soap_selected_{SUBSET_FRAMES}_frames_seed{seed}.xyz")
    write(subset_xyz, subset_frames)
    print(f"   ✔ 500-fra 子集写入 → {subset_xyz}")

    # —— 3-b  对子集再算 SOAP / PCA —— #
    X_sub = []
    for atoms in tqdm(subset_frames, desc=f"SOAP(seed={seed})"):
        sel = get_surface_atoms(atoms, LOW_COORD_ATOMS, NEIGHBOR_CUTOFF)
        X_sub.append(np.mean(soap.create(atoms, sel), axis=0))
    X_sub = np.asarray(X_sub)

    scaler_sub = StandardScaler().fit(X_sub)
    X_pca = PCA(n_components=2, random_state=seed)\
            .fit_transform(scaler_sub.transform(X_sub))

    # —— 3-c  10 % 测试集 —— #
    n_test = max(1, int(round(TEST_RATIO * SUBSET_FRAMES)))
    km_test = KMeans(n_clusters=n_test,
                     random_state=seed, n_init="auto").fit(X_pca)
    _, test_idx = np.unique(km_test.labels_, return_index=True)
    train_idx   = sorted(set(range(SUBSET_FRAMES)) - set(test_idx))

    train_xyz = os.path.join(subset_dir,
                             f"train_seed{seed}_{len(train_idx)}.xyz")
    test_xyz  = os.path.join(subset_dir,
                             f"test_seed{seed}_{len(test_idx)}.xyz")
    write(train_xyz, [subset_frames[i] for i in train_idx])
    write(test_xyz,  [subset_frames[i] for i in test_idx])
    print(f"   ✅ Train → {train_xyz}")
    print(f"   ✅ Test  → {test_xyz}")

    # —— 3-d  可视化 —— #
    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[train_idx, 0], X_pca[train_idx, 1],
                s=4, c="red",  alpha=0.4, label=f"Train {len(train_idx)}")
    plt.scatter(X_pca[test_idx, 0],  X_pca[test_idx, 1],
                s=10, c="blue", alpha=1.0, label=f"Test {len(test_idx)}")
    plt.xlabel("PCA 1"); plt.ylabel("PCA 2")
    plt.title(f"500-fra(seed {seed}) – Train vs Test")
    plt.legend(); plt.tight_layout()
    png = os.path.join(subset_dir, f"soap_pca_split_seed{seed}.png")
    plt.savefig(png, dpi=300); plt.close()
    print(f"   📈 图  → {png}")

    # —— 3-e  索引 .dat —— #
    dat = os.path.join(subset_dir, f"soap_pca_split_seed{seed}.dat")
    with open(dat, "w") as f:
        f.write("# idx  PC1  PC2  split(0=train,1=test)\n")
        for i, (pc1, pc2) in enumerate(X_pca):
            flag = 0 if i in train_idx else 1
            f.write(f"{i} {pc1:.6f} {pc2:.6f} {flag}\n")
    print(f"   📄 Dat → {dat}")

print("\n🎉  五套 500-fra 采样 + Train/Test 划分全部完成！")

# %% ---- cell ----
#!/usr/bin/env python3
# two_stage_soap_sampling.py
# """
# 目的：
# 1. 先在“总数据集”里用 SOAP+KMeans(n=500) 选出 5 套 500-fra 子集（不同 seed 产生差异，子集间允许重叠，但每套内部无重复）；
# 2. 对每套 500-fra 再执行 SOAP→PCA→KMeans(10 %) 划分 450/50 的 Train/Test；
# 3. 输出 xyz、PCA 图、索引 .dat 文件，方便后续交叉验证。
# """

import os, numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list
from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========= 用户可配置 =========
TOTAL_XYZ          = "./pbe/total_mace.xyz"   # 整个轨迹
SEEDS              = [1, 42, 139, 10086, 512]                   # 五折交叉验证
SUBSET_FRAMES      = 500                               # 每折取 500 帧
TEST_RATIO         = 0.10                              # 10 % 测试集
LOW_COORD_ATOMS    = 20                                # 每帧挑 20 个低配位原子
NEIGHBOR_CUTOFF    = 4.0                               # 配位判定半径
SOAP_PARAMS = dict(
    r_cut=6.0, n_max=8, l_max=6, sigma=0.5,
    species=["Li", "La", "Zr", "O", "Ta", "Nb"],
    periodic=False, sparse=False,
)
# =================================

def get_surface_atoms(atoms, k=20, cutoff=4.0):
    """返回配位数最低的 k 个原子索引（表面/低配位原子近似）"""
    i_list, _ = neighbor_list("ij", atoms, cutoff)
    coord = np.bincount(i_list, minlength=len(atoms))
    return np.argsort(coord)[:min(k, len(atoms))]

# ---------- 1) 读取总数据集 ----------
frames_all = read(TOTAL_XYZ, index=":")
n_total    = len(frames_all)
assert SUBSET_FRAMES <= n_total, "子集帧数 > 总帧数！"
print(f"📦  总数据集: {n_total} 帧  (文件: {TOTAL_XYZ})")

# ---------- 2) 计算全局 SOAP 描述符 ----------
print("🔧  计算全局 SOAP 描述符 …")
soap = SOAP(**SOAP_PARAMS)
X_all = []
for atoms in tqdm(frames_all, desc="全局 SOAP"):
    sel = get_surface_atoms(atoms, LOW_COORD_ATOMS, NEIGHBOR_CUTOFF)
    X_all.append(np.mean(soap.create(atoms, sel), axis=0))
X_all = np.asarray(X_all)
scaler_global = StandardScaler().fit(X_all)
X_all_scaled  = scaler_global.transform(X_all)

# ---------- 3) 对每个随机种子生成 500-fra ----------
for seed in SEEDS:
    subset_dir = f"500-fra_seed{seed}"
    os.makedirs(subset_dir, exist_ok=True)

    # —— 3-a  全局 KMeans(n=500) 采样 —— #
    print(f"\n[Seed {seed}] ⏩  全局 KMeans 采样 500 帧 …")
    km_500 = KMeans(n_clusters=SUBSET_FRAMES,
                    random_state=seed, n_init="auto")
    km_500.fit(X_all_scaled)
    centers = km_500.cluster_centers_

    # **FIX** 递补 + 随机补齐，保证内部无重复
    subset_idx = []
    for i, c in enumerate(centers):
        members = np.where(km_500.labels_ == i)[0]
        if len(members) == 0:
            continue                       # 空簇略过
        d = np.linalg.norm(X_all_scaled[members] - c, axis=1)
        for m in members[np.argsort(d)]:  # 最近、次近、…
            if m not in subset_idx:
                subset_idx.append(m)
                break
    if len(subset_idx) < SUBSET_FRAMES:    # 随机补齐
        remain = list(set(range(n_total)) - set(subset_idx))
        rng = np.random.default_rng(seed)
        extra = rng.choice(remain, SUBSET_FRAMES - len(subset_idx),
                           replace=False)
        subset_idx.extend(extra)
    # 现在 subset_idx 含 500 个“全局唯一”索引（不同 seed 允许交叠）
    subset_frames = [frames_all[i] for i in subset_idx]

    subset_xyz = os.path.join(
        subset_dir, f"soap_selected_{SUBSET_FRAMES}_frames_seed{seed}.xyz")
    write(subset_xyz, subset_frames)
    print(f"   ✔ 500-fra 子集写入 → {subset_xyz}")

    # —— 3-b  子集再算 SOAP / PCA —— #
    print(f"   🔧  计算子集 SOAP & PCA …")
    X_sub = []
    for atoms in tqdm(subset_frames, desc=f"SOAP(seed={seed})"):
        sel = get_surface_atoms(atoms, LOW_COORD_ATOMS, NEIGHBOR_CUTOFF)
        X_sub.append(np.mean(soap.create(atoms, sel), axis=0))
    X_sub = np.asarray(X_sub)

    scaler_sub = StandardScaler().fit(X_sub)
    X_pca = PCA(n_components=2, random_state=seed)\
            .fit_transform(scaler_sub.transform(X_sub))

    # —— 3-c  10 % 测试集 (KMeans 多样性) —— #
    n_test = max(1, int(round(TEST_RATIO * SUBSET_FRAMES)))
    km_test = KMeans(n_clusters=n_test,
                     random_state=seed, n_init="auto").fit(X_pca)
    _, test_idx = np.unique(km_test.labels_, return_index=True)
    train_idx   = sorted(set(range(SUBSET_FRAMES)) - set(test_idx))

    train_xyz = os.path.join(subset_dir,
                             f"train_seed{seed}_{len(train_idx)}.xyz")
    test_xyz  = os.path.join(subset_dir,
                             f"test_seed{seed}_{len(test_idx)}.xyz")
    write(train_xyz, [subset_frames[i] for i in train_idx])
    write(test_xyz,  [subset_frames[i] for i in test_idx])
    print(f"   ✅ Train → {train_xyz}")
    print(f"   ✅ Test  → {test_xyz}")

    # —— 3-d  可视化 —— #
    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[train_idx, 0], X_pca[train_idx, 1],
                s=4, c="red",  alpha=0.4, label=f"Train {len(train_idx)}")
    plt.scatter(X_pca[test_idx, 0],  X_pca[test_idx, 1],
                s=10, c="blue", alpha=1.0, label=f"Test {len(test_idx)}")
    plt.xlabel("PCA 1"); plt.ylabel("PCA 2")
    plt.title(f"500-fra (seed {seed}) – Train vs Test")
    plt.legend(); plt.tight_layout()
    png = os.path.join(subset_dir, f"soap_pca_split_seed{seed}.png")
    plt.savefig(png, dpi=300); plt.close()
    print(f"   📈 图  → {png}")

    # —— 3-e  索引 .dat —— #
    dat = os.path.join(subset_dir, f"soap_pca_split_seed{seed}.dat")
    with open(dat, "w") as f:
        f.write("# idx  PC1  PC2  split(0=train,1=test)\n")
        for i, (pc1, pc2) in enumerate(X_pca):
            flag = 0 if i in train_idx else 1
            f.write(f"{i} {pc1:.6f} {pc2:.6f} {flag}\n")
    print(f"   📄 Dat → {dat}")

print("\n🎉  全部完成：五套 500-fra 子集 + 各自的 Train/Test 划分已生成！")


# %% ---- cell ----
import os
import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list
from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------- Configuration ----------
input_xyz    = "./pbe/total_mace.xyz"
output_root  = "."                                   # 创建子目录的根路径
target_sizes = [100, 200, 400]         # 需要的代表帧数量
soap_params  = {
    "r_cut":   6.0,
    "n_max":   8,
    "l_max":   6,
    "sigma":   0.5,
    "species": ["Li", "La", "Zr", "O","Ta","Nb"],
    "periodic": False,
    "sparse":   False,
}
sample_atoms_per_frame = 20                          # 每帧抽多少原子
neighbor_cutoff        = 4.0                         # “表面原子”配位半径
random_state           = 42

# ---------- Helper: pick low-coordination (“surface”) atoms ----------
def get_surface_atoms(atoms, num_select=20, cutoff=4.0):
    """返回 num_select 个低配位原子下标；不足时全部返回。"""
    i_list, _ = neighbor_list("ij", atoms, cutoff)
    coord = np.bincount(i_list, minlength=len(atoms))
    order = np.argsort(coord)              # 配位数从低到高
    return order[:min(num_select, len(order))]

# ---------- Load structures ----------
print("📥 Loading XYZ frames …")
frames = read(input_xyz, index=":")
n_total = len(frames)
print(f"✅ Loaded {n_total} frames")

# ---------- Pre-compute SOAP on all frames ----------
print("🔍 Calculating SOAP descriptors (surface-prioritized) …")
soap = SOAP(**soap_params)
X, valid_frames = [], []
for atoms in tqdm(frames, desc="SOAP"):
    try:
        sel_idx = get_surface_atoms(atoms,
                                    num_select=sample_atoms_per_frame,
                                    cutoff=neighbor_cutoff)
        desc = soap.create(atoms, sel_idx)
        X.append(np.mean(desc, axis=0))
        valid_frames.append(atoms)
    except Exception as e:
        print(f"⚠️  Skip a frame: {e}")

X = np.asarray(X)
print(f"✅ SOAP done on {len(valid_frames)} frames")

# ---------- PCA once ----------
print("📊 Running PCA …")
X_scaled = StandardScaler().fit_transform(X)
X_pca    = PCA(n_components=2, random_state=random_state).fit_transform(X_scaled)

# ---------- Core routine for one target size ----------
def export_subset(n_select: int):
    if len(valid_frames) < n_select:
        print(f"[WARN] only {len(valid_frames)} frames available; skip {n_select}.")
        return

    print(f"\n[ {n_select} ] KMeans clustering …")
    kmeans = KMeans(n_clusters=n_select, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(X_pca)
    _, unique_idx = np.unique(labels, return_index=True)
    sel_idx = np.sort(unique_idx)
    subset  = [valid_frames[i] for i in sel_idx]

    # make directory
    d = f"{n_select}-fra-20250611"
    os.makedirs(d, exist_ok=True)

    # save xyz
    xyzf = os.path.join(d, f"soap_selected_{n_select}_frames.xyz")
    write(xyzf, subset)
    print(f"    ✅ XYZ  -> {xyzf}")

    # save png
    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=1, c="grey", alpha=0.3, label="All")
    plt.scatter(X_pca[sel_idx, 0], X_pca[sel_idx, 1],
                s=10, c="red", label=f"Selected {n_select}")
    plt.xlabel("PCA 1");  plt.ylabel("PCA 2")
    plt.title(f"SOAP-PCA (n = {n_select})")
    plt.legend(); plt.tight_layout()
    pngf = os.path.join(d, f"soap_pca_{n_select}.png")
    plt.savefig(pngf, dpi=300); plt.close()
    print(f"    📈 PNG  -> {pngf}")

    # save dat
    datf = os.path.join(d, f"soap_pca_{n_select}.dat")
    with open(datf, "w") as f:
        f.write("# idx  PC1  PC2  selected(0/1)\n")
        for i, (x, y) in enumerate(X_pca):
            flag = 1 if i in sel_idx else 0
            f.write(f"{i} {x:.6f} {y:.6f} {flag}\n")
    print(f"    📄 DAT  -> {datf}")

# ---------- Loop over required sizes ----------
for n in target_sizes:
    export_subset(n)

print("\n🎉  All tasks finished.")

# %% ---- cell ----
import os
import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list
from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------- Configuration ----------
input_xyz    = "./pbe/total_mace.xyz"
output_root  = "."                                   # 创建子目录的根路径
target_sizes = [100, 200, 400]         # 需要的代表帧数量
soap_params  = {
    "r_cut":   6.0,
    "n_max":   8,
    "l_max":   6,
    "sigma":   0.5,
    "species": ["Li", "La", "Zr", "O","Ta","Nb"],
    "periodic": False,
    "sparse":   False,
}
sample_atoms_per_frame = 20                          # 每帧抽多少原子
neighbor_cutoff        = 4.0                         # "表面原子"配位半径
random_state           = 42

# ---------- Helper: pick low-coordination ("surface") atoms ----------
def get_surface_atoms(atoms, num_select=20, cutoff=4.0):
    """返回 num_select 个低配位原子下标；不足时全部返回。"""
    i_list, _ = neighbor_list("ij", atoms, cutoff)
    coord = np.bincount(i_list, minlength=len(atoms))
    order = np.argsort(coord)              # 配位数从低到高
    return order[:min(num_select, len(order))]

# ---------- Load structures ----------
print("📥 Loading XYZ frames …")
frames = read(input_xyz, index=":")
n_total = len(frames)
print(f"✅ Loaded {n_total} frames")

# ---------- Pre-compute SOAP on all frames ----------
print("🔍 Calculating SOAP descriptors (surface-prioritized) …")
soap = SOAP(**soap_params)
X, valid_frames = [], []
for atoms in tqdm(frames, desc="SOAP"):
    try:
        sel_idx = get_surface_atoms(atoms,
                                    num_select=sample_atoms_per_frame,
                                    cutoff=neighbor_cutoff)
        desc = soap.create(atoms, sel_idx)
        X.append(np.mean(desc, axis=0))
        valid_frames.append(atoms)
    except Exception as e:
        print(f"⚠️  Skip a frame: {e}")

X = np.asarray(X)
print(f"✅ SOAP done on {len(valid_frames)} frames")

# ---------- PCA once ----------
print("📊 Running PCA …")
X_scaled = StandardScaler().fit_transform(X)
X_pca    = PCA(n_components=2, random_state=random_state).fit_transform(X_scaled)

# ---------- Core routine for one target size ----------
def process_subset(n_select: int, subset, orig_indices):
    """对选出的代表帧子集执行完整的SOAP-PCA流程并分割数据集"""
    # 为子集创建SOAP描述符
    print(f"\n🔍 Calculating SOAP for {n_select}-frame subset...")
    X_sub = []
    for i, atoms in tqdm(enumerate(subset), total=len(subset), desc="Subset SOAP"):
        sel_idx = get_surface_atoms(atoms,
                                    num_select=sample_atoms_per_frame,
                                    cutoff=neighbor_cutoff)
        desc = soap.create(atoms, sel_idx)
        X_sub.append(np.mean(desc, axis=0))
    
    X_sub = np.asarray(X_sub)
    
    # 对子集执行PCA
    print(f"📊 Running PCA for {n_select}-frame subset...")
    X_sub_scaled = StandardScaler().fit_transform(X_sub)
    pca_sub = PCA(n_components=2, random_state=random_state)
    X_sub_pca = pca_sub.fit_transform(X_sub_scaled)
    
    # 分割训练集和测试集 (9:1)
    print(f"✂️ Splitting {n_select}-frame subset into train/test...")
    train_idx, test_idx = train_test_split(
        np.arange(len(subset)), 
        test_size=0.1, 
        random_state=random_state
    )
    
    train_set = [subset[i] for i in train_idx]
    test_set = [subset[i] for i in test_idx]
    
    # 获取原始索引用于绘图
    train_orig_idx = [orig_indices[i] for i in train_idx]
    test_orig_idx = [orig_indices[i] for i in test_idx]
    
    # 创建目录
    d = f"{n_select}-fra-20250611"
    os.makedirs(d, exist_ok=True)
    
    # 保存完整子集
    xyzf = os.path.join(d, f"soap_selected_{n_select}_frames.xyz")
    write(xyzf, subset)
    print(f"    ✅ Full subset XYZ -> {xyzf}")
    
    # 保存训练集
    train_xyzf = os.path.join(d, f"soap_selected_{n_select}_train.xyz")
    write(train_xyzf, train_set)
    print(f"    ✅ Train set XYZ   -> {train_xyzf} ({len(train_set)} frames)")
    
    # 保存测试集
    test_xyzf = os.path.join(d, f"soap_selected_{n_select}_test.xyz")
    write(test_xyzf, test_set)
    print(f"    ✅ Test set XYZ    -> {test_xyzf} ({len(test_set)} frames)")
    
    # 保存子集PCA图
    plt.figure(figsize=(8, 6))
    plt.scatter(X_sub_pca[:, 0], X_sub_pca[:, 1], s=20, alpha=0.7)
    plt.scatter(X_sub_pca[train_idx, 0], X_sub_pca[train_idx, 1], 
                s=40, c='blue', label=f'Train ({len(train_set)})')
    plt.scatter(X_sub_pca[test_idx, 0], X_sub_pca[test_idx, 1], 
                s=60, c='red', marker='x', label=f'Test ({len(test_set)})')
    plt.xlabel("Subset PCA 1")
    plt.ylabel("Subset PCA 2")
    plt.title(f"SOAP-PCA for {n_select}-frame Subset")
    plt.legend()
    plt.tight_layout()
    pngf = os.path.join(d, f"subset_pca_{n_select}.png")
    plt.savefig(pngf, dpi=300)
    plt.close()
    print(f"    📈 Subset PCA plot -> {pngf}")
    
    # 保存子集PCA数据
    datf = os.path.join(d, f"subset_pca_{n_select}.dat")
    with open(datf, "w") as f:
        f.write("# orig_idx  subset_idx  PC1  PC2  split(train/test)\n")
        for i, (orig_i, (x, y)) in enumerate(zip(orig_indices, X_sub_pca)):
            if i in train_idx:
                split = "train"
            else:
                split = "test"
            f.write(f"{orig_i} {i} {x:.6f} {y:.6f} {split}\n")
    print(f"    📄 Subset PCA data -> {datf}")
    
    # 在全局PCA图上标记子集分割
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=1, c='grey', alpha=0.3, label="All frames")
    plt.scatter(X_pca[orig_indices, 0], X_pca[orig_indices, 1], 
                s=20, c='green', alpha=0.7, label=f"Selected {n_select}")
    plt.scatter(X_pca[train_orig_idx, 0], X_pca[train_orig_idx, 1], 
                s=40, c='blue', label="Train")
    plt.scatter(X_pca[test_orig_idx, 0], X_pca[test_orig_idx, 1], 
                s=60, c='red', marker='x', label="Test")
    plt.xlabel("Global PCA 1")
    plt.ylabel("Global PCA 2")
    plt.title(f"Global PCA with {n_select}-frame Subset Split")
    plt.legend()
    plt.tight_layout()
    global_pngf = os.path.join(d, f"global_pca_{n_select}.png")
    plt.savefig(global_pngf, dpi=300)
    plt.close()
    print(f"    🌍 Global PCA plot -> {global_pngf}")

# ---------- Loop over required sizes ----------
for n in target_sizes:
    if len(valid_frames) < n:
        print(f"[WARN] only {len(valid_frames)} frames available; skip {n}.")
        continue
    
    print(f"\n[ {n} ] KMeans clustering …")
    kmeans = KMeans(n_clusters=n, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(X_pca)
    _, unique_idx = np.unique(labels, return_index=True)
    sel_idx = np.sort(unique_idx)
    subset = [valid_frames[i] for i in sel_idx]
    
    # 对选出的子集进行完整处理
    process_subset(n, subset, sel_idx)

print("\n🎉  All tasks finished.")

# %% ---- cell ----
#!/usr/bin/env python
# random_select_xyz_v2.py
import os
import numpy as np
from ase.io import read, write
from sklearn.model_selection import train_test_split

# ---------- 配置 ----------
input_xyz    = "./pbe/total_mace.xyz"
output_root  = "randselect"
target_sizes = [500, 1000, 2000, 4000]
test_ratio   = 0.1
random_state = 42

# ---------- 读入全部帧 ----------
frames   = read(input_xyz, index=":")
n_total  = len(frames)
print(f"✅ Loaded {n_total} frames")

# ---------- 主循环 ----------
rng = np.random.RandomState(random_state)
os.makedirs(output_root, exist_ok=True)

for n in target_sizes:
    if n_total < n:
        print(f"[WARN] 只有 {n_total} 帧，跳过 {n}。")
        continue

    sel_idx = rng.choice(n_total, size=n, replace=False)
    subset  = [frames[i] for i in sel_idx]

    train_idx, test_idx = train_test_split(
        np.arange(n), test_size=test_ratio, random_state=random_state
    )
    train_set = [subset[i] for i in train_idx]
    test_set  = [subset[i] for i in test_idx]

    train_size = len(train_set)
    test_size  = len(test_set)

    dir_path = os.path.join(output_root, f"{n}-randselect")
    os.makedirs(dir_path, exist_ok=True)

    write(os.path.join(dir_path, f"rand_selected_{n}_frames.xyz"), subset)
    write(os.path.join(dir_path, f"rand_selected_{train_size}_train.xyz"), train_set)
    write(os.path.join(dir_path, f"rand_selected_{test_size}_test.xyz"),  test_set)

    print(f"✅ {n}-frame dataset → {dir_path}  "
          f"(train {train_size}, test {test_size})")

print("\n🎉  All datasets generated.")

# %% ---- cell ----
#!/usr/bin/env python
# random_select_xyz_v2.py
import os
import numpy as np
from ase.io import read, write
from sklearn.model_selection import train_test_split

# ---------- 配置 ----------
input_xyz    = "./pbe/total_mace.xyz"
output_root  = "randselect"
target_sizes = [100, 200, 400]
test_ratio   = 0.1
random_state = 42

# ---------- 读入全部帧 ----------
frames   = read(input_xyz, index=":")
n_total  = len(frames)
print(f"✅ Loaded {n_total} frames")

# ---------- 主循环 ----------
rng = np.random.RandomState(random_state)
os.makedirs(output_root, exist_ok=True)

for n in target_sizes:
    if n_total < n:
        print(f"[WARN] 只有 {n_total} 帧，跳过 {n}。")
        continue

    sel_idx = rng.choice(n_total, size=n, replace=False)
    subset  = [frames[i] for i in sel_idx]

    train_idx, test_idx = train_test_split(
        np.arange(n), test_size=test_ratio, random_state=random_state
    )
    train_set = [subset[i] for i in train_idx]
    test_set  = [subset[i] for i in test_idx]

    train_size = len(train_set)
    test_size  = len(test_set)

    dir_path = os.path.join(output_root, f"{n}-randselect")
    os.makedirs(dir_path, exist_ok=True)

    write(os.path.join(dir_path, f"rand_selected_{n}_frames.xyz"), subset)
    write(os.path.join(dir_path, f"rand_selected_{train_size}_train.xyz"), train_set)
    write(os.path.join(dir_path, f"rand_selected_{test_size}_test.xyz"),  test_set)

    print(f"✅ {n}-frame dataset → {dir_path}  "
          f"(train {train_size}, test {test_size})")

print("\n🎉  All datasets generated.")

# %% ---- cell ----
#!/usr/bin/env python
# random_select_xyz_v2.py
import os
import numpy as np
from ase.io import read, write
from sklearn.model_selection import train_test_split

# ---------- 配置 ----------
input_xyz    = "./pbe/total_mace.xyz"
output_root  = "randselect/100-seed10086"
target_sizes = [100]
test_ratio   = 0.1
random_state = 10086

# ---------- 读入全部帧 ----------
frames   = read(input_xyz, index=":")
n_total  = len(frames)
print(f"✅ Loaded {n_total} frames")

# ---------- 主循环 ----------
rng = np.random.RandomState(random_state)
os.makedirs(output_root, exist_ok=True)

for n in target_sizes:
    if n_total < n:
        print(f"[WARN] 只有 {n_total} 帧，跳过 {n}。")
        continue

    sel_idx = rng.choice(n_total, size=n, replace=False)
    subset  = [frames[i] for i in sel_idx]

    train_idx, test_idx = train_test_split(
        np.arange(n), test_size=test_ratio, random_state=random_state
    )
    train_set = [subset[i] for i in train_idx]
    test_set  = [subset[i] for i in test_idx]

    train_size = len(train_set)
    test_size  = len(test_set)

    dir_path = os.path.join(output_root, f"{n}-randselect")
    os.makedirs(dir_path, exist_ok=True)

    write(os.path.join(dir_path, f"rand_selected_{n}_frames.xyz"), subset)
    write(os.path.join(dir_path, f"rand_selected_{train_size}_train.xyz"), train_set)
    write(os.path.join(dir_path, f"rand_selected_{test_size}_test.xyz"),  test_set)

    print(f"✅ {n}-frame dataset → {dir_path}  "
          f"(train {train_size}, test {test_size})")

print("\n🎉  All datasets generated.")

# %% ---- cell ----
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
soap_pca_sampling_timer.py — 带绝对时间戳与阶段耗时统计的 SOAP-PCA 采样脚本
"""

import os, time
from datetime import datetime
import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list
from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------- Configuration ----------
input_xyz    = "./pbe/total_mace.xyz"
output_root  = "./PCA-time/"
target_sizes = [100, 200, 400,500,1000,2000,4000]  # 需要的代表帧数量
soap_params  = {
    "r_cut":   6.0,
    "n_max":   8,
    "l_max":   6,
    "sigma":   0.5,
    "species": ["Li", "La", "Zr", "O", "Ta", "Nb"],
    "periodic": False,
    "sparse":   False,
}
sample_atoms_per_frame = 20
neighbor_cutoff        = 4.0
random_state           = 42

# ========== Helper ==========
def get_surface_atoms(atoms, num_select=20, cutoff=4.0):
    i_list, _ = neighbor_list("ij", atoms, cutoff)
    coord = np.bincount(i_list, minlength=len(atoms))
    return np.argsort(coord)[:min(num_select, len(atoms))]

# ========== Global timer ==========
global_start_dt = datetime.now()
global_start    = time.perf_counter()
print(f"⏰ Start: {global_start_dt:%Y-%m-%d %H:%M:%S}")

# ========== Load structures ==========
t0 = time.perf_counter()
print("📥 Loading XYZ frames …", end="", flush=True)
frames = read(input_xyz, index=":")
print(f"  ⏱ {time.perf_counter() - t0:.2f} s")
print(f"✅ Loaded {len(frames)} frames")

# ========== Pre-compute SOAP ==========
t0 = time.perf_counter()
print("🔍 Calculating SOAP descriptors (surface-prioritized) …", end="", flush=True)
soap = SOAP(**soap_params)
X, valid_frames = [], []
for atoms in tqdm(frames, desc="SOAP"):
    try:
        sel_idx = get_surface_atoms(atoms, sample_atoms_per_frame, neighbor_cutoff)
        X.append(np.mean(soap.create(atoms, sel_idx), axis=0))
        valid_frames.append(atoms)
    except Exception as e:
        print(f"\n⚠️  Skip a frame: {e}")
print(f"\n⏱ {time.perf_counter() - t0:.2f} s")
X = np.asarray(X)
print(f"✅ SOAP done on {len(valid_frames)} frames")

pre_end_dt = datetime.now()
print(f"⏰ End Preprocess: {pre_end_dt:%Y-%m-%d %H:%M:%S}  |  ⏳ Elapsed: {pre_end_dt.timestamp() - global_start_dt.timestamp():.2f} s")

# ========== Global PCA ==========
t0 = time.perf_counter()
print("📊 Running global PCA …", end="", flush=True)
X_pca = PCA(2, random_state=random_state).fit_transform(StandardScaler().fit_transform(X))
print(f"  ⏱ {time.perf_counter() - t0:.2f} s")

# ========== Subset pipeline ==========
def process_subset(n_select, subset, orig_indices):
    subset_start_dt = datetime.now()
    subset_start_pc = time.perf_counter()
    print(f"\n⏰ Subset-{n_select} start: {subset_start_dt:%H:%M:%S}")

    # 1) SOAP for subset
    t0 = time.perf_counter()
    print(f"[ {n_select} ] KMeans clustering …", end="", flush=True)
    print(f"  ⏱ {t0 - subset_start_pc:.2f} s (should be ~0)")
    t1 = time.perf_counter()
    print("🔍 Calculating SOAP for "
          f"{n_select}-frame subset …", end="", flush=True)
    X_sub = [np.mean(soap.create(at, get_surface_atoms(at,
                sample_atoms_per_frame, neighbor_cutoff)), axis=0)
             for at in tqdm(subset, desc="Subset SOAP")]
    print(f"  ⏱ {time.perf_counter() - t1:.2f} s")

    # 2) PCA
    t2 = time.perf_counter()
    print(f"📊 Running PCA for {n_select}-frame subset …", end="", flush=True)
    X_sub_pca = PCA(2, random_state=random_state).fit_transform(
        StandardScaler().fit_transform(np.asarray(X_sub)))
    print(f"  ⏱ {time.perf_counter() - t2:.2f} s")

    # 3) Train/test split
    t3 = time.perf_counter()
    print(f"✂️ Splitting {n_select}-frame subset into train/test …", end="", flush=True)
    train_idx, test_idx = train_test_split(
        np.arange(len(subset)), test_size=0.1, random_state=random_state)
    print(f"  ⏱ {time.perf_counter() - t3:.2f} s")

    # ——（后续文件保存与画图同原脚本，可按需要保留/删除）——

    subset_end_dt = datetime.now()
    print(f"⏰ Subset-{n_select} end  : {subset_end_dt:%H:%M:%S}  |  "
          f"⏳ Elapsed: {time.perf_counter() - subset_start_pc:.2f} s")

# ========== Loop over required sizes ==========
for n in target_sizes:
    if len(valid_frames) < n:
        print(f"[WARN] only {len(valid_frames)} frames available; skip {n}.")
        continue
    # KMeans 选帧
    km = KMeans(n_clusters=n, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X_pca)
    subset_idx = np.sort(np.unique(labels, return_index=True)[1])
    subset_atoms = [valid_frames[i] for i in subset_idx]
    process_subset(n, subset_atoms, subset_idx)

# ========== Finish ==========
global_end_dt = datetime.now()
print("\n🎉  All tasks finished.")
print(f"🕛 Finish: {global_end_dt:%Y-%m-%d %H:%M:%S}  |  "
      f"⏳ Total elapsed: {time.perf_counter() - global_start:.2f} s")


# %% ---- cell ----
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
soap_pca_sampling_v2.py

V2 improvements:
1) select nearest-to-centroid frame per cluster (no first-index bias)
2) cluster in higher-dimensional feature space (PCA for plotting only)
3) mixed atom sampling per frame (low-coord + random + high-coord)
4) minimum quota constraints (element / coordination bin / composition)
"""

import os
import time
from collections import Counter, defaultdict

import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list
from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------- Configuration ----------
input_xyz = "./pbe/total_mace.xyz"
output_root = "./PCA-v2"
target_sizes = [100, 200, 400, 500, 1000, 2000, 4000]

soap_params = {
    "r_cut": 6.0,
    "n_max": 8,
    "l_max": 6,
    "sigma": 0.5,
    "species": ["Li", "La", "Zr", "O", "Ta", "Nb"],
    "periodic": False,
    "sparse": False,
}

# Mixed sampling config: low + random + high coordination atoms
sample_atoms_per_frame = 24
mix_low_ratio = 0.4
mix_rand_ratio = 0.2
mix_high_ratio = 0.4
neighbor_cutoff = 4.0

# Clustering space config
cluster_dim = 32            # KMeans works in this dim (not 2D plot space)
random_state = 42

# Quota config
quota_element_frac = 0.005  # each element minimum quota = max(1, round(frac * n_select))
quota_coord_frac = 0.005    # each coord-bin minimum quota
max_comp_quota = 10         # number of rare compositions to enforce (1 each)


# ---------- Helpers ----------
def frame_composition_key(atoms):
    counts = Counter(atoms.get_chemical_symbols())
    return tuple(sorted(counts.items()))


def mixed_atom_indices(atoms, num_select=24, cutoff=4.0, seed=42):
    """Select atoms using low-coord + random + high-coord mix."""
    i_list, _ = neighbor_list("ij", atoms, cutoff)
    coord = np.bincount(i_list, minlength=len(atoms))

    order = np.argsort(coord)
    n_atoms = len(order)
    n_take = min(num_select, n_atoms)

    k_low = int(round(n_take * mix_low_ratio))
    k_rand = int(round(n_take * mix_rand_ratio))
    k_high = n_take - k_low - k_rand

    low_idx = order[:k_low]
    high_idx = order[max(0, n_atoms - k_high):] if k_high > 0 else np.array([], dtype=int)

    chosen = set(low_idx.tolist()) | set(high_idx.tolist())
    remain = [int(i) for i in order if int(i) not in chosen]

    if k_rand > 0 and remain:
        rng = np.random.default_rng(seed)
        rand_idx = rng.choice(remain, size=min(k_rand, len(remain)), replace=False)
        chosen.update(int(i) for i in rand_idx)

    # pad if rounding/overlap leaves too few
    if len(chosen) < n_take:
        for i in order:
            chosen.add(int(i))
            if len(chosen) >= n_take:
                break

    idx = np.array(sorted(chosen), dtype=int)
    return idx, coord


def nearest_center_indices(X, labels, centers):
    """One representative per cluster: nearest sample to centroid."""
    reps = []
    n_clusters = centers.shape[0]
    for c in range(n_clusters):
        members = np.where(labels == c)[0]
        if members.size == 0:
            continue
        d2 = np.sum((X[members] - centers[c]) ** 2, axis=1)
        reps.append(int(members[np.argmin(d2)]))
    # unique while preserving order
    out, seen = [], set()
    for i in reps:
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


def choose_diverse_from_pool(pool_idx, selected_set, min_dist):
    cands = [i for i in pool_idx if i not in selected_set]
    if not cands:
        return None
    cands = np.asarray(cands, dtype=int)
    return int(cands[np.argmax(min_dist[cands])])


def update_min_dist(X, min_dist, new_idx):
    d2 = np.sum((X - X[new_idx]) ** 2, axis=1)
    np.minimum(min_dist, d2, out=min_dist)


# ---------- Load data ----------
os.makedirs(output_root, exist_ok=True)
print("Loading XYZ frames ...")
frames = read(input_xyz, index=":")
n_total = len(frames)
print(f"Loaded {n_total} frames")

# ---------- Build frame descriptors + metadata ----------
print("Computing SOAP descriptors with mixed atom sampling ...")
soap = SOAP(**soap_params)
X = []
frame_elements = []
frame_comp = []
coord_means = []
valid_frames = []

for i, atoms in tqdm(enumerate(frames), total=n_total, desc="SOAP"):
    try:
        sel_idx, coord = mixed_atom_indices(
            atoms,
            num_select=sample_atoms_per_frame,
            cutoff=neighbor_cutoff,
            seed=random_state + i,
        )
        desc = soap.create(atoms, sel_idx)
        X.append(np.mean(desc, axis=0))
        valid_frames.append(atoms)

        syms = atoms.get_chemical_symbols()
        frame_elements.append(set(syms))
        frame_comp.append(frame_composition_key(atoms))
        coord_means.append(float(np.mean(coord[sel_idx])) if len(sel_idx) > 0 else 0.0)
    except Exception as e:
        print(f"Skip frame {i}: {e}")

X = np.asarray(X, dtype=np.float32)
coord_means = np.asarray(coord_means, dtype=np.float32)
n_valid = len(valid_frames)
print(f"SOAP done on {n_valid} frames")

# ---------- Build clustering and plotting spaces ----------
print("Building feature spaces ...")
X_scaled = StandardScaler().fit_transform(X)

k_dim = min(cluster_dim, X_scaled.shape[1], max(2, n_valid - 1))
if k_dim < X_scaled.shape[1]:
    X_cluster = PCA(n_components=k_dim, random_state=random_state).fit_transform(X_scaled)
else:
    X_cluster = X_scaled

X_plot = PCA(n_components=2, random_state=random_state).fit_transform(X_scaled)

# ---------- Metadata indices for quota ----------
# Coord bins from global quantiles (3 bins)
q1, q2 = np.quantile(coord_means, [0.33, 0.66])
coord_bins = np.digitize(coord_means, [q1, q2], right=False)  # 0/1/2

all_elements = sorted(set().union(*frame_elements))

element_to_idx = defaultdict(list)
for i, elems in enumerate(frame_elements):
    for e in elems:
        element_to_idx[e].append(i)

coord_to_idx = defaultdict(list)
for i, b in enumerate(coord_bins):
    coord_to_idx[int(b)].append(i)

comp_counts = Counter(frame_comp)
comp_to_idx = defaultdict(list)
for i, c in enumerate(frame_comp):
    comp_to_idx[c].append(i)
rare_comps = [c for c, _ in sorted(comp_counts.items(), key=lambda kv: kv[1])]


def select_subset_indices(n_select):
    """KMeans + nearest center + quota-constrained top-up + diversity fill."""
    if n_select > n_valid:
        raise ValueError(f"n_select={n_select} > n_valid={n_valid}")

    km = MiniBatchKMeans(
        n_clusters=n_select,
        random_state=random_state,
        batch_size=min(4096, max(256, n_select * 4)),
        n_init=5,
    )
    labels = km.fit_predict(X_cluster)
    centers = km.cluster_centers_

    reps = nearest_center_indices(X_cluster, labels, centers)

    selected = []
    selected_set = set()
    mandatory = set()
    min_dist = np.full(n_valid, np.inf, dtype=np.float64)

    def add_idx(i, is_mandatory=False):
        if i in selected_set:
            return False
        selected.append(i)
        selected_set.add(i)
        update_min_dist(X_cluster, min_dist, i)
        if is_mandatory:
            mandatory.add(i)
        return True

    # ---- Quota targets ----
    elem_min = max(1, int(round(quota_element_frac * n_select)))
    coord_min = max(1, int(round(quota_coord_frac * n_select)))
    n_comp_quota = min(max(2, n_select // 200), max_comp_quota, len(rare_comps))

    # ---- Element quota (rare element first) ----
    elem_order = sorted(all_elements, key=lambda e: len(element_to_idx[e]))
    for e in elem_order:
        need = min(elem_min, len(element_to_idx[e]))
        while len(selected) < n_select:
            cur = sum(1 for i in selected if e in frame_elements[i])
            if cur >= need:
                break
            cand = choose_diverse_from_pool(element_to_idx[e], selected_set, min_dist)
            if cand is None:
                break
            add_idx(cand, is_mandatory=True)

    # ---- Coordination-bin quota ----
    for b in [0, 1, 2]:
        pool = coord_to_idx.get(b, [])
        need = min(coord_min, len(pool))
        while len(selected) < n_select:
            cur = sum(1 for i in selected if coord_bins[i] == b)
            if cur >= need:
                break
            cand = choose_diverse_from_pool(pool, selected_set, min_dist)
            if cand is None:
                break
            add_idx(cand, is_mandatory=True)

    # ---- Rare composition quota (1 each) ----
    for comp in rare_comps[:n_comp_quota]:
        if len(selected) >= n_select:
            break
        pool = comp_to_idx[comp]
        cand = choose_diverse_from_pool(pool, selected_set, min_dist)
        if cand is not None:
            add_idx(cand, is_mandatory=True)

    # ---- Fill from nearest-center reps ----
    for i in reps:
        if len(selected) >= n_select:
            break
        add_idx(i, is_mandatory=False)

    # ---- Diversity fill to exact n_select ----
    while len(selected) < n_select:
        mask = np.ones(n_valid, dtype=bool)
        if selected:
            mask[np.array(list(selected_set), dtype=int)] = False
        cand = int(np.argmax(np.where(mask, min_dist, -1.0)))
        if cand in selected_set:
            # fallback safety
            remain = np.where(mask)[0]
            if remain.size == 0:
                break
            cand = int(remain[0])
        add_idx(cand, is_mandatory=False)

    # safety exact size
    selected = selected[:n_select]
    selected = np.array(sorted(set(selected)), dtype=int)
    if selected.size < n_select:
        # fill deterministically if dedup happened
        remain = [i for i in range(n_valid) if i not in set(selected.tolist())]
        need = n_select - selected.size
        selected = np.concatenate([selected, np.array(remain[:need], dtype=int)])

    return np.array(sorted(selected), dtype=int)


# ---------- Main loop ----------
for n in target_sizes:
    if n > n_valid:
        print(f"[WARN] n={n} > n_valid={n_valid}, skip")
        continue

    t0 = time.perf_counter()
    print(f"\n[{n}] selecting subset ...")
    sel_idx = select_subset_indices(n)
    subset_atoms = [valid_frames[i] for i in sel_idx]

    # split train/test (size consistency kept)
    strat = coord_bins[sel_idx]
    use_strat = (len(np.unique(strat)) > 1) and np.all(np.bincount(strat) >= 2)
    train_i, test_i = train_test_split(
        np.arange(len(subset_atoms)),
        test_size=0.1,
        random_state=random_state,
        stratify=strat if use_strat else None,
    )

    train_atoms = [subset_atoms[i] for i in train_i]
    test_atoms = [subset_atoms[i] for i in test_i]

    # outputs
    d = os.path.join(output_root, f"{n}-fra-20250611-v2")
    os.makedirs(d, exist_ok=True)

    write(os.path.join(d, f"soap_selected_{n}_frames.xyz"), subset_atoms)
    write(os.path.join(d, f"soap_selected_{n}_train.xyz"), train_atoms)
    write(os.path.join(d, f"soap_selected_{n}_test.xyz"), test_atoms)

    # subset PCA plot
    X_sub_plot = X_plot[sel_idx]
    plt.figure(figsize=(8, 6))
    plt.scatter(X_sub_plot[:, 0], X_sub_plot[:, 1], s=20, alpha=0.6)
    plt.scatter(X_sub_plot[train_i, 0], X_sub_plot[train_i, 1], s=40, c="blue", label=f"Train ({len(train_atoms)})")
    plt.scatter(X_sub_plot[test_i, 0], X_sub_plot[test_i, 1], s=60, c="red", marker="x", label=f"Test ({len(test_atoms)})")
    plt.title(f"V2 SOAP-PCA for {n}-frame subset")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(d, f"subset_pca_{n}.png"), dpi=300)
    plt.close()

    # global PCA overlay
    plt.figure(figsize=(8, 6))
    plt.scatter(X_plot[:, 0], X_plot[:, 1], s=1, c="grey", alpha=0.25, label="All")
    plt.scatter(X_plot[sel_idx, 0], X_plot[sel_idx, 1], s=20, c="green", alpha=0.7, label=f"Selected {n}")
    plt.scatter(X_plot[sel_idx[train_i], 0], X_plot[sel_idx[train_i], 1], s=40, c="blue", label="Train")
    plt.scatter(X_plot[sel_idx[test_i], 0], X_plot[sel_idx[test_i], 1], s=60, c="red", marker="x", label="Test")
    plt.title(f"V2 Global PCA with {n}-frame subset")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(d, f"global_pca_{n}.png"), dpi=300)
    plt.close()

    # data log
    with open(os.path.join(d, f"subset_pca_{n}.dat"), "w", encoding="utf-8") as f:
        f.write("# orig_idx subset_idx PC1 PC2 split coord_bin\n")
        for i_sub, i_global in enumerate(sel_idx):
            split = "train" if i_sub in set(train_i.tolist()) else "test"
            x, y = X_plot[i_global]
            f.write(f"{i_global} {i_sub} {x:.6f} {y:.6f} {split} {int(coord_bins[i_global])}\n")

    # quota summary
    sel_elems = Counter()
    for i in sel_idx:
        for e in frame_elements[i]:
            sel_elems[e] += 1
    sel_bins = Counter(coord_bins[sel_idx].tolist())
    sel_comp = Counter([frame_comp[i] for i in sel_idx])

    with open(os.path.join(d, f"quota_report_{n}.txt"), "w", encoding="utf-8") as f:
        f.write(f"n_selected: {len(sel_idx)}\n")
        f.write("element_counts:\n")
        for e in sorted(sel_elems):
            f.write(f"  {e}: {sel_elems[e]}\n")
        f.write("coord_bin_counts:\n")
        for b in [0, 1, 2]:
            f.write(f"  bin{b}: {sel_bins.get(b, 0)}\n")
        f.write("rare_composition_examples:\n")
        for comp, c in sel_comp.most_common(10):
            f.write(f"  {comp}: {c}\n")

    print(f"  saved -> {d}")
    print(f"  elapsed: {time.perf_counter() - t0:.2f} s")

print("\nV2 pipeline finished.")


# %% ---- cell ----
