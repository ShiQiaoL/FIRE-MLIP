# Auto-exported from 数据转换和PCA分析.ipynb (code cells only; paths are the authors' original ones)

import os
import dpdata as dp
from dpdata.plugins.ase import ASEStructureFormat
from ase.io import write

# ======================== 变量定义区 ========================
# 根目录：脚本会递归搜索此目录及其所有子目录
root_dir = "./HECN"

# 原子类型映射：如果 type.raw 或 npy 中是整型 (0,1,2,3)，
# 则需要在此列表中按顺序指定对应的元素符号。
my_type_map = ["Ti", "Zr", "Ta", "Hf", "Nb", "C", "N"] 
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
import os
import dpdata as dp
from dpdata.plugins.ase import ASEStructureFormat
from ase.io import write

# ======================== 变量定义区 ========================
# 根目录：脚本会递归搜索此目录及其所有子目录
root_dir = "./LixFePO4"

# 原子类型映射：如果 type.raw 或 npy 中是整型 (0,1,2,3)，
# 则需要在此列表中按顺序指定对应的元素符号。
my_type_map = ["Fe", "Li", "O", "P"]
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
import os
import dpdata as dp
from dpdata.plugins.ase import ASEStructureFormat
from ase.io import write

# ======================== 变量定义区 ========================
# 根目录：脚本会递归搜索此目录及其所有子目录
root_dir = "./ZnOGaN"

# 原子类型映射：如果 type.raw 或 npy 中是整型 (0,1,2,3)，
# 则需要在此列表中按顺序指定对应的元素符号。
my_type_map = ["Ga", "N", "O", "Zn"]

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
        ls = dp.LabeledSystem(dirpath, fmt="deepmd/raw", type_map=my_type_map)
        
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
from ase.io import read, write
from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== Configuration ==========
input_xyz = "./HECN/total_mace.xyz"
output_xyz = "./HECN/soap_selected_1000_frames.xyz"
output_plot = "./HECN/soap_pca_1000.png"
n_selected_frames = 1000
soap_params = {
    "r_cut": 6.0,
    "n_max": 8,
    "l_max": 6,
    "sigma": 0.5,
    "species":["Ti", "Zr", "Ta", "Hf", "Nb", "C", "N"] ,  # adjust to match your data
    "periodic": False,
    "sparse": False
}
sample_atoms_per_frame = 20

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
plt.scatter(X_pca[selected_idxs, 0], X_pca[selected_idxs, 1], s=10, c="red", label="Selected 1000")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("SOAP PCA of Local Atomic Environments")
plt.legend()
plt.tight_layout()
plt.savefig(output_plot, dpi=300)
print(f"📊 PCA plot saved to {output_plot}")

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
input_xyz = "./HECN/soap_selected_1000_frames.xyz"
train_output = "./HECN/soap_train_900.xyz"
test_output = "./HECN/soap_test_100.xyz"
plot_output = "./HECN/soap_pca_split_train_test.png"
soap_params = {
    "r_cut": 6.0,
    "n_max": 8,
    "l_max": 6,
    "sigma": 0.5,
    "species": ["Ti", "Zr", "Ta", "Hf", "Nb", "C", "N"] ,
    "periodic": False,
    "sparse": False
}
sample_atoms_per_frame = 20
n_test = 100

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
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=1, c='gray', alpha=0.3, label="All 1000")
plt.scatter(X_pca[test_indices, 0], X_pca[test_indices, 1], s=10, c='blue', label="Test 100")
plt.scatter(X_pca[train_indices, 0], X_pca[train_indices, 1], s=10, c='red', label="Train 900")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("SOAP PCA split: Train vs Test")
plt.legend()
plt.tight_layout()
plt.savefig(plot_output, dpi=300)
print(f"📊 PCA plot saved to {plot_output}")


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
input_xyz = "./LixFePO4/total_mace.xyz"
output_xyz = "./LixFePO4/soap_selected_1000_frames.xyz"
output_plot = "./LixFePO4/soap_pca_1000.png"
n_selected_frames = 1000
soap_params = {
    "r_cut": 6.0,
    "n_max": 8,
    "l_max": 6,
    "sigma": 0.5,
    "species":["Fe", "Li", "O", "P"] ,  # adjust to match your data
    "periodic": False,
    "sparse": False
}
sample_atoms_per_frame = 20

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
plt.scatter(X_pca[selected_idxs, 0], X_pca[selected_idxs, 1], s=10, c="red", label="Selected 1000")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("SOAP PCA of Local Atomic Environments")
plt.legend()
plt.tight_layout()
plt.savefig(output_plot, dpi=300)
print(f"📊 PCA plot saved to {output_plot}")

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
input_xyz = "./LixFePO4//soap_selected_1000_frames.xyz"
train_output = "./LixFePO4/soap_train_900.xyz"
test_output = "./LixFePO4//soap_test_100.xyz"
plot_output = "./LixFePO4//soap_pca_split_train_test.png"
soap_params = {
    "r_cut": 6.0,
    "n_max": 8,
    "l_max": 6,
    "sigma": 0.5,
    "species": ["Fe", "Li", "O", "P"] ,
    "periodic": False,
    "sparse": False
}
sample_atoms_per_frame = 20
n_test = 100

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
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=1, c='gray', alpha=0.3, label="All 1000")
plt.scatter(X_pca[test_indices, 0], X_pca[test_indices, 1], s=10, c='blue', label="Test 100")
plt.scatter(X_pca[train_indices, 0], X_pca[train_indices, 1], s=10, c='red', label="Train 900")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("SOAP PCA split: Train vs Test")
plt.legend()
plt.tight_layout()
plt.savefig(plot_output, dpi=300)
print(f"📊 PCA plot saved to {plot_output}")


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
input_xyz = "./ZnOGaN/total_mace.xyz"
output_xyz = "./ZnOGaN/soap_selected_1000_frames.xyz"
output_plot = "./ZnOGaN/soap_pca_1000.png"
n_selected_frames = 1000
soap_params = {
    "r_cut": 6.0,
    "n_max": 8,
    "l_max": 6,
    "sigma": 0.5,
    "species":["Ga", "N", "O", "Zn"],  # adjust to match your data
    "periodic": False,
    "sparse": False
}
sample_atoms_per_frame = 20

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
plt.scatter(X_pca[selected_idxs, 0], X_pca[selected_idxs, 1], s=10, c="red", label="Selected 1000")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("SOAP PCA of Local Atomic Environments")
plt.legend()
plt.tight_layout()
plt.savefig(output_plot, dpi=300)
print(f"📊 PCA plot saved to {output_plot}")

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
input_xyz = "./ZnOGaN/soap_selected_1000_frames.xyz"
train_output = "./ZnOGaN/soap_train_900.xyz"
test_output = "./ZnOGaN/soap_test_100.xyz"
plot_output = "./ZnOGaN/soap_pca_split_train_test.png"
soap_params = {
    "r_cut": 6.0,
    "n_max": 8,
    "l_max": 6,
    "sigma": 0.5,
    "species": ["Ga", "N", "O", "Zn"] ,
    "periodic": False,
    "sparse": False
}
sample_atoms_per_frame = 20
n_test = 100

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
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=1, c='gray', alpha=0.3, label="All 1000")
plt.scatter(X_pca[test_indices, 0], X_pca[test_indices, 1], s=10, c='blue', label="Test 100")
plt.scatter(X_pca[train_indices, 0], X_pca[train_indices, 1], s=10, c='red', label="Train 900")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("SOAP PCA split: Train vs Test")
plt.legend()
plt.tight_layout()
plt.savefig(plot_output, dpi=300)
print(f"📊 PCA plot saved to {plot_output}")

# %% ---- cell ----
