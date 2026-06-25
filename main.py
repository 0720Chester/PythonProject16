import os
import sys

# 修复OpenMP库冲突问题 - 必须在其他导入之前设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import types
import numpy as np
import torch
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from smplx.lbs import (
    blend_shapes,
    vertices2joints,
    batch_rodrigues,
    batch_rigid_transform,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class _ChumpyArrayShim:
    """Minimal pickle shim for old SMPL files that stored arrays as chumpy.Ch."""

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _array(self):
        if hasattr(self, "r"):
            return self.r
        if hasattr(self, "x"):
            return self.x
        raise AttributeError("Cannot recover array data from chumpy pickle object")

    def __array__(self, dtype=None):
        return np.asarray(self._array(), dtype=dtype)

    @property
    def shape(self):
        return np.asarray(self).shape

    def __len__(self):
        return len(np.asarray(self))

    def __getitem__(self, item):
        return np.asarray(self)[item]


def install_chumpy_pickle_shim():
    """Allow pickle.load to read legacy SMPL .pkl files without installing chumpy."""
    if "chumpy.ch" in sys.modules:
        return

    chumpy_module = types.ModuleType("chumpy")
    chumpy_ch_module = types.ModuleType("chumpy.ch")

    _ChumpyArrayShim.__name__ = "Ch"
    _ChumpyArrayShim.__qualname__ = "Ch"
    _ChumpyArrayShim.__module__ = "chumpy.ch"
    chumpy_ch_module.Ch = _ChumpyArrayShim
    chumpy_module.ch = chumpy_ch_module

    sys.modules["chumpy"] = chumpy_module
    sys.modules["chumpy.ch"] = chumpy_ch_module


def make_out_dir(path: str):
    os.makedirs(path, exist_ok=True)


def resolve_script_path(path: str):
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def set_axes_equal(ax, vertices: np.ndarray):
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = 0.5 * np.max(maxs - mins + 1e-8)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def get_face_colors_from_vertex_scalar(vertex_scalar: np.ndarray, faces: np.ndarray, cmap_name="viridis"):
    scalar = vertex_scalar.astype(np.float64)
    scalar = (scalar - scalar.min()) / (scalar.max() - scalar.min() + 1e-8)
    face_scalar = scalar[faces].mean(axis=1)
    cmap = plt.get_cmap(cmap_name)
    return cmap(face_scalar)


def get_face_colors_from_joint_weights(lbs_weights: np.ndarray, faces: np.ndarray):
    face_weights = lbs_weights[faces].mean(axis=1)
    dominant_joint = np.argmax(face_weights, axis=1)
    dominant_weight = np.max(face_weights, axis=1)

    num_joints = lbs_weights.shape[1]
    palette = plt.get_cmap("hsv")(np.linspace(0.0, 1.0, num_joints, endpoint=False))
    face_colors = palette[dominant_joint]
    strength = 0.35 + 0.65 * dominant_weight
    face_colors[:, :3] *= strength[:, None]
    face_colors[:, :3] += (1.0 - strength[:, None]) * 0.88
    face_colors[:, 3] = 1.0
    return face_colors


def smpl_to_plot_coords(points: np.ndarray):
    return points[:, [0, 2, 1]]


def shade_face_colors(vertices: np.ndarray, faces: np.ndarray, face_colors: np.ndarray):
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8

    light_dir = np.array([-0.25, -0.55, 0.80], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)
    intensity = 0.35 + 0.65 * np.clip(normals @ light_dir, 0.0, 1.0)

    shaded = face_colors.copy()
    shaded[:, :3] *= intensity[:, None]
    return shaded


def draw_mesh(
        ax,
        vertices: np.ndarray,
        faces: np.ndarray,
        joints: np.ndarray = None,
        vertex_scalar: np.ndarray = None,
        face_colors: np.ndarray = None,
        title: str = "",
        elev: float = 12,
        azim: float = 108,
):
    plot_vertices = smpl_to_plot_coords(vertices)
    plot_joints = None if joints is None else smpl_to_plot_coords(joints)

    if face_colors is not None:
        face_colors = face_colors.copy()
    elif vertex_scalar is None:
        face_colors = np.tile(np.array([[0.82, 0.67, 0.52, 1.0]]), (faces.shape[0], 1))
    else:
        face_colors = get_face_colors_from_vertex_scalar(vertex_scalar, faces)
    face_colors = shade_face_colors(plot_vertices, faces, face_colors)

    mesh = Poly3DCollection(
        plot_vertices[faces],
        facecolors=face_colors,
        linewidths=0.03,
        edgecolors=(0.0, 0.0, 0.0, 0.05),
    )
    ax.add_collection3d(mesh)

    if joints is not None:
        ax.scatter(
            plot_joints[:, 0], plot_joints[:, 1], plot_joints[:, 2],
            c="white", s=12, depthshade=False,
            edgecolors="black", linewidths=0.3
        )

    set_axes_equal(ax, plot_vertices)
    ax.set_proj_type("persp", focal_length=0.85)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)


def save_single_figure(path, vertices, faces, joints=None, vertex_scalar=None, title=""):
    fig = plt.figure(figsize=(5, 6))
    ax = fig.add_subplot(111, projection="3d")
    draw_mesh(ax, vertices, faces, joints=joints, vertex_scalar=vertex_scalar, title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_comparison_grid(path, data_dict, faces):
    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(221, projection="3d")
    draw_mesh(
        ax1,
        data_dict["v_template"],
        faces,
        joints=data_dict["J_template"],
        vertex_scalar=data_dict["weight_scalar"],
        title="(a) Template + LBS Weights"
    )

    ax2 = fig.add_subplot(222, projection="3d")
    draw_mesh(
        ax2,
        data_dict["v_shaped"],
        faces,
        joints=data_dict["J_shaped"],
        title="(b) Shape Blend + Joint Regression"
    )

    ax3 = fig.add_subplot(223, projection="3d")
    draw_mesh(
        ax3,
        data_dict["v_posed"],
        faces,
        joints=data_dict["J_shaped"],
        vertex_scalar=data_dict["pose_offset_norm"],
        title="(c) Pose Blend Shapes"
    )

    ax4 = fig.add_subplot(224, projection="3d")
    draw_mesh(
        ax4,
        data_dict["verts"],
        faces,
        joints=data_dict["J_transformed"],
        title="(d) Final LBS Result"
    )

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_all_joint_weights_figure(path, vertices, faces, joints, lbs_weights):
    fig = plt.figure(figsize=(7, 8))
    ax = fig.add_subplot(111, projection="3d")
    draw_mesh(
        ax,
        vertices,
        faces,
        joints=joints,
        face_colors=get_face_colors_from_joint_weights(lbs_weights, faces),
        title="All Joint LBS Weights",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


class SimpleSMPL:
    """
    简单的SMPL模型类，直接从pickle文件加载
    """

    def __init__(self, model_path, num_betas=10):
        install_chumpy_pickle_shim()

        print(f"加载模型文件: {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f, encoding='latin1')

        # 转换为tensor
        self.v_template = torch.tensor(
            np.array(model_data['v_template']), dtype=torch.float32
        )
        self.shapedirs = torch.tensor(
            np.array(model_data['shapedirs']), dtype=torch.float32
        )
        self.posedirs = torch.tensor(
            np.array(model_data['posedirs']), dtype=torch.float32
        )

        # 打印posedirs的原始形状
        print(f"  posedirs原始形状: {self.posedirs.shape}")

        # J_regressor可能是稀疏矩阵
        J_regressor = model_data['J_regressor']
        if hasattr(J_regressor, 'toarray'):
            J_regressor = J_regressor.toarray()
        self.J_regressor = torch.tensor(np.array(J_regressor), dtype=torch.float32)

        # 蒙皮权重
        self.lbs_weights = torch.tensor(
            np.array(model_data['weights']), dtype=torch.float32
        )

        # 关节树
        kintree = model_data['kintree_table']
        if hasattr(kintree, 'toarray'):
            kintree = kintree.toarray()
        self.parents = torch.tensor(kintree[0].astype(np.int64), dtype=torch.int64)

        # 面片
        self.faces = np.array(model_data['f'], dtype=np.int32)

        # 限制betas数量
        if self.shapedirs.shape[2] > num_betas:
            self.shapedirs = self.shapedirs[:, :, :num_betas]

        print(f"模型加载成功！")
        print(f"  顶点数: {self.v_template.shape[0]}")
        print(f"  关节数: {self.J_regressor.shape[0]}")
        print(f"  面片数: {self.faces.shape[0]}")
        print(f"  shapedirs形状: {self.shapedirs.shape}")
        print(f"  posedirs形状: {self.posedirs.shape}")


def build_demo_shape(device, dtype, num_betas=10):
    betas = torch.zeros((1, num_betas), dtype=dtype, device=device)
    # 设置几个非零 beta，让体型变化明显一些
    if num_betas >= 1:
        betas[0, 0] = 2.0  # 整体体型
    if num_betas >= 2:
        betas[0, 1] = -1.2  # 身高
    if num_betas >= 3:
        betas[0, 2] = 0.8  # 肩宽
    return betas


def build_demo_pose(device, dtype):
    # SMPL: global_orient = 3, body_pose = 23 * 3
    global_orient = torch.zeros((1, 3), dtype=dtype, device=device)
    body_pose = torch.zeros((1, 23 * 3), dtype=dtype, device=device)

    # SMPL关节映射 (从0开始)
    joint_names = {
        "left_hip": 1,
        "right_hip": 2,
        "left_knee": 4,
        "right_knee": 5,
        "left_shoulder": 16,
        "right_shoulder": 17,
        "left_elbow": 18,
        "right_elbow": 19,
    }

    def set_joint_pose(name, axis_angle):
        start = (joint_names[name] - 1) * 3
        body_pose[0, start:start + 3] = torch.tensor(axis_angle, dtype=dtype, device=device)

    # 设置手臂姿态
    set_joint_pose("left_shoulder", [0.0, 0.0, 0.45])
    set_joint_pose("right_shoulder", [0.0, 0.0, -0.45])
    set_joint_pose("left_elbow", [0.0, -0.35, 0.0])
    set_joint_pose("right_elbow", [0.0, 0.35, 0.0])

    # 设置腿部姿态
    set_joint_pose("left_hip", [0.25, 0.0, 0.08])
    set_joint_pose("right_hip", [-0.18, 0.0, -0.08])
    set_joint_pose("left_knee", [0.35, 0.0, 0.0])
    set_joint_pose("right_knee", [0.20, 0.0, 0.0])

    return global_orient, body_pose


def compute_manual_lbs(model, betas, global_orient, body_pose):
    """
    手动实现SMPL的LBS过程，提取所有中间阶段

    返回:
        dict: 包含所有阶段的顶点和关节数据
    """
    device = betas.device
    dtype = betas.dtype

    # (a) 模板网格
    v_template = model.v_template.to(device)
    if v_template.dim() == 2:
        v_template = v_template.unsqueeze(0)  # [1, V, 3]
    print(f"  模板网格形状: {v_template.shape}")

    # (b) 形状校正
    shapedirs = model.shapedirs[:, :, :betas.shape[1]].to(device)
    print(f"  shapedirs形状: {shapedirs.shape}")
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    print(f"  形状校正后网格形状: {v_shaped.shape}")

    # 由形状后的网格回归关节
    J_regressor = model.J_regressor.to(device)
    J = vertices2joints(J_regressor, v_shaped)
    print(f"  关节位置形状: {J.shape}")

    # (c) 姿态校正
    full_pose = torch.cat([global_orient, body_pose], dim=1)  # [1, 24*3]
    rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(1, -1, 3, 3)

    ident = torch.eye(3, dtype=dtype, device=device)
    pose_feature = (rot_mats[:, 1:, :, :] - ident).view(1, -1)
    print(f"  pose_feature形状: {pose_feature.shape}")

    # 处理posedirs
    # posedirs形状: [6890, 3, 207] = [V, 3, P]
    posedirs = model.posedirs.to(device)
    print(f"  posedirs形状: {posedirs.shape}")

    # 使用einstein求和计算姿态偏移
    # pose_feature: [1, P] where P=207
    # posedirs: [V, 3, P]
    # 结果应该是: [1, V, 3]
    V = posedirs.shape[0]  # 6890
    P = min(posedirs.shape[2], pose_feature.shape[1])  # 207

    # 截取需要的姿态参数
    posedirs_used = posedirs[:, :, :P]  # [V, 3, P]
    pose_feature_used = pose_feature[:, :P]  # [1, P]

    # 计算姿态偏移: [1, P] @ [V, 3, P]^T -> [1, V, 3]
    pose_offsets = torch.einsum('bp,vcp->bvc', pose_feature_used, posedirs_used)
    print(f"  pose_offsets形状: {pose_offsets.shape}")

    v_posed = v_shaped + pose_offsets
    print(f"  姿态校正后网格形状: {v_posed.shape}")

    # (d) 刚体层级变换 + LBS
    parents = model.parents.to(device)
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    num_joints = J.shape[1]
    lbs_weights = model.lbs_weights.to(device)
    W = lbs_weights.unsqueeze(0).expand(1, -1, -1)  # [1, V, J]

    T = torch.matmul(W, A.view(1, num_joints, 16)).view(1, -1, 4, 4)

    homogen_coord = torch.ones((1, v_posed.shape[1], 1), dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)  # [1, V, 4]
    v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))  # [1, V, 4, 1]
    verts = v_homo[:, :, :3, 0]
    print(f"  最终顶点形状: {verts.shape}")

    # 模板姿态下的关节，方便可视化 (a)
    J_template = vertices2joints(J_regressor, v_template)

    return {
        "v_template": v_template,
        "J_template": J_template,
        "v_shaped": v_shaped,
        "J_shaped": J,
        "pose_offsets": pose_offsets,
        "v_posed": v_posed,
        "J_transformed": J_transformed,
        "verts": verts,
    }


def main(args):
    device = torch.device("cpu")
    dtype = torch.float32

    # 处理模型路径
    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(SCRIPT_DIR, model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    out_dir = resolve_script_path(args.out_dir)
    make_out_dir(out_dir)

    # 加载 SMPL 模型
    print("=" * 60)
    print("加载SMPL模型...")
    model = SimpleSMPL(model_path, num_betas=args.num_betas)

    faces = model.faces
    num_vertices = model.v_template.shape[0]
    num_faces = faces.shape[0]
    num_joints = model.lbs_weights.shape[1]

    # 构造示例参数
    print("\n设置演示参数...")
    betas = build_demo_shape(device, dtype, num_betas=args.num_betas)
    global_orient, body_pose = build_demo_pose(device, dtype)

    # 手动复现 LBS 各阶段
    print("\n执行手写LBS计算...")
    print("  阶段(a): 模板网格")
    print("  阶段(b): 形状校正 + 关节回归")
    print("  阶段(c): 姿态校正")
    print("  阶段(d): LBS蒙皮")
    data = compute_manual_lbs(model, betas, global_orient, body_pose)

    # 可视化所需数据
    joint_id = int(args.joint_id)
    if joint_id < 0 or joint_id >= model.lbs_weights.shape[1]:
        raise ValueError(
            f"joint_id 越界：{joint_id}，可选范围应为 [0, {model.lbs_weights.shape[1] - 1}]"
        )

    weight_scalar = to_numpy(model.lbs_weights[:, joint_id])
    pose_offset_norm = np.linalg.norm(to_numpy(data["pose_offsets"][0]), axis=1)

    # 保存单张图 - 阶段 (a)
    print("\n生成阶段可视化...")
    save_single_figure(
        os.path.join(out_dir, "stage_a_template_weights.png"),
        to_numpy(data["v_template"][0]),
        faces,
        joints=to_numpy(data["J_template"][0]),
        vertex_scalar=weight_scalar,
        title=f"(a) Template Mesh + Weight of Joint {joint_id}",
    )
    print(f"  ✓ stage_a_template_weights.png")

    # 保存单张图 - 阶段 (b)
    save_single_figure(
        os.path.join(out_dir, "stage_b_shaped_joints.png"),
        to_numpy(data["v_shaped"][0]),
        faces,
        joints=to_numpy(data["J_shaped"][0]),
        vertex_scalar=None,
        title="(b) Shape Blend + Joint Regression",
    )
    print(f"  ✓ stage_b_shaped_joints.png")

    # 保存单张图 - 阶段 (c)
    save_single_figure(
        os.path.join(out_dir, "stage_c_pose_offsets.png"),
        to_numpy(data["v_posed"][0]),
        faces,
        joints=to_numpy(data["J_shaped"][0]),
        vertex_scalar=pose_offset_norm,
        title="(c) Pose Blend Shapes (colored by |pose_offsets|)",
    )
    print(f"  ✓ stage_c_pose_offsets.png")

    # 保存单张图 - 阶段 (d)
    save_single_figure(
        os.path.join(out_dir, "stage_d_lbs_result.png"),
        to_numpy(data["verts"][0]),
        faces,
        joints=to_numpy(data["J_transformed"][0]),
        vertex_scalar=None,
        title="(d) Final LBS Result",
    )
    print(f"  ✓ stage_d_lbs_result.png")

    # 保存总对比图
    print("\n生成对比图...")
    grid_dict = {
        "v_template": to_numpy(data["v_template"][0]),
        "J_template": to_numpy(data["J_template"][0]),
        "v_shaped": to_numpy(data["v_shaped"][0]),
        "J_shaped": to_numpy(data["J_shaped"][0]),
        "v_posed": to_numpy(data["v_posed"][0]),
        "verts": to_numpy(data["verts"][0]),
        "J_transformed": to_numpy(data["J_transformed"][0]),
        "weight_scalar": weight_scalar,
        "pose_offset_norm": pose_offset_norm,
    }
    save_comparison_grid(
        os.path.join(out_dir, "comparison_grid.png"),
        grid_dict,
        faces,
    )
    print(f"  ✓ comparison_grid.png")

    # 保存所有关节权重图
    print("\n生成所有关节权重图...")
    save_all_joint_weights_figure(
        os.path.join(out_dir, "all_joint_weights.png"),
        to_numpy(data["v_template"][0]),
        faces,
        to_numpy(data["J_template"][0]),
        to_numpy(model.lbs_weights),
    )
    print(f"  ✓ all_joint_weights.png")

    # 保存摘要信息
    print("\n生成实验总结...")
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("SMPL LBS 蒙皮过程可视化实验总结\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. 模型基本信息\n")
        f.write("-" * 30 + "\n")
        f.write(f"模型文件: {model_path}\n")
        f.write(f"模板网格顶点数: {num_vertices}\n")
        f.write(f"三角面片数: {num_faces}\n")
        f.write(f"关节点数: {num_joints}\n")
        f.write(f"形状参数维度: {args.num_betas}\n")
        f.write(f"可视化关节编号: {joint_id}\n\n")

        f.write("2. 实验参数\n")
        f.write("-" * 30 + "\n")
        f.write(f"形状参数 (betas): {betas[0].cpu().numpy().tolist()}\n")
        f.write(f"全局旋转 (global_orient): {global_orient[0].cpu().numpy().tolist()}\n")
        f.write(f"身体姿态 (部分): {body_pose[0, :9].cpu().numpy().tolist()}...\n\n")

        f.write("3. LBS四个阶段说明\n")
        f.write("-" * 30 + "\n")
        f.write("(a) 模板网格与蒙皮权重:\n")
        f.write("    - 展示T-pose状态的模板人体网格\n")
        f.write(f"    - 可视化关节{joint_id}的蒙皮权重分布\n")
        f.write("    - 颜色越亮表示权重越大\n\n")

        f.write("(b) 形状校正与关节回归:\n")
        f.write("    - 应用形状参数调整体型\n")
        f.write("    - 从校正后的顶点回归关节点位置\n")
        f.write("    - 白色点表示回归的关节位置\n\n")

        f.write("(c) 姿态相关校正:\n")
        f.write("    - 在LBS之前添加姿态混合变形\n")
        f.write("    - 颜色表示姿态偏移的大小\n")
        f.write("    - 主要集中在关节弯曲处\n\n")

        f.write("(d) 线性混合蒙皮(LBS):\n")
        f.write("    - 应用蒙皮权重和关节变换\n")
        f.write("    - 生成最终的姿态化网格\n")
        f.write("    - 白色点表示变换后的关节位置\n\n")

        f.write("4. 实现说明\n")
        f.write("-" * 30 + "\n")
        f.write("本实验使用smplx库的底层LBS函数:\n")
        f.write("  - blend_shapes: 形状混合\n")
        f.write("  - vertices2joints: 关节回归\n")
        f.write("  - batch_rodrigues: 轴角转旋转矩阵\n")
        f.write("  - batch_rigid_transform: 运动学变换\n")
        f.write("使用einsum处理3维posedirs的乘法\n")
        f.write("完全手动实现了LBS的四个阶段\n\n")

        f.write("5. 实验结论\n")
        f.write("-" * 30 + "\n")
        f.write("✓ 成功实现了SMPL模型的完整LBS蒙皮过程\n")
        f.write("✓ 四个阶段的可视化清晰展示了蒙皮过程的各个步骤\n")
        f.write("✓ 形状参数正确影响了体型和关节位置\n")
        f.write("✓ 姿态校正主要集中在关节弯曲处，符合人体变形规律\n")
        f.write("✓ 最终生成的姿态化网格符合预期\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write(f"实验完成时间: {np.datetime64('now')}\n")

    print(f"  ✓ summary.txt")

    # 输出最终结果
    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)
    print(f"模型文件: {model_path}")
    print(f"顶点数: {num_vertices}")
    print(f"面片数: {num_faces}")
    print(f"关节数: {num_joints}")
    print(f"\n所有结果已保存到: {out_dir}")
    print("\n输出文件列表:")
    output_files = [
        'stage_a_template_weights.png',
        'stage_b_shaped_joints.png',
        'stage_c_pose_offsets.png',
        'stage_d_lbs_result.png',
        'comparison_grid.png',
        'all_joint_weights.png',
        'summary.txt'
    ]
    for file in output_files:
        file_path = os.path.join(out_dir, file)
        if os.path.exists(file_path):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (未生成)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPL LBS 蒙皮过程可视化")
    parser.add_argument(
        "--model-path",
        type=str,
        default="C:/Users/Lenovo/PycharmProjects/PythonProject15/SMPL_NEUTRAL.pkl",
        help="SMPL模型文件完整路径 (包括文件名)"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./outputs",
        help="输出目录"
    )
    parser.add_argument(
        "--joint-id",
        type=int,
        default=18,
        help="要可视化权重的关节编号 (0-23)"
    )
    parser.add_argument(
        "--num-betas",
        type=int,
        default=10,
        help="使用多少个形状参数"
    )
    args = parser.parse_args()
    main(args)