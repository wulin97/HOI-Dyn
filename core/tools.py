import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from scipy.spatial.transform import Rotation as Rota
import pytorch3d.transforms as transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from manip.data.cano_traj_dataset import CanoObjectTrajDataset
from manip.data.cano_traj_dataset import quat_ik_torch, quat_fk_torch
from trainer_chois import run_smplx_model


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def move_data_to_device(inp, device):
    for k, v in inp.items():
        if isinstance(v, torch.Tensor):
            inp[k] = v.to(device)
    return inp


def extract_rotation_euler_batch(rotation_flat_batch):
    rotation_matrices = rotation_flat_batch.reshape(-1, 3, 3)
    r = Rota.from_matrix(rotation_matrices)
    return r.as_euler("xyz", degrees=True)


def stratified_sample_from_ranges(ranges, B):
    """
    Sample integer indices from inclusive ranges.

    Args:
        ranges: [N, 2], each row is [left, right].
        B: batch size.

    Returns:
        samples: [B, N].
    """
    assert (ranges[:, 0] <= ranges[:, 1]).all(), f"Invalid ranges: {ranges}"

    N = ranges.size(0)
    rand_values = torch.rand(N, B, device=ranges.device)
    left = ranges[:, 0].unsqueeze(1).float()
    right = ranges[:, 1].unsqueeze(1).float()
    samples = torch.round(left + (right - left) * rand_values)
    return samples.transpose(0, 1)


def get_advanced_timespan(B=1, H=1, F=1, T=120, S=3):
    """
    Build temporal windows.

    History:
        H contiguous frames.

    Future:
        F contiguous frames.
        The first future frame is sampled from [last_history + 1, last_history + S],
        clipped to keep the full future window inside the sequence.

    Args:
        B: batch size.
        H: number of history frames.
        F: number of future frames.
        T: sequence length.
        S: maximum prediction stride from the last history frame.

    Returns:
        history_ts: [B, H, T']
        future_ts: [B, F, T']
    """
    assert H >= 1 and F >= 1 and S >= 1
    assert T >= H + F, f"Sequence too short: T={T}, H={H}, F={F}"

    # The last valid future start is T - F.
    # The last valid history anchor must be <= T - F - 1.
    # Since anchor = start + H - 1, start <= T - H - F.
    num_windows = T - H - F + 1

    history_ts = [torch.arange(h, num_windows + h) for h in range(H)]
    history_ts = torch.stack(history_ts, dim=0).unsqueeze(0).repeat(B, 1, 1).to(torch.int64)

    t_last = history_ts[0, -1]  # [T']
    left = t_last + 1
    right = torch.minimum(t_last + S, torch.tensor(T - F, dtype=torch.int64))

    ranges = torch.stack((left, right), dim=-1)
    next_t = stratified_sample_from_ranges(ranges, B).to(torch.int64)

    future_ts = next_t.unsqueeze(1) + torch.arange(F, dtype=torch.int64).view(1, F, 1)
    future_ts = future_ts.to(torch.int64)

    assert (history_ts >= 0).all() and (history_ts < T).all()
    assert (future_ts >= 0).all() and (future_ts < T).all()

    return history_ts, future_ts


def get_timespan(B, T=120, max_step=3):
    """
    Legacy 4-frame history timespan.
    Kept for compatibility with old experiments.
    """
    t1 = torch.arange(0, T - 4)
    t2 = torch.arange(1, T - 3)
    t3 = torch.arange(2, T - 2)
    t4 = torch.arange(3, T - 1)

    left = t4 + 1
    right = torch.minimum(t4 + max_step, torch.tensor(T - 1))
    ranges = torch.stack((left, right), dim=-1)

    next_ts = stratified_sample_from_ranges(ranges, B).to(torch.int64)
    ts1 = t1.unsqueeze(0).repeat(B, 1).to(torch.int64)
    ts2 = t2.unsqueeze(0).repeat(B, 1).to(torch.int64)
    ts3 = t3.unsqueeze(0).repeat(B, 1).to(torch.int64)
    ts4 = t4.unsqueeze(0).repeat(B, 1).to(torch.int64)

    return ts1, ts2, ts3, ts4, next_ts


def _normalize_timespan(timespan):
    """
    Convert timespan to [B, T'].

    Supported shapes:
        [B, T']
        [B, 1, T']
    """
    if timespan.dim() == 3:
        assert timespan.shape[1] == 1, f"Expected [B,1,T'] when 3D, got {timespan.shape}"
        timespan = timespan[:, 0, :]
    return timespan


def get_state_by_ts(x, timespan, bt_merge=False):
    """
    Gather temporal states.

    Args:
        x: [B, T, D]
        timespan: [B, T'] or [B, 1, T']
        bt_merge: whether to reshape output to [B*T', D]

    Returns:
        [B, T', D] or [B*T', D]
    """
    timespan = _normalize_timespan(timespan)
    B, T, D = x.shape

    timespan = timespan.to(x.device)
    batch_idx = torch.arange(B, device=x.device).unsqueeze(-1)
    new_x = x[batch_idx, timespan]

    if bt_merge:
        new_x = new_x.reshape(B * timespan.shape[1], D)

    return new_x


def get_delta_obj_motion(obj_motion, timespan, next_timespan, method="theta", ret="trans_and_rot"):
    """
    Compute object delta motion from current frame to target frame.

    Args:
        obj_motion: [B, T, 12]
        timespan: [B, T'] or [B, 1, T']
        next_timespan: [B, T'] or [B, 1, T']
        method: "theta" or "fro"
        ret: "trans_and_rot", "trans", or "rot"

    Returns:
        Delta object motion.
    """
    timespan = _normalize_timespan(timespan)
    next_timespan = _normalize_timespan(next_timespan)

    B, T, _ = obj_motion.shape
    T2 = timespan.shape[1]

    timespan = timespan.to(obj_motion.device)
    next_timespan = next_timespan.to(obj_motion.device)
    batch_idx = torch.arange(B, device=obj_motion.device).unsqueeze(-1)

    cur = obj_motion[batch_idx, timespan]
    nxt = obj_motion[batch_idx, next_timespan]

    delta_trans = nxt[:, :, :3] - cur[:, :, :3]

    if method == "fro":
        delta_rot = nxt[:, :, 3:] - cur[:, :, 3:]
    elif method == "theta":
        R_next = nxt[:, :, 3:].reshape(B * T2, 3, 3)
        R_cur = cur[:, :, 3:].reshape(B * T2, 3, 3)
        delta_rot = torch.bmm(R_cur.transpose(1, 2), R_next).view(B, T2, 9)
    else:
        raise NotImplementedError(f"Unknown rotation method: {method}")

    if ret == "trans_and_rot":
        return torch.cat([delta_trans, delta_rot], dim=-1)
    if ret == "trans":
        return delta_trans
    if ret == "rot":
        return delta_rot

    raise ValueError(f"Unknown return type: {ret}")


def get_world_object_geo(obj_name, obj_motion, reference_obj_rot_mat, dataset):
    """
    Convert normalized object motion to world-space object geometry.

    Args:
        obj_name: list of object names.
        obj_motion: [B, T, 12].
        reference_obj_rot_mat: [B, 3, 3] or [B, 1, 3, 3].
        dataset: dataset object.

    Returns:
        List of tensors. Each tensor has shape [T, N, 3].
    """
    B, T, _ = obj_motion.shape

    if reference_obj_rot_mat.dim() == 4:
        reference_obj_rot_mat = reference_obj_rot_mat[:, 0]

    batch_obj_mesh_verts = []

    for i in range(B):
        obj_rest_verts, _ = dataset.load_rest_pose_object_geometry(obj_name[i])
        obj_rest_verts = torch.from_numpy(obj_rest_verts).float().to(obj_motion.device)

        obj_com_pos = dataset.de_normalize_obj_pos_min_max(obj_motion[i, :, :3])
        obj_rot_mat = torch.bmm(obj_motion[i, :, 3:].view(T, 3, 3), reference_obj_rot_mat[i].repeat(T, 1, 1))
        obj_mesh_verts = dataset.load_object_geometry_w_rest_geo(obj_rot_mat, obj_com_pos, obj_rest_verts)
        batch_obj_mesh_verts.append(obj_mesh_verts)

    return batch_obj_mesh_verts


def get_world_human_geo(x, dataset, joints_only=True):
    """
    Convert normalized human motion to world-space joints or meshes.
    """
    trans2joint = x["trans2joint"]
    rest_human_offsets = x["rest_human_offsets"]
    motion = x["motion"]
    gender = x["gender"]
    betas = x["betas"]

    B, T, _ = motion.shape
    normed_jpos = motion[:, :, :24 * 3].reshape(B, T, 24, 3)
    global_jpos = dataset.de_normalize_jpos_min_max(normed_jpos)

    global_joint_rot_6d = motion[:, :, 24 * 3:].reshape(B, T, 22, 6)
    global_joint_rot_mat = transforms.rotation_6d_to_matrix(global_joint_rot_6d)

    if joints_only:
        rest_human_offsets = rest_human_offsets[:, None].repeat(1, T, 1, 1)
        batch_human_jnts = []

        for i in range(B):
            curr_seq_local_jpos = rest_human_offsets[i]
            curr_seq_local_jpos[:, 0, :] = global_jpos[i][:, 0, :]

            local_joint_rot_mat = quat_ik_torch(global_joint_rot_mat[i])
            _, human_jnts = quat_fk_torch(local_joint_rot_mat, curr_seq_local_jpos)
            batch_human_jnts.append(human_jnts)

        return torch.cat(batch_human_jnts, dim=0)

    global_root_jpos = global_jpos[:, :, 0, :]
    batch_mesh_jnts, batch_mesh_verts, batch_mesh_faces = [], [], []

    for i in range(B):
        root_trans = global_root_jpos[i] + trans2joint[i:i + 1].to(global_root_jpos.device)
        curr_local_rot_mat = quat_ik_torch(global_joint_rot_mat[i])
        curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(curr_local_rot_mat)

        mesh_jnts, mesh_verts, mesh_faces = run_smplx_model(
            root_trans[None].cuda(),
            curr_local_rot_aa_rep[None].cuda(),
            betas[i].cuda(),
            [gender[i]],
            dataset.bm_dict,
            return_joints24=True,
        )

        batch_mesh_jnts.append(mesh_jnts)
        batch_mesh_verts.append(mesh_verts)
        batch_mesh_faces.append(mesh_faces.unsqueeze(0))

    batch_mesh_jnts = torch.cat(batch_mesh_jnts, dim=0)
    batch_mesh_verts = torch.cat(batch_mesh_verts, dim=0)
    batch_mesh_faces = torch.cat(batch_mesh_faces, dim=0)

    return batch_mesh_jnts, batch_mesh_verts, batch_mesh_faces


def get_delta_human_motion(inputs, timespan, next_timespan):
    """
    Compute human joint translation and rotation deltas.
    """
    human_joints = inputs["human_joints"]
    human_rot_mat = inputs["human_rot_mat"]

    cur_human_rot_mat = get_state_by_ts(human_rot_mat, timespan, bt_merge=False)
    next_human_rot_mat = get_state_by_ts(human_rot_mat, next_timespan, bt_merge=False)
    cur_human_joints = get_state_by_ts(human_joints, timespan, bt_merge=False)
    next_human_joints = get_state_by_ts(human_joints, next_timespan, bt_merge=False)

    B, T2, _ = cur_human_rot_mat.shape

    delta_human_joints_trans = next_human_joints.view(B, T2, 24, 3) - cur_human_joints.view(B, T2, 24, 3)
    delta_human_joints_rot = torch.matmul(
        next_human_rot_mat.view(B, T2, 22, 3, 3),
        cur_human_rot_mat.view(B, T2, 22, 3, 3).transpose(-1, -2),
    )

    return delta_human_joints_trans.view(B * T2, 24, 3), delta_human_joints_rot.view(B * T2, 22, 9)


def prepare_dataset(train=False, batch_size=1, window_size=120, data_root_folder="./processed_data"):
    dataset = CanoObjectTrajDataset(
        train=train,
        data_root_folder=data_root_folder,
        window=window_size,
        use_object_splits=False,
        input_language_condition=True,
        use_random_frame_bps=True,
        use_object_keypoints=True,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    return dataset, data_loader


def prepare_train_val_dataset_by_randsplit(train_batch_size=128, val_batch_size=128, val_ratio=0.1):
    train_dataset, _ = prepare_dataset(train=True)

    val_size = int(len(train_dataset) * val_ratio)
    train_size = len(train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataset, train_loader, val_dataset, val_loader