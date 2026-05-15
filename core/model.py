import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d.transforms as transforms
import numpy as np
import copy

from tools import *



def cat_data_dict_for_pred_gt(data_dict):
    """
    Build data_dict for concatenated [pred, gt] batch.
    If tensor value has shape [B, ...], output is [2B, ...] with [v, v].
    """
    out = {}
    for k, v in data_dict.items():
        if torch.is_tensor(v):
            out[k] = torch.cat([v, v], dim=0)
        elif isinstance(v, list):
            out[k] = v + v
        else:
            out[k] = v
    return out


def masked_mean(x, mask):
    """
    x: [B, T', ...]
    mask: [B, T']
    """
    if mask.sum() == 0:
        return x.sum() * 0.0

    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)

    return x[mask.expand_as(x)].mean()


def get_human_info(x, dataset=None):
    """
    NOTE: inputs are normalized.
    human_joints: [B, T, 24, 3]
    human_rot_mat: [B, T, 22, 3, 3], global rotation
    human_hands_feet_contact: [B, T, 4]
    """
    human_motion = x["motion"]
    human_hands_feet_contact = x["contact_labels"]
    B, T, _ = human_motion.shape

    human_joints = human_motion[:, :, :24 * 3].reshape(B, T, 24, 3)
    global_joint_rot_6d = human_motion[:, :, 24 * 3:].reshape(B, T, 22, 6)
    human_rot_mat = transforms.rotation_6d_to_matrix(global_joint_rot_6d)
    return human_joints, human_rot_mat, human_hands_feet_contact


def get_object_info(x, dataset, floor_height=0.02, quick_mode=True):
    """
    obj_bps: [B, 1, 1024, 3]
    obj_motion: [B, T, 12], normalized translation + local rotation
    obj_floor_contact: [B, T, 1]
    obj_pc: [B, T, Nv, 3], world-space object keypoints/geometry
    """
    obj_bps = x["input_obj_bps"]
    obj_motion = x["obj_motion"]

    if quick_mode:
        obj_pc = get_obj_world_keypoints_parallel(obj_motion, x["reference_obj_rot_mat"], x["rest_pose_obj_pts"], dataset)
        min_values = torch.min(obj_pc[:, :, :, -1], dim=2)[0]
        obj_floor_contact = ((min_values < floor_height) * 1.).unsqueeze(-1)
    else:
        obj_pc = get_world_object_geo(x["obj_name"], x["obj_motion"], x["reference_obj_rot_mat"], dataset)
        min_values = torch.stack([torch.min(item[:, :, -1], dim=1)[0] for item in obj_pc])
        obj_floor_contact = ((min_values < floor_height) * 1.).unsqueeze(-1)

    return obj_motion, obj_bps, obj_floor_contact, obj_pc


def det3x3(m):
    """
    m: [N, 3, 3]
    return: [N]
    """
    return (
        m[:, 0, 0] * (m[:, 1, 1] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 1])
        - m[:, 0, 1] * (m[:, 1, 0] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 0])
        + m[:, 0, 2] * (m[:, 1, 0] * m[:, 2, 1] - m[:, 1, 1] * m[:, 2, 0])
    )


def symmetric_orthogonalization(x):
    """
    Maps 9D input vectors onto SO(3).
    GPU version without torch.det, avoiding MAGMA batched-LU crash.

    x: [N, 9]
    return:
        r: [N, 3, 3]
        det_m: [N]
    """
    m = x.view(-1, 3, 3)

    if not torch.isfinite(m).all():
        raise RuntimeError("[symmetric_orthogonalization] Input contains NaN or Inf.")

    u, s, vh = torch.linalg.svd(m, full_matrices=False)

    r_tmp = torch.bmm(u, vh)
    det = det3x3(r_tmp).sign().view(-1, 1, 1)

    vh_fixed = torch.cat((vh[:, :2, :], vh[:, -1:, :] * det), dim=1)
    r = torch.bmm(u, vh_fixed)
    det_m = torch.prod(s, dim=-1)

    return r, det_m

# def symmetric_orthogonalization(x):
#     """
#     Maps 9D input vectors onto SO(3).
#     x: [N, 9]
#     return:
#         r: [N, 3, 3]
#         det_m: [N]
#     """
#     device = x.device
#     x = x.cpu()
#     m = x.view(-1, 3, 3)
#     u, s, v = torch.svd(m)
#     vt = torch.transpose(v, 1, 2)
#     det = torch.det(torch.matmul(u, vt)).view(-1, 1, 1)
#     vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), dim=1)
#     r = torch.matmul(u, vt)
#     det_m = torch.prod(s, dim=-1)
#     return r.to(device), det_m.to(device)


def get_obj_world_keypoints_parallel(obj_motion, reference_obj_rot_mat, rest_pose_obj_pts, dataset):
    """
    obj_motion: [B, T', 12], normalized trans + local rot
    reference_obj_rot_mat: [B, 3, 3]
    rest_pose_obj_pts: [B, Nv, 3]
    return: [B, T', Nv, 3], world-space object keypoints
    """
    B, T2, _ = obj_motion.shape
    Nv = rest_pose_obj_pts.shape[1]

    transition = obj_motion[:, :, :3]
    rotation = obj_motion[:, :, 3:].reshape(B, T2, 3, 3)
    curr_obj_com_pos = dataset.de_normalize_obj_pos_min_max(transition)

    curr_obj_rot_mat = torch.bmm(rotation.reshape(-1, 3, 3), reference_obj_rot_mat.repeat(1, T2, 1, 1).reshape(-1, 3, 3))
    curr_obj_rot_mat = curr_obj_rot_mat.reshape(B, T2, 3, 3)

    rest_verts = rest_pose_obj_pts[:, None].repeat(1, T2, 1, 1).reshape(B * T2, Nv, 3)
    transformed = torch.bmm(curr_obj_rot_mat.reshape(-1, 3, 3), rest_verts.transpose(1, 2)).transpose(1, 2)
    transformed = transformed.reshape(B, T2, Nv, 3) + curr_obj_com_pos[:, :, None, :]
    return transformed




class MiniTransformer(nn.Module):
    def __init__(self, n=24, joint_dim=12, cond_dim=92, dim=64, heads=4, depth=1):
        super().__init__()
        self.joint_encoder = nn.Sequential(nn.Linear(joint_dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim))
        self.cond_encoder = nn.Sequential(nn.Linear(cond_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        # self.pos_embed = nn.Parameter(torch.randn(1, n, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n, dim))

        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim * 2, batch_first=True, activation="relu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.attn_pool = nn.Linear(dim, 1)

    def forward(self, joints, cond):
        x = self.joint_encoder(joints) + self.cond_encoder(cond).unsqueeze(1) + self.pos_embed
        x = self.transformer(x)
        attn = F.softmax(self.attn_pool(x).squeeze(-1), dim=1)
        feat = (x * attn.unsqueeze(-1)).sum(dim=1)
        return feat, attn


class Dynamics(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.history_win = cfg["history_win"]
        self.future_win = cfg["future_win"]
        self.feat_dim = cfg["feat_dim"]
        self.head = cfg["head"]
        self.depth = cfg["depth"]
        self.rot_loss_method = cfg["rot_loss_method"]

        assert self.future_win == 1, "Clean Dynamics only supports future_win=1. Use max_step to control prediction stride."

        self.bps_encoder = nn.Sequential(nn.Linear(1024 * 3, 128), nn.ReLU(), nn.Linear(128, self.feat_dim))

        cond_dim = self.history_win * 12 + self.feat_dim + 4
        self.human_joints_transformer = MiniTransformer(n=24, joint_dim=12, cond_dim=cond_dim, dim=self.feat_dim, heads=self.head, depth=self.depth)
        self.tran_rot_enc = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim // 2), nn.ReLU(), nn.Linear(self.feat_dim // 2, 12))


    @staticmethod
    def data_prepare(x, dataset):
        obj_motion, obj_bps, obj_floor_contact, obj_pc = get_object_info(x, dataset)
        human_joints, human_rot_mat, human_contact = get_human_info(x, dataset)
        B, T, _ = obj_motion.shape
        hand_idx = [20, 21, 22, 23]
        feet_idx = [7, 8, 10, 11]

        inputs = {
            "obj_motion": obj_motion,
            "human_joints": human_joints.view(B, T, 24 * 3),
            "human_rot_mat": human_rot_mat.view(B, T, 22 * 3 * 3),
            "contact": human_contact.view(B, T, 4),
        }

        others = {
            "obj_pc": obj_pc.reshape(B, T, 100 * 3),
            "obj_bps": obj_bps.view(B, 1, 1024 * 3),
            "obj_floor_contact": obj_floor_contact,
            "human_hands_joints": human_joints[:, :, hand_idx].view(B, T, 4 * 3),
            "human_feet_joints": human_joints[:, :, feet_idx].view(B, T, 4 * 3),
        }

        return inputs, others

    @staticmethod
    def compose_object_history(inputs, history_ts):
        obj_motion, contact = inputs["obj_motion"], inputs["contact"]
        B, H, T2 = history_ts.shape

        obj_hist = []
        for i in range(H):
            obj_hist.append(get_state_by_ts(obj_motion, history_ts[:, i, :], bt_merge=True).unsqueeze(1))

        obj_hist = torch.cat(obj_hist, dim=1)
        obj_hist_pose = obj_hist.reshape(B * T2, H * 12)
        cur_contact = get_state_by_ts(contact, history_ts[:, -1, :], bt_merge=True)
        return obj_hist_pose, cur_contact

    @staticmethod
    def compose_human_drive(inputs, history_ts, future_ts):
        human_joints, human_rot_mat = inputs["human_joints"], inputs["human_rot_mat"]
        B, H, T2 = history_ts.shape
        anchor_ts, target_ts = history_ts[:, -1, :], future_ts[:, 0, :]

        cur_joints = get_state_by_ts(human_joints, anchor_ts, bt_merge=False)
        tgt_joints = get_state_by_ts(human_joints, target_ts, bt_merge=False)
        cur_rot = get_state_by_ts(human_rot_mat, anchor_ts, bt_merge=False)
        tgt_rot = get_state_by_ts(human_rot_mat, target_ts, bt_merge=False)

        delta_joints = tgt_joints.view(B, T2, 24, 3) - cur_joints.view(B, T2, 24, 3)
        delta_rot = torch.matmul(tgt_rot.view(B, T2, 22, 3, 3), cur_rot.view(B, T2, 22, 3, 3).transpose(-1, -2)).reshape(B * T2, 22, 9)
        pad_rot = torch.zeros(B * T2, 2, 9, device=delta_rot.device, dtype=delta_rot.dtype)

        delta_joints = delta_joints.reshape(B * T2, 24, 3)
        delta_rot = torch.cat([delta_rot, pad_rot], dim=1)
        human_drive = torch.cat([delta_joints, delta_rot], dim=-1)
        return human_drive
    
    def adaptive_object_history(self, obj_hist_pose, cur_contact):
        """
        Contact-adaptive object history conditioning.

        For H=2:
        - strong contact: use [obj_t, obj_t]
        - no/weak contact: use [obj_{t-1}, obj_t]
        """
        if self.history_win != 2:
            return obj_hist_pose

        obj_dim = obj_hist_pose.shape[-1] // 2

        prev_obj = obj_hist_pose[..., :obj_dim]
        curr_obj = obj_hist_pose[..., obj_dim:]

        obj_hist_contact = torch.cat([curr_obj, curr_obj], dim=-1)
        obj_hist_nocontact = torch.cat([prev_obj, curr_obj], dim=-1)

        # robust contact score: avoid treating tiny positive noise as contact
        contact_score = cur_contact.clamp(0.0, 1.0).max(dim=-1, keepdim=True).values
        print(
            "gate mean:", contact_score.mean().item(),
            "gate == 1:", (contact_score > 0.5).float().mean().item(),
            "gate == 0:", (contact_score <= 0.5).float().mean().item(),
        )

        # soft gate, better for generated continuous contact
        contact_gate = contact_score.detach()

        obj_hist_pose = contact_gate * obj_hist_contact + (1.0 - contact_gate) * obj_hist_nocontact

        return obj_hist_pose

    def encode_adaptive_object_history(self, obj_hist_pose, cur_contact):
        if self.history_win != 2:
            return obj_hist_pose

        obj_dim = obj_hist_pose.shape[-1] // 2

        prev_obj = obj_hist_pose[..., :obj_dim]   # [B*T2, 12]
        curr_obj = obj_hist_pose[..., obj_dim:]   # [B*T2, 12]

        obj_hist_h2 = torch.cat([prev_obj, curr_obj], dim=-1)  # [B*T2, 24]

        feat_h1 = self.obj_hist_h1_encoder(curr_obj)       # [B*T2, 24]
        feat_h2 = self.obj_hist_h2_encoder(obj_hist_h2)    # [B*T2, 24]

        gate = cur_contact.clamp(0.0, 1.0).max(dim=-1, keepdim=True).values.detach()

        obj_hist_feat = gate * feat_h1 + (1.0 - gate) * feat_h2

        # explicitly tell the shared backbone which mode it is seeing
        obj_hist_feat = obj_hist_feat + self.obj_hist_branch_embed(gate)

        return obj_hist_feat

    def predict(self, inputs, others, max_step=1):
        H, F, S = self.history_win, self.future_win, max_step
        B, T, _ = inputs["contact"].shape
        device = inputs["contact"].device

        history_ts, future_ts = get_advanced_timespan(B=B, H=H, F=F, T=T, S=S)
        history_ts, future_ts = history_ts.to(device), future_ts.to(device)
        B, H, T2 = history_ts.shape

        encoded_bps = self.bps_encoder(others["obj_bps"]).repeat(1, T, 1)
        obj_bps_feat = get_state_by_ts(encoded_bps, history_ts[:, -1, :], bt_merge=True)

        obj_hist_pose, cur_contact = self.compose_object_history(inputs, history_ts)

        condition = torch.cat([obj_hist_pose, obj_bps_feat, cur_contact], dim=-1)

        human_drive = self.compose_human_drive(inputs, history_ts, future_ts)
        human_feat, joints_attn = self.human_joints_transformer(human_drive, condition)

        out = self.tran_rot_enc(human_feat).view(B, T2, F, 12)
        trans, raw_rot = out[..., :3], out[..., 3:12]

        if self.rot_loss_method == "fro":
            rot, det_m = raw_rot, None
        elif self.rot_loss_method == "theta":
            rot_flat, det_m = symmetric_orthogonalization(raw_rot.reshape(-1, 9).float())
            rot = rot_flat.view(B, T2, F, 9)
        else:
            raise NotImplementedError

        return {
            "trans": trans,
            "rot": rot,
            "raw_rot": raw_rot,
            "contact": cur_contact,
            "timespan": history_ts,
            "next_timespan": future_ts,
            "method": self.rot_loss_method,
            "joints_attn": joints_attn,
            "det_m": det_m,
        }

    def forward(self, x, dataset, max_step=1):
        if isinstance(dataset, torch.utils.data.dataset.Subset):
            dataset = dataset.dataset

        inputs, others = self.data_prepare(x, dataset)
        pred = self.predict(inputs, others, max_step=max_step)
        pred.update({"hands_feet_contact": inputs["contact"], "obj_floor_contact": others["obj_floor_contact"]})
        return pred




    #######################################################################################################
    # The following functions are for computing guidance loss from generated motion, modified by wulin 05/12/2026
    #######################################################################################################

    @staticmethod
    def double_data_dict(data_dict):
        for key in data_dict.keys():
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = torch.cat([data_dict[key], data_dict[key]], dim=0)
            elif isinstance(data_dict[key], list):
                data_dict[key] = data_dict[key] + data_dict[key]
            else:
                raise NotImplementedError
        return data_dict


    @staticmethod
    def parse_hoi(x):
        # B, T, 220 = obj(3+9) + human_joints(24*3) + human_rot(22*6) + contact(4)
        B, T, D = x.shape
        assert D == 220, f"Expected HOI dim=220, got {D}"

        obj_motion = x[:, :, :3 + 9]
        human_joints = x[:, :, 3 + 9:3 + 9 + 24 * 3]
        human_rot_6d = x[:, :, 3 + 9 + 24 * 3:3 + 9 + 24 * 3 + 22 * 6].view(B, T, 22, 6)
        human_rot_mat = transforms.rotation_6d_to_matrix(human_rot_6d).view(B, T, -1)
        contact = x[:, :, 3 + 9 + 24 * 3 + 22 * 6:3 + 9 + 24 * 3 + 22 * 6 + 4]

        return {
            "obj_motion": obj_motion,
            "human_joints": human_joints,
            "human_rot_mat": human_rot_mat,
            "contact": contact,
        }


    def prepare_hoi_for_dynamics(self, x, data_dict=None, contact_source="pred"):
        inputs = self.parse_hoi(x["gt_x0"])
        diffused = self.parse_hoi(x["pred_x0"])

        if contact_source == "gt":
            diffused["contact"] = inputs["contact"].detach()
        elif contact_source != "pred":
            raise ValueError(f"Unknown contact_source: {contact_source}")

        if "bps" in x:
            bps = x["bps"]
        elif data_dict is not None and "input_obj_bps" in data_dict:
            bps = data_dict["input_obj_bps"]
        else:
            raise KeyError("Cannot find bps from x['bps'] or data_dict['input_obj_bps'].")

        B, T, _ = x["pred_x0"].shape
        bps = bps.to(x["pred_x0"].device).view(B, 1, 1024 * 3)

        if "padding_mask" in x and x["padding_mask"] is not None:
            padding_mask = x["padding_mask"][:, :, 1:]
        else:
            padding_mask = torch.ones(B, 1, T, device=x["pred_x0"].device)

        return inputs, diffused, bps, padding_mask


    def _rot_loss_each(self, pred_rot, target_rot, method):
        if method == "fro":
            return torch.abs(pred_rot - target_rot)

        elif method == "theta":
            B, T2, _ = pred_rot.shape
            R_pred = pred_rot.reshape(B * T2, 3, 3)
            R_gt = target_rot.reshape(B * T2, 3, 3)
            R_rel = torch.bmm(R_pred.transpose(-1, -2), R_gt)
            trace = torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1)
            theta = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6))
            return theta.view(B, T2, 1)

        else:
            raise NotImplementedError


    def dynamics_loss(self, pred, inputs, padding_mask, data_dict=None, dataset=None, loss_type="pc", rot_weight=0.05):
        obj_motion = inputs["obj_motion"]

        anchor_ts = pred["timespan"][:, -1, :]
        target_ts = pred["next_timespan"][:, 0, :]
        method = pred["method"]

        dyn_trans = pred["trans"][:, :, 0]
        dyn_rot = pred["rot"][:, :, 0]

        target_delta = get_delta_obj_motion(obj_motion, anchor_ts, target_ts, method=method, ret="trans_and_rot")
        target_trans, target_rot = target_delta[:, :, :3], target_delta[:, :, 3:]
        dt = (target_ts - anchor_ts).clamp(min=1).float().unsqueeze(-1)

        mask = get_state_by_ts(
            padding_mask.transpose(1, 2).float(),
            anchor_ts,
            bt_merge=False,
        )

        trans_loss = torch.abs(dyn_trans - target_trans) / dt
        trans_loss = trans_loss * mask

        rot_loss = self._rot_loss_each(dyn_rot, target_rot, method) / dt
        rot_loss = rot_loss * mask

        losses = {"trans_loss": trans_loss.detach(), "rot_loss": rot_loss.detach()}

        if loss_type == "trans":
            losses["pc_loss"] = trans_loss
            return losses

        elif loss_type == "trans_rot":
            losses["pc_loss"] = trans_loss + rot_weight * rot_loss
            return losses

        elif loss_type == "pc":
            obj_from_dyn = self.compose_obj_motion_from_delta(obj_motion=obj_motion, pred_trans=dyn_trans, pred_rot=dyn_rot, anchor_ts=anchor_ts, method=method)
            obj_target = get_state_by_ts(obj_motion, target_ts, bt_merge=False)

            ref_rot = data_dict["reference_obj_rot_mat"].to(obj_motion.device)
            rest_pts = data_dict["rest_pose_obj_pts"].to(obj_motion.device)

            from_dyn_verts = get_obj_world_keypoints_parallel(obj_from_dyn, ref_rot, rest_pts, dataset)
            target_verts = get_obj_world_keypoints_parallel(obj_target, ref_rot, rest_pts, dataset)

            B, T2 = anchor_ts.shape
            pc_loss = torch.abs(target_verts - from_dyn_verts).reshape(B, T2, -1)
            pc_loss = pc_loss * mask

            losses["pc_loss"] = pc_loss
            return losses

        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")


    def compute_dynamics_loss_for_generated_hoi(
        self, x, max_step=1, data_dict=None, dataset=None,
        loss_type="pc", dyn_loss_type="res", rot_weight=0.05,
        contact_source="pred", detach_gt_branch=True,
    ):
        
        self.eval()  # freeze dynamics behavior: disable dropout/BN, but keep gradients w.r.t. inputs

        data_dict = copy.deepcopy(data_dict)
        inputs, diffused, bps, padding_mask = self.prepare_hoi_for_dynamics(x, data_dict=data_dict, contact_source=contact_source)

        B = bps.shape[0]
        unified_input = {key: torch.cat([inputs[key], diffused[key]], dim=0) for key in inputs.keys()}
        unified_others = {"obj_bps": torch.cat([bps, bps], dim=0)}
        unified_padding_mask = torch.cat([padding_mask, padding_mask], dim=0)
        unified_data_dict = self.double_data_dict(data_dict)

        unified_pred = self.predict(unified_input, unified_others, max_step=max_step)
        unified_losses = self.dynamics_loss(
            unified_pred, unified_input, unified_padding_mask,
            data_dict=unified_data_dict, dataset=dataset,
            loss_type=loss_type, rot_weight=rot_weight,
        )

        learn_loss = {key: unified_losses[key][:B].detach() for key in unified_losses.keys()}

        if dyn_loss_type == "res":
            gt_loss = {key: unified_losses[key][:B].detach() if detach_gt_branch else unified_losses[key][:B] for key in unified_losses.keys()}
            regulate_loss = {key: torch.abs(unified_losses[key][B:] - gt_loss[key]) for key in unified_losses.keys()}
        elif dyn_loss_type == "non_res":
            regulate_loss = {key: unified_losses[key][B:] for key in unified_losses.keys()}
        else:
            raise ValueError(f"Unknown dyn_loss_type: {dyn_loss_type}")

        return {
            "learn_dynamics": learn_loss,
            "regulate_dynamics": regulate_loss,
            "debug": {
                "loss_type": loss_type,
                "dyn_loss_type": dyn_loss_type,
                "contact_source": contact_source,
                "detach_gt_branch": detach_gt_branch,
                "cat_batch": True,
                "batch_size_original": B,
                "batch_size_cat": 2 * B,
            },
        }
