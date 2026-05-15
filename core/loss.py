import torch
from tools import get_state_by_ts, get_advanced_timespan
from model import get_delta_obj_motion, get_obj_world_keypoints_parallel


def get_rot_loss(pred_obj_rot, gt_obj_rot, timespan, next_timespan, method="theta"):
    B, T2, _ = pred_obj_rot.shape
    dt = (next_timespan - timespan).clamp(min=1).float()

    if method == "fro":
        diff = pred_obj_rot.reshape(B, T2, 3, 3) - gt_obj_rot.reshape(B, T2, 3, 3)
        loss = torch.linalg.matrix_norm(diff, ord="fro", dim=(-2, -1)) / dt
    elif method == "theta":
        R_rel = torch.bmm(pred_obj_rot.reshape(B * T2, 3, 3).transpose(-2, -1), gt_obj_rot.reshape(B * T2, 3, 3))
        trace = torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1)
        theta = torch.acos(torch.clamp((trace - 1) / 2, -1.0 + 1e-6, 1.0 - 1e-6)).view(B, T2)
        loss = theta / dt
    else:
        raise NotImplementedError

    return loss.unsqueeze(-1)


def get_trans_loss(pred_obj_trans, gt_obj_trans, timespan, next_timespan):
    dt = (next_timespan - timespan).clamp(min=1).float().unsqueeze(-1)
    return (pred_obj_trans - gt_obj_trans).abs() / dt


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


def get_contact_status(pred):
    """
    contact:
        anchor frame or target frame has human hand/feet contact.
    no_contact:
        both anchor frame and target frame have no human hand/feet contact.

    This uses both anchor and target because max_step may be larger than 1.
    """
    anchor_ts = pred["timespan"][:, -1, :]
    target_ts = pred["next_timespan"][:, 0, :]

    anchor_contact = get_state_by_ts(pred["hands_feet_contact"], anchor_ts, bt_merge=False)
    target_contact = get_state_by_ts(pred["hands_feet_contact"], target_ts, bt_merge=False)

    anchor_has_contact = anchor_contact.sum(-1).to(torch.bool)
    target_has_contact = target_contact.sum(-1).to(torch.bool)

    contact_mask = anchor_has_contact | target_has_contact
    no_contact_mask = ~contact_mask

    return {
        "contact_mask": contact_mask,
        "no_contact_mask": no_contact_mask,
        "contact_ratio": contact_mask.float().mean(),
        "no_contact_ratio": no_contact_mask.float().mean(),
    }


def get_static_status(pred, use_floor_contact=True):
    """
    static state:
        object is on floor and no human hand/feet contact.
    This is mainly for monitoring or optional regularization.

    Note:
        This only checks anchor frame.
        If you mainly use contact/no-contact split, you can disable static loss in cfg.
    """
    anchor_ts = pred["timespan"][:, -1, :]
    hands_feet_contact = get_state_by_ts(pred["hands_feet_contact"], anchor_ts, bt_merge=False)
    is_human_contact = hands_feet_contact.sum(-1).to(torch.bool)

    if use_floor_contact:
        obj_floor_contact = get_state_by_ts(pred["obj_floor_contact"], anchor_ts, bt_merge=False)
        is_on_floor = obj_floor_contact.sum(-1).to(torch.bool)
        static_mask = is_on_floor & (~is_human_contact)
    else:
        static_mask = ~is_human_contact

    moving_contact_mask = is_human_contact
    return {"static_mask": static_mask, "moving_contact_mask": moving_contact_mask}



def get_static_object_loss(self, pred, pc_loss):
    status = get_static_status(pred, use_floor_contact=self.use_floor_contact_for_static)
    static_trans_loss = get_static_trans_loss(pred, status)
    static_rot_loss = get_static_rot_loss(pred, status)
    static_pc_loss = masked_mean(pc_loss[:, :, 0], status["static_mask"])
    static_ratio = status["static_mask"].float().mean()
    moving_contact_ratio = status["moving_contact_mask"].float().mean()

    return {
        "static_trans_loss": static_trans_loss,
        "static_rot_loss": static_rot_loss,
        "static_pc_loss": static_pc_loss,
        "static_ratio": static_ratio,
        "moving_contact_ratio": moving_contact_ratio,
    }


def get_static_trans_loss(pred, status):
    """
    Penalize predicted object translation under static states.
    pred["trans"]: [B, T', 1, 3]
    """
    pred_trans = pred["trans"][:, :, 0]
    anchor_ts = pred["timespan"][:, -1, :]
    target_ts = pred["next_timespan"][:, 0, :]
    dt = (target_ts - anchor_ts).clamp(min=1).float()

    err = torch.norm(pred_trans, p=2, dim=-1) / dt
    if status["static_mask"].sum() == 0:
        return err.sum() * 0.0

    return err[status["static_mask"]].mean()


def get_static_rot_loss(pred, status):
    """
    Penalize predicted object rotation under static states.
    For theta mode, identity rotation means no object rotation.
    pred["rot"]: [B, T', 1, 9]
    """
    pred_rot = pred["rot"][:, :, 0]
    B, T2, _ = pred_rot.shape
    anchor_ts = pred["timespan"][:, -1, :]
    target_ts = pred["next_timespan"][:, 0, :]
    dt = (target_ts - anchor_ts).clamp(min=1).float()

    if pred["method"] == "fro":
        eye = torch.eye(3, device=pred_rot.device, dtype=pred_rot.dtype).view(1, 1, 9)
        err = torch.linalg.matrix_norm((pred_rot - eye).reshape(B, T2, 3, 3), ord="fro", dim=(-2, -1)) / dt
    elif pred["method"] == "theta":
        R = pred_rot.reshape(B * T2, 3, 3)
        trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
        theta = torch.acos(torch.clamp((trace - 1) / 2, -1.0 + 1e-6, 1.0 - 1e-6)).view(B, T2)
        err = theta / dt
    else:
        raise NotImplementedError

    if status["static_mask"].sum() == 0:
        return err.sum() * 0.0

    return err[status["static_mask"]].mean()


class DynamicsLoss:
    def __init__(self, cfg):
        self.use_delta_motion = cfg["use_delta_motion"]
        self.use_higher_order = cfg["use_higher_order"]
        self.use_static_object_loss = cfg["use_static_object_loss"]
        self.use_floor_contact_for_static = cfg["use_floor_contact_for_static"]
        self.use_contact_split_loss = cfg["use_contact_split_loss"]
        self.obj_pc_loss = torch.nn.L1Loss(reduction="none")

    def get_obj_motions(self, inp, pred_trans, pred_rot, anchor_ts, target_ts, method="theta"):
        obj_motion = inp["obj_motion"]
        gt_obj_motion = get_state_by_ts(obj_motion, target_ts, bt_merge=False)

        if self.use_delta_motion:
            cur_obj_motion = get_state_by_ts(obj_motion, anchor_ts, bt_merge=False)

            if method == "fro":
                pred_obj_motion = cur_obj_motion + torch.cat([pred_trans, pred_rot], dim=-1)
            elif method == "theta":
                B, T2, _ = cur_obj_motion.shape
                pred_obj_trans = cur_obj_motion[:, :, :3] + pred_trans
                pred_obj_rot = torch.bmm(
                    cur_obj_motion[:, :, 3:].reshape(B * T2, 3, 3),
                    pred_rot.reshape(B * T2, 3, 3)
                ).view(B, T2, 9)
                pred_obj_motion = torch.cat([pred_obj_trans, pred_obj_rot], dim=-1)
            else:
                raise NotImplementedError
        else:
            pred_obj_motion = torch.cat([pred_trans, pred_rot], dim=-1)

        return pred_obj_motion, gt_obj_motion

    def get_loss(self, pred, inp):
        anchor_ts = pred["timespan"][:, -1, :]
        target_ts = pred["next_timespan"][:, 0, :]

        if self.use_delta_motion:
            gt_obj_motion = get_delta_obj_motion(
                inp["obj_motion"],
                anchor_ts,
                target_ts,
                method=pred["method"],
                ret="trans_and_rot"
            )
        else:
            gt_obj_motion = get_state_by_ts(inp["obj_motion"], target_ts, bt_merge=False)

        gt_trans = gt_obj_motion[:, :, :3]
        gt_rot = gt_obj_motion[:, :, 3:]
        pred_trans = pred["trans"][:, :, 0]
        pred_rot = pred["rot"][:, :, 0]

        trans_loss = get_trans_loss(pred_trans, gt_trans, anchor_ts, target_ts).unsqueeze(2)
        rot_loss = get_rot_loss(pred_rot, gt_rot, anchor_ts, target_ts, pred["method"]).unsqueeze(2)

        return trans_loss, rot_loss

    def get_obj_motions_and_verts(self, pred, inp, dataset):
        anchor_ts = pred["timespan"][:, -1, :]
        target_ts = pred["next_timespan"][:, 0, :]

        pred_obj_motion, gt_obj_motion = self.get_obj_motions(
            inp,
            pred["trans"][:, :, 0],
            pred["rot"][:, :, 0],
            anchor_ts,
            target_ts,
            pred["method"]
        )

        pred_obj_mesh_verts = get_obj_world_keypoints_parallel(
            pred_obj_motion,
            inp["reference_obj_rot_mat"],
            inp["rest_pose_obj_pts"],
            dataset
        )
        gt_obj_mesh_verts = get_obj_world_keypoints_parallel(
            gt_obj_motion,
            inp["reference_obj_rot_mat"],
            inp["rest_pose_obj_pts"],
            dataset
        )

        return (
            pred_obj_motion.unsqueeze(2),
            gt_obj_motion.unsqueeze(2),
            pred_obj_mesh_verts.unsqueeze(2),
            gt_obj_mesh_verts.unsqueeze(2)
        )

    def get_contact_split_loss(self, pred, trans_loss, rot_loss, pc_loss):
        """
        Split dynamics losses into contact and no-contact parts.

        trans_loss: [B, T', 1, 3]
        rot_loss:   [B, T', 1, 1]
        pc_loss:    [B, T', 1, Nv*3]
        """
        status = get_contact_status(pred)

        return {
            "contact_trans_loss": masked_mean(trans_loss[:, :, 0], status["contact_mask"]),
            "contact_rot_loss": masked_mean(rot_loss[:, :, 0], status["contact_mask"]),
            "contact_pc_loss": masked_mean(pc_loss[:, :, 0], status["contact_mask"]),

            "no_contact_trans_loss": masked_mean(trans_loss[:, :, 0], status["no_contact_mask"]),
            "no_contact_rot_loss": masked_mean(rot_loss[:, :, 0], status["no_contact_mask"]),
            "no_contact_pc_loss": masked_mean(pc_loss[:, :, 0], status["no_contact_mask"]),

            "contact_ratio": status["contact_ratio"],
            "no_contact_ratio": status["no_contact_ratio"],
        }

    def get_static_object_loss(self, pred, pc_loss):
        status = get_static_status(pred, use_floor_contact=self.use_floor_contact_for_static)
        static_trans_loss = get_static_trans_loss(pred, status)
        static_rot_loss = get_static_rot_loss(pred, status)
        static_pc_loss = masked_mean(pc_loss[:, :, 0], status["static_mask"])
        static_ratio = status["static_mask"].float().mean()
        moving_contact_ratio = status["moving_contact_mask"].float().mean()

        return {
            "static_trans_loss": static_trans_loss,
            "static_rot_loss": static_rot_loss,
            "static_pc_loss": static_pc_loss,
            "static_ratio": static_ratio,
            "moving_contact_ratio": moving_contact_ratio,
        }

    def get_higher_order_loss(self, pred, inp, dataset):
        device = pred["trans"].device
        zero = torch.tensor(0.0, device=device)

        return {
            "higher_trans_loss": zero,
            "higher_rot_loss": zero,
            "higher_pc_loss": zero,
        }
    
    def __call__(self, pred, inp, dataset, isTrain=True):
        if isinstance(dataset, torch.utils.data.dataset.Subset):
            dataset = dataset.dataset

        trans_loss, rot_loss = self.get_loss(pred, inp)
        pred_motion, gt_motion, pred_mesh_verts, gt_mesh_verts = self.get_obj_motions_and_verts(pred, inp, dataset)

        B, T2, F, _ = pred_motion.shape
        pc_loss = self.obj_pc_loss(
            pred_mesh_verts.reshape(B, T2, F, -1),
            gt_mesh_verts.reshape(B, T2, F, -1)
        )

        losses = {
            "trans_loss": trans_loss,
            "rot_loss": rot_loss,
            "pc_loss": pc_loss,
        }
        
        losses.update({"pred_world_obj_motion": pred_motion, "gt_world_obj_motion": gt_motion,})

        if self.use_contact_split_loss:
            losses.update(self.get_contact_split_loss(pred, trans_loss, rot_loss, pc_loss))

        if self.use_static_object_loss:
            losses.update(self.get_static_object_loss(pred, pc_loss))

        if self.use_higher_order:
            losses.update(self.get_higher_order_loss(pred, inp, dataset))

        return losses