import os
import copy
import argparse
import tqdm
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from model import Dynamics
from loss import DynamicsLoss
from utils import read_yaml, mkdirs
from tools import (
    prepare_train_val_dataset_by_randsplit,
    prepare_dataset,
    move_data_to_device,
    get_state_by_ts,
    get_delta_obj_motion,
    get_world_object_geo,
    extract_rotation_euler_batch,
)


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

loss_w_t = 0.0
loss_w_r = 0.0
loss_w_pc = 1000.0


def tensor_to_float(x):
    if isinstance(x, torch.Tensor):
        return x.detach().item()
    return float(x)


def reduce_with_mask(x, mask):
    if not isinstance(x, torch.Tensor):
        return x
    if x.dim() == 0:
        return x
    return (x * mask).mean()


def get_training_loss(
    converted_losses,
    progress=1.0,
    use_balanced_contact_pc_loss=True,
):
    if use_balanced_contact_pc_loss:
        pc_loss = (
            converted_losses["pc_loss"]
            # + converted_losses["contact_pc_loss"]
            + converted_losses["no_contact_pc_loss"]
        )
    else:
        pc_loss = converted_losses["pc_loss"]

    return loss_w_pc * pc_loss


def get_total_loss(losses, mask, progress=1.0):
    converted_losses = {
        "trans_loss": reduce_with_mask(losses["trans_loss"], mask),
        "rot_loss": reduce_with_mask(losses["rot_loss"], mask),
        "pc_loss": reduce_with_mask(losses["pc_loss"], mask),
    }

    optional_keys = [
        "higher_trans_loss", "higher_rot_loss", "higher_pc_loss",
        "contact_trans_loss", "contact_rot_loss", "contact_pc_loss",
        "no_contact_trans_loss", "no_contact_rot_loss", "no_contact_pc_loss",
        "contact_ratio", "no_contact_ratio",
        "static_trans_loss", "static_rot_loss", "static_pc_loss",
        "static_ratio", "moving_contact_ratio",
    ]

    for k in optional_keys:
        if k in losses:
            converted_losses[k] = losses[k]

    total_loss = get_training_loss(converted_losses, progress)
    return total_loss, converted_losses


def compose_loss_string(losses, progress=1.0):
    total_loss = get_training_loss(losses, progress).item()
    s = (
        f"Loss [Total {total_loss:.4f}, "
        f"R {losses['rot_loss'].item():.6f}, "
        f"T {losses['trans_loss'].item():.6f}, "
        f"PC {losses['pc_loss'].item():.6f}"
    )

    if "contact_pc_loss" in losses:
        s += (
            f", C-PC {tensor_to_float(losses['contact_pc_loss']):.6f}, "
            f"NC-PC {tensor_to_float(losses['no_contact_pc_loss']):.6f}, "
            f"C-ratio {tensor_to_float(losses['contact_ratio']):.3f}"
        )

    if "static_pc_loss" in losses:
        s += (
            f", Static-PC {tensor_to_float(losses['static_pc_loss']):.6f}, "
            f"Static-ratio {tensor_to_float(losses['static_ratio']):.3f}"
        )

    if "higher_pc_loss" in losses:
        s += f", HPC {tensor_to_float(losses['higher_pc_loss']):.6f}"

    s += "]"
    return s


def average_metric_dict(metric_dicts):
    avg = {}
    if len(metric_dicts) == 0:
        return avg

    skip_keys = {"pred_world_obj_motion", "gt_world_obj_motion"}
    keys = metric_dicts[0].keys()

    for k in keys:
        if k in skip_keys:
            continue
        vals = []
        for m in metric_dicts:
            if k not in m:
                continue
            v = m[k]
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                continue
            vals.append(tensor_to_float(v))
        if len(vals) > 0:
            avg[k] = float(np.mean(vals))

    return avg


def resolve_ckpt(cfg):
    return cfg["ckpt"] if cfg["ckpt"] is not None else "best.pth"


class DynamicsTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.g_exp_name = cfg["name"]
        self.g_exp_config = cfg["model"]
        self.device = torch.device(f"cuda:{cfg['gpu']}")
        self.g_mode = cfg["mode"]
        self.g_resume = cfg["resume"]
        self.g_num_epochs = cfg["epochs"]
        self.g_runs_record_root = cfg["save_dir"]
        self.g_test_batch_size = cfg["test_batch_size"]
        self.g_train_batch_size = cfg["train_batch_size"]
        self.g_val_batch_size = cfg["val_batch_size"]
        self.g_init_lr = cfg["lr"]
        self.g_start_epoch = 0
        self.use_tb = cfg["use_tb"]

        self.dst_root = mkdirs(os.path.join(self.g_runs_record_root, self.g_exp_name))
        self.g_model_save_root = os.path.join(self.dst_root, "best.pth")
        self.vis_save_dir = self.resolve_vis_save_dir(cfg)
        os.makedirs(self.vis_save_dir, exist_ok=True)

        if self.use_tb:
            self.writer = SummaryWriter(self.dst_root)

        if "train" in self.g_mode or "val" in self.g_mode:
            self.train_dataset, self.train_loader, self.val_dataset, self.val_loader = prepare_train_val_dataset_by_randsplit(
                train_batch_size=self.g_train_batch_size, val_batch_size=self.g_val_batch_size, val_ratio=0.1
            )

        if "test" in self.g_mode:
            test_bs = 1 if "ar" in self.g_mode or "autoregressive" in self.g_mode else self.g_test_batch_size
            self.test_dataset, self.test_loader = prepare_dataset(train=False, batch_size=test_bs)

        self.model = Dynamics(cfg=self.g_exp_config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.g_init_lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=cfg["T0"], T_mult=cfg["T_mult"], eta_min=cfg["eta_min"]
        )
        self.multi_loss = DynamicsLoss(self.g_exp_config)

        if self.g_resume and "train" in self.g_mode:
            self.g_start_epoch = self.load_model(ckpt_path=cfg["ckpt"])

    def resolve_vis_save_dir(self, cfg):
        if "vis_save_dir" in cfg and cfg["vis_save_dir"] is not None:
            return cfg["vis_save_dir"]

        ckpt = resolve_ckpt(cfg)
        if os.path.isabs(ckpt):
            return os.path.dirname(ckpt)
        return self.dst_root

    def train(self):
        log_every_n_interval = self.g_train_batch_size * 100
        iter_cnt = 0
        best_val_loss = 1e6

        for epoch in tqdm.tqdm(range(self.g_start_epoch, self.g_num_epochs), desc="Training"):
            self.model.train()

            for batch_idx, inp in tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader), colour="green", leave=False):
                inp = move_data_to_device(inp, self.device)
                iter_cnt += self.g_train_batch_size

                pred = self.model(inp, self.train_dataset, max_step=self.g_exp_config["max_step"])
                losses = self.multi_loss(pred, inp, self.train_dataset, isTrain=True)
                mask = self.get_padding_mask(inp, pred)
                loss, converted_losses = get_total_loss(losses, mask=mask, progress=epoch / self.g_num_epochs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if iter_cnt % log_every_n_interval == 0:
                    self.monitor_metrics(converted_losses, epoch, iter_cnt, phase="train")

            val_metrics = self.evaluate(self.val_loader, self.val_dataset, max_step=self.g_exp_config["max_step"], desc="Validation")
            self.monitor_metrics(val_metrics, epoch, iter_cnt, phase="val")

            val_loss = val_metrics["total_loss"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(self.g_model_save_root, epoch)

            self.save_model(self.g_model_save_root.replace("best.pth", "current.pth"), epoch)

            if hasattr(self, "test_loader"):
                test_metrics = self.evaluate(self.test_loader, self.test_dataset, max_step=1, desc="Testing")
                self.monitor_metrics(test_metrics, epoch, iter_cnt, phase="test")

            self.scheduler.step()

    def evaluate(self, inp_loader, inp_dataset, max_step=1, desc="Evaluation"):
        self.model.eval()
        metric_traj = []

        with torch.no_grad():
            for batch_idx, inp in enumerate(inp_loader):
                inp = move_data_to_device(inp, self.device)
                pred = self.model(inp, inp_dataset, max_step=max_step)
                losses = self.multi_loss(pred, inp, inp_dataset, isTrain=False)
                mask = self.get_padding_mask(inp, pred)
                total_loss, converted_losses = get_total_loss(losses, mask=mask)
                converted_losses["total_loss"] = total_loss
                metric_traj.append(converted_losses)

        avg_metrics = average_metric_dict(metric_traj)
        print(f"[{desc}] total_loss: {avg_metrics['total_loss']:.6f}, pc_loss: {avg_metrics['pc_loss']:.6f}, pc_loss(cm): {avg_metrics['pc_loss'] * 100:.4f}")
        return avg_metrics

    def monitor_metrics(self, metrics, epoch, iter_cnt, phase):
        if phase == "train":
            lr = self.optimizer.param_groups[0]["lr"]
            progress = epoch / self.g_num_epochs
            print(
                f"[Train] Epoch {epoch}/{self.g_num_epochs}, "
                f"Iter {iter_cnt}/{len(self.train_dataset) * self.g_num_epochs}, "
                f"LR {lr:.6e}, " + compose_loss_string(metrics, progress=progress)
            )
            if self.use_tb:
                total_loss = get_training_loss(metrics, progress=progress).item()
                self.writer.add_scalar("Loss/train_total_loss", total_loss, iter_cnt)
                self.write_metrics_to_tb(metrics, iter_cnt, prefix="Train")

        elif phase == "val":
            print(f"[Val]   Epoch {epoch}/{self.g_num_epochs}, Total {metrics['total_loss']:.6f}, PC {metrics['pc_loss']:.6f}")
            if self.use_tb:
                self.write_metrics_to_tb(metrics, epoch, prefix="Val")

        elif phase == "test":
            print(f"[Test]  Epoch {epoch}/{self.g_num_epochs}, Total {metrics['total_loss']:.6f}, PC {metrics['pc_loss']:.6f}")
            if self.use_tb:
                self.write_metrics_to_tb(metrics, epoch, prefix="Test")

        else:
            raise ValueError(f"Unknown phase: {phase}")

    def write_metrics_to_tb(self, metrics, step, prefix):
        for k, v in metrics.items():
            if k in ["pred_world_obj_motion", "gt_world_obj_motion"]:
                continue
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                continue
            self.writer.add_scalar(f"{prefix}/{k}", tensor_to_float(v), step)

    def get_padding_mask(self, inp, pred):
        actual_seq_len = inp["seq_len"]
        history_ts = pred["timespan"]
        future_ts = pred["next_timespan"]
        B, F, T2 = future_ts.shape

        anchor_ts = history_ts[:, -1, :]
        padding_mask_f = []
        for i in range(F):
            target_ts = future_ts[:, i, :]
            m1 = anchor_ts < actual_seq_len[:, None].repeat(1, T2)
            m2 = target_ts < actual_seq_len[:, None].repeat(1, T2)
            m = m1 & m2
            padding_mask_f.append(m.unsqueeze(-1).unsqueeze(2))

        return torch.cat(padding_mask_f, dim=2)

    def save_model(self, ckpt_path, epoch):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
        }, ckpt_path)
        print(f"[Checkpoint] Model saved to '{ckpt_path}' at epoch {epoch}.")

    def load_model(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint.get("epoch", 0)
        print(f"[Checkpoint] Checkpoint '{ckpt_path}' @ {epoch} loaded successfully.")
        return epoch

    def load_model_for_eval(self, model_to_load="best.pth"):
        ckpt_path = model_to_load if os.path.isabs(model_to_load) else os.path.join(self.dst_root, model_to_load)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "NO_EPOCH")
        print(f"[Checkpoint] Eval model loaded from '{ckpt_path}' @ epoch {epoch}.")
        return epoch

    def test(self, model_to_load="best.pth", vis=False, save_pc4vis=False, max_step=1):
        self.load_model_for_eval(model_to_load)
        self.model.eval()

        metric_traj = []
        gt_obj_motion_traj = []
        pred_obj_motion_traj = []
        inp_obj_motion_traj = []

        with torch.no_grad():
            for batch_idx, inp in enumerate(self.test_loader):
                inp = move_data_to_device(inp, self.device)
                pred = self.model(inp, self.test_dataset, max_step=max_step)
                losses = self.multi_loss(pred, inp, self.test_dataset, isTrain=False)
                mask = self.get_padding_mask(inp, pred)
                total_loss, converted_losses = get_total_loss(losses, mask=mask)
                converted_losses["total_loss"] = total_loss
                metric_traj.append(converted_losses)

                gt_obj_motion_traj.append(get_delta_obj_motion(inp["obj_motion"], pred["timespan"][:, -1, :], pred["next_timespan"][:, 0, :], method=pred["method"], ret="trans_and_rot"))
                pred_obj_motion_traj.append(torch.cat([pred["trans"][:, :, 0], pred["rot"][:, :, 0]], dim=-1))
                inp_obj_motion_traj.append(get_state_by_ts(inp["obj_motion"], pred["timespan"][:, -1, :], bt_merge=False))

                if save_pc4vis:
                    self.save_pc_for_vis(inp, losses["pc_loss"], losses["pred_world_obj_motion"], losses["gt_world_obj_motion"])

        avg_metrics = average_metric_dict(metric_traj)
        print(f"[Test] total_loss: {avg_metrics['total_loss']:.6f}, pc_loss(cm): {avg_metrics['pc_loss'] * 100:.4f}")

        gt_obj_motion_traj = torch.cat(gt_obj_motion_traj, dim=0)
        pred_obj_motion_traj = torch.cat(pred_obj_motion_traj, dim=0)
        inp_obj_motion_traj = torch.cat(inp_obj_motion_traj, dim=0)

        if vis:
            save_path = os.path.join(self.vis_save_dir, "test_vis_t3r3.png")
            self.vis_t3r3(gt_obj_motion_traj, pred_obj_motion_traj, save_path=save_path)

        return avg_metrics

    def test_auto_regressive(self, model_to_load="best.pth", max_step=1, vis=True, save_pc4vis=False):
        self.load_model_for_eval(model_to_load)
        self.model.eval()

        gt_obj_motion_traj = []
        pred_obj_motion_traj = []

        with torch.no_grad():
            for batch_idx, inp in enumerate(self.test_loader):
                inp = move_data_to_device(inp, self.device)
                pred_motion, gt_motion = self.auto_regressive_rollout(inp, self.test_dataset, max_step=max_step)
                pred_obj_motion_traj.append(pred_motion)
                gt_obj_motion_traj.append(gt_motion)

                if save_pc4vis:
                    self.save_ar_pc_for_vis(inp, pred_motion, gt_motion)

        pred_obj_motion_traj = torch.cat(pred_obj_motion_traj, dim=0)
        gt_obj_motion_traj = torch.cat(gt_obj_motion_traj, dim=0)

        if vis:
            save_path = os.path.join(self.vis_save_dir, "ar_vis_t3r3_nice.png")
            self.vis_t3r3_nice(gt_obj_motion_traj, pred_obj_motion_traj, save_path=save_path)

        return pred_obj_motion_traj, gt_obj_motion_traj

    def auto_regressive_rollout(self, inp, dataset, max_step=1):
        H = self.g_exp_config["history_win"]
        S = max_step
        B, T, _ = inp["obj_motion"].shape
        assert B == 1, "Autoregressive test is recommended with batch_size=1."
        assert self.g_exp_config["future_win"] == 1, "AR rollout assumes future_win=1."

        pred_obj_motion = inp["obj_motion"].clone()
        gt_obj_motion = inp["obj_motion"].clone()

        max_start = T - H - S + 1
        for start in range(max_start):
            target_idx = start + H - 1 + S
            state = self.slice_temporal_batch(inp, start, start + H + S)
            state = copy.deepcopy(state)
            state["obj_motion"][:, :H] = pred_obj_motion[:, start:start + H]

            pred = self.model(state, dataset, max_step=S)
            anchor_motion = pred_obj_motion[:, start + H - 1]
            delta_trans = pred["trans"][:, 0, 0]
            delta_rot = pred["rot"][:, 0, 0]
            cur_trans = anchor_motion[:, :3] + delta_trans

            if pred["method"] == "theta":
                cur_rot = torch.bmm(anchor_motion[:, 3:].reshape(B, 3, 3), delta_rot.reshape(B, 3, 3)).reshape(B, 9)
            else:
                cur_rot = anchor_motion[:, 3:] + delta_rot

            pred_obj_motion[:, target_idx] = torch.cat([cur_trans, cur_rot], dim=-1)

        return pred_obj_motion, gt_obj_motion

    def slice_temporal_batch(self, inp, start, end):
        out = {}
        T = inp["obj_motion"].shape[1]

        for k, v in inp.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[1] == T:
                out[k] = v[:, start:end].clone()
            else:
                out[k] = v

        if "seq_len" in out:
            out["seq_len"] = torch.clamp(out["seq_len"] - start, min=0, max=end - start)

        return out

    def save_pc_for_vis(self, inp, batch_loss, pred_obj_motion, gt_obj_motion, save_root="./plugin/dynamics/world_pc"):
        os.makedirs(save_root, exist_ok=True)
        batch_mean = batch_loss.mean(dim=(1, 2, 3)).detach().cpu().numpy()
        worst_idx = int(np.argmax(batch_mean))

        pred_world_pc = get_world_object_geo([inp["obj_name"][worst_idx]], pred_obj_motion[worst_idx, None, :, 0], inp["reference_obj_rot_mat"][worst_idx, None], self.test_dataset)[0]
        gt_world_pc = get_world_object_geo([inp["obj_name"][worst_idx]], gt_obj_motion[worst_idx, None, :, 0], inp["reference_obj_rot_mat"][worst_idx, None], self.test_dataset)[0]

        seq_name = inp["seq_name"][worst_idx] + "_" + str(np.round(batch_mean[worst_idx], 6))
        save_path = os.path.join(save_root, seq_name)
        np.save(save_path, np.stack([pred_world_pc.cpu().numpy(), gt_world_pc.cpu().numpy()]))

    def save_ar_pc_for_vis(self, inp, pred_obj_motion, gt_obj_motion, save_root="./plugin/dynamics/world_pc_ar"):
        os.makedirs(save_root, exist_ok=True)
        inp_cpu = move_data_to_device(inp, torch.device("cpu"))
        pred_obj_motion = pred_obj_motion.cpu()
        gt_obj_motion = gt_obj_motion.cpu()

        pred_world_pc = get_world_object_geo([inp_cpu["obj_name"][0]], pred_obj_motion[0, None], inp_cpu["reference_obj_rot_mat"][0, None], self.test_dataset)[0]
        gt_world_pc = get_world_object_geo([inp_cpu["obj_name"][0]], gt_obj_motion[0, None], inp_cpu["reference_obj_rot_mat"][0, None], self.test_dataset)[0]

        loss = torch.abs(pred_world_pc - gt_world_pc).mean().item()
        seq_name = inp_cpu["seq_name"][0] + "_" + str(np.round(loss, 6))
        save_path = os.path.join(save_root, seq_name)
        np.save(save_path, np.stack([pred_world_pc.cpu().numpy(), gt_world_pc.cpu().numpy()]))

    def vis_tr_diff(self, gt_obj_motion_traj, pred_obj_motion_traj, method="theta", save_path=None):
        gt = gt_obj_motion_traj.mean(0).cpu()
        pred = pred_obj_motion_traj.mean(0).cpu()
        fig, axes = plt.subplots(5, 1, figsize=(8, 8), sharex=True)

        trans_diff = pred[:, :3] - gt[:, :3]
        delta_trans = torch.norm(trans_diff, p=2, dim=-1)

        if method == "fro":
            diff = gt[:, 3:] - pred[:, 3:]
            delta_rot = torch.norm(diff.reshape(-1, 3, 3), p="fro", dim=(-2, -1))
        elif method == "theta":
            R_gt = gt[:, 3:].reshape(-1, 3, 3)
            R = pred[:, 3:].reshape(-1, 3, 3)
            diff = torch.bmm(R.transpose(1, 2), R_gt)
            trace = torch.diagonal(diff, dim1=-2, dim2=-1).sum(-1)
            delta_rot = torch.acos(torch.clamp((trace - 1) / 2, -1.0 + 1e-6, 1.0 - 1e-6))
        else:
            raise NotImplementedError

        for i in range(3):
            axes[i].plot(trans_diff[:, i], linewidth=2)
            axes[i].set_ylabel(f"T-Diff {i + 1}")
            axes[i].grid(True)

        axes[3].plot(delta_trans, linewidth=2)
        axes[3].set_ylabel("Overall T-Diff")
        axes[3].grid(True)

        axes[4].plot(delta_rot / 3.14 * 180, linewidth=2)
        axes[4].set_ylabel("Overall R-Diff")
        axes[4].grid(True)
        axes[-1].set_xlabel("Frame")
        plt.tight_layout()

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[Vis] Figure saved to {save_path}")
            plt.close(fig)
        else:
            plt.show()

    def vis_t3r3(self, gt_obj_motion_traj, pred_obj_motion_traj, save_path=None):
        gt = gt_obj_motion_traj.mean(0).cpu()
        pred = pred_obj_motion_traj.mean(0).cpu()

        gt[:, 3:6] = torch.tensor(extract_rotation_euler_batch(gt[:, 3:]))
        pred[:, 3:6] = torch.tensor(extract_rotation_euler_batch(pred[:, 3:]))

        fig, axes = plt.subplots(6, 1, figsize=(8, 8), sharex=True)
        labels = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]

        for i in range(6):
            axes[i].plot(pred[:, i], label="Pred", linewidth=2)
            axes[i].plot(gt[:, i], label="GT", linewidth=2)
            axes[i].set_ylabel(labels[i])
            axes[i].legend(loc="upper right")
            axes[i].grid(True)

        axes[-1].set_xlabel("Frame")
        plt.tight_layout()

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[Vis] Figure saved to {save_path}")
            plt.close(fig)
        else:
            plt.show()

    def vis_t3r3_nice(self, gt_obj_motion_traj, pred_obj_motion_traj, save_path=None):
        import matplotlib as mpl
        mpl.rcParams["font.size"] = 11

        gt = gt_obj_motion_traj.mean(0).cpu()
        pred = pred_obj_motion_traj.mean(0).cpu()

        gt[:, 3:6] = torch.tensor(extract_rotation_euler_batch(gt[:, 3:]))
        pred[:, 3:6] = torch.tensor(extract_rotation_euler_batch(pred[:, 3:]))

        colors = {"pred": "#1f77b4", "gt": "#2ca02c"}
        labels = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]

        fig, axes = plt.subplots(6, 1, figsize=(10, 10), sharex=True)

        for i in range(6):
            axes[i].plot(pred[:, i], label="Prediction", color=colors["pred"], linewidth=2)
            axes[i].plot(gt[:, i], label="Ground Truth", color=colors["gt"], linewidth=2)
            axes[i].set_ylabel(labels[i])
            axes[i].grid(True, linestyle="--", alpha=0.5)
            axes[i].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)

        axes[-1].set_xlabel("Frame")
        fig.suptitle("Object Motion Trajectory: Translation and Rotation", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.subplots_adjust(right=0.83, top=0.92)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[Vis] Figure saved to {save_path}")
            plt.close(fig)
        else:
            plt.show()


def main(cfg_path):
    cfg = read_yaml(cfg_path)
    trainer = DynamicsTrainer(cfg)
    ckpt = resolve_ckpt(cfg)

    if cfg["mode"] == "train-val-test":
        trainer.train()
    elif cfg["mode"] == "test":
        metrics = trainer.test(model_to_load=ckpt, vis=True, save_pc4vis=False, max_step=cfg["model"]["max_step"])
        print(metrics)
    elif cfg["mode"] == "val":
        trainer.load_model_for_eval(ckpt)
        metrics = trainer.evaluate(trainer.val_loader, trainer.val_dataset, max_step=cfg["model"]["max_step"], desc="Validation")
        print(metrics)
    elif cfg["mode"] in ["test_ar", "test-autoregressive"]:
        trainer.test_auto_regressive(model_to_load=ckpt, max_step=1, vis=True, save_pc4vis=False)
    else:
        raise ValueError(f"Unknown mode: {cfg['mode']}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="cfg/test.yaml", help="Path to the yaml config file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(cfg_path=args.cfg)