from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path
import cv2
import os
import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from src.loftr import LoFTR
from src.loftr.utils.supervision import (
    compute_supervision_coarse,
    compute_supervision_fine,
)
from src.losses.loftr_loss import LoFTRLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics,
)
from src.utils.plotting import make_matching_figures
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler
from src.loftr.utils.geometry import  get_masks

def warp_coor_cells_with_depth(
    points2d,
    depth0,
    intrinsics,
    T_0to1,
):
    coor_max = np.array([depth0.shape[1], depth0.shape[0]]) - 1
    coor_pixels0 = np.concatenate((points2d, np.ones((len(points2d), 1))), axis=1)
    depth = np.array([depth0[int(y), int(x)] for (x, y) in points2d])
    coor_pixels0 *= depth.reshape(-1, 1)
    coor_cam0 = np.linalg.inv(intrinsics) @ coor_pixels0.T
    coor_cam1 = T_0to1[:3, :3] @ coor_cam0 + T_0to1[:3, 3:]
    coor_pixels1 = (intrinsics @ coor_cam1).T
    coor_pixels1 = coor_pixels1[:, :2] / coor_pixels1[:, 2:]

    # filter points
    mask = (coor_pixels1 >= 0) * (coor_pixels1 <= coor_max)
    mask = np.prod(mask, axis=-1) == 1
    validmask = depth > 0
    return points2d, coor_pixels1, mask, validmask

def drawMatches(src, tar, kpsA, kpsB, kpsC, kpsD, name, status=None, thickness=3):
    (hA, wA) = src.shape[:2]
    (hB, wB) = tar.shape[:2]
    if status == None:
        status = np.ones((kpsA.shape[0],))
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = src
    vis[0:hB, wA:] = tar
    for kp1, kp2, kt in zip(kpsA, kpsB, status):
        if kt:
            kp2 = (int(kp2[0] + wA), int(kp2[1]))
            kp1 = (int(kp1[0]), int(kp1[1]))
            cv2.line(vis, kp1, kp2, (0, 255, 0), thickness)
    for kp1, kp2, kt in zip(kpsC, kpsD, status):
        if kt:
            kp2 = (int(kp2[0] + wA), int(kp2[1]))
            kp1 = (int(kp1[0]), int(kp1[1]))
            cv2.line(vis, kp1, kp2, (0, 0, 255), thickness)
    cv2.imwrite("matching_result/" + name + ".png", vis)
    return vis


class PL_LoFTR(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.loftr_cfg = lower_config(_config["loftr"])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(
            config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1
        )

        # Matcher: LoFTR
        self.matcher = LoFTR(config=_config["loftr"])
        self.loss = LoFTRLoss(_config)

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location="cpu")["state_dict"]
            self.matcher.load_state_dict(state_dict, strict=True)
            logger.info(f"Load '{pretrained_ckpt}' as pretrained checkpoint")

        # Testing
        self.dump_dir = dump_dir

    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == "linear":
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + (
                    self.trainer.global_step / self.config.TRAINER.WARMUP_STEP
                ) * abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            elif self.config.TRAINER.WARMUP_TYPE == "constant":
                pass
            else:
                raise ValueError(
                    f"Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}"
                )

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("LoFTR"):
            self.matcher(batch)

        with self.profiler.profile("Compute fine supervision"):
            compute_supervision_fine(batch, self.config)

        with self.profiler.profile("Compute losses"):
            self.loss(batch)

    def _compute_metrics(self, batch):
        with self.profiler.profile("Copmute metrics"):
            compute_symmetrical_epipolar_errors(
                batch
            )  # compute epi_errs for each match
            compute_pose_errors(
                batch, self.config
            )  # compute R_errs, t_errs, pose_errs for each pair

            rel_pair_names = list(zip(*batch["pair_names"]))
            bs = batch["image0"].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                "identifiers": ["#".join(rel_pair_names[b]) for b in range(bs)],
                "epi_errs": [
                    batch["epi_errs"][batch["m_bids"] == b].cpu().numpy()
                    for b in range(bs)
                ],
                "R_errs": batch["R_errs"],
                "t_errs": batch["t_errs"],
                "inliers": batch["inliers"],
            }
            ret_dict = {"metrics": metrics}
        return ret_dict, rel_pair_names

    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        # logging
        if (
            self.trainer.global_rank == 0
            and self.global_step % self.trainer.log_every_n_steps == 0
        ):
            # scalars
            for k, v in batch["loss_scalars"].items():
                self.logger.experiment.add_scalar(f"train/{k}", v, self.global_step)

            # net-params
            if self.config.LOFTR.MATCH_COARSE.MATCH_TYPE == "sinkhorn":
                self.logger.experiment.add_scalar(
                    f"skh_bin_score",
                    self.matcher.coarse_matching.bin_score.clone().detach().cpu().data,
                    self.global_step,
                )

            # figures
            if self.config.TRAINER.ENABLE_PLOTTING:
                compute_symmetrical_epipolar_errors(
                    batch
                )  # compute epi_errs for each match
                figures = make_matching_figures(
                    batch, self.config, self.config.TRAINER.PLOT_MODE
                )
                for k, v in figures.items():
                    self.logger.experiment.add_figure(
                        f"train_match/{k}", v, self.global_step
                    )

        return {"loss": batch["loss"]}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                "train/avg_loss_on_epoch", avg_loss, global_step=self.current_epoch
            )

    def valcoor_pixels0idation_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        ret_dict, _ = self._compute_metrics(batch)

        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}
        if batch_idx % val_plot_interval == 0:
            figures = make_matching_figures(
                batch, self.config, mode=self.config.TRAINER.PLOT_MODE
            )

        return {
            **ret_dict,
            "loss_scalars": batch["loss_scalars"],
            "figures": figures,
        }

    def validation_epoch_end(self, outputs):
        # return
        # handle multiple validation sets
        multi_outputs = (
            [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        )
        multi_val_metrics = defaultdict(list)

        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if (
                not self.trainer.resume_from_checkpoint
                # and self.trainer.running_sanity_check
            ):
                cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o["loss_scalars"] for o in outputs]
            loss_scalars = {
                k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars]))
                for k in _loss_scalars[0]
            }

            # 2. val metrics: dict of list, numpy
            _metrics = [o["metrics"] for o in outputs]
            metrics = {
                k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics])))
                for k in _metrics[0]
            }
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0
            val_metrics_4tb = aggregate_metrics(
                metrics, self.config.TRAINER.EPI_ERR_THR
            )
            for thr in [5, 10, 20]:
                multi_val_metrics[f"auc@{thr}"].append(val_metrics_4tb[f"auc@{thr}"])

            # 3. figures
            _figures = [o["figures"] for o in outputs]
            figures = {
                k: flattenList(gather(flattenList([_me[k] for _me in _figures])))
                for k in _figures[0]
            }

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.add_scalar(
                        f"val_{valset_idx}/avg_{k}", mean_v, global_step=cur_epoch
                    )

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(
                        f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch
                    )

                for k, v in figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.logger.experiment.add_figure(
                                f"val_match_{valset_idx}/{k}/pair-{plot_idx}",
                                fig,
                                cur_epoch,
                                close=True,
                            )
            plt.close("all")

        for thr in [5, 10, 20]:
            # log on all ranks for ModelCheckpoint callback to work properly
            self.log(
                f"auc@{thr}", torch.tensor(np.mean(multi_val_metrics[f"auc@{thr}"]))
            )  # ckpt monitors on this

    def test_step(self, batch, batch_idx):
        with self.profiler.profile("LoFTR"):
            self.matcher(batch)

        os.makedirs("matching_result/" + batch["dataset_name"][0], exist_ok=True)
        matches_points0 = batch["mkpts0_f"][::10].cpu().numpy()
        matches_points1 = batch["mkpts1_f"][::10].cpu().numpy()
        matches_points0, matches_points1_gt, mask, validmask = warp_coor_cells_with_depth(
            matches_points0,
            batch["depth0"][0].cpu().numpy(),
            batch["K0"][0].cpu().numpy(),
            batch["T_0to1"][0].cpu().numpy(),
        )
        right_mask = mask * (np.linalg.norm(matches_points1_gt - matches_points1, axis=1) <= 9)
        wrong_mask = ~right_mask

        drawMatches(
            batch["image0_ori"][0].permute(1, 2, 0).cpu().numpy() * 255,
            batch["image1_ori"][0].permute(1, 2, 0).cpu().numpy() * 255,
            matches_points0[right_mask * validmask],
            matches_points1[right_mask * validmask],
            matches_points0[wrong_mask * validmask],
            matches_points1[wrong_mask * validmask],
            batch["dataset_name"][0] + "/" + str(batch_idx) + "_loftr",
            None,
            1,
        )

        ret_dict, rel_pair_names = self._compute_metrics(batch)
        # 左侧遮挡右侧的情况
        valid_mask0, occluded_mask0, non_occluded_mask0, w_kpts0 = \
            get_masks(
                batch["mkpts0_f"][None], 
                batch["depth0"], 
                batch["depth1"], 
                batch["T_0to1"], 
                batch["K0"], 
                batch["K1"],
                )
        # 整张图的左侧遮挡右侧的比例
        valid_mask1, occluded_mask1, non_occluded_mask1, w_kpts1 = \
            get_masks(
                batch["mkpts1_f"][None], 
                batch["depth1"], 
                batch["depth0"], 
                batch["T_1to0"], 
                batch["K1"], 
                batch["K0"],
                )        
        # 整张图还有右侧遮挡左侧
        H, W = batch['depth0'].shape[1:]
        y_position = torch.ones((H, W)).cumsum(0).float().unsqueeze(0) - 1
        x_position = torch.ones((H, W)).cumsum(1).float().unsqueeze(0) - 1
        pts = torch.stack([x_position.view(-1), y_position.view(-1)], dim=-1)[None].to(batch['mkpts0_f'].device)
        valid_maskw0, occluded_maskw0, non_occluded_maskw0, w_kptsw0 = \
            get_masks(
                pts, 
                batch["depth0"], 
                batch["depth1"], 
                batch["T_0to1"], 
                batch["K0"], 
                batch["K1"],
                )        
        valid_maskw1, occluded_maskw1, non_occluded_maskw1, w_kptsw1 = \
            get_masks(
                pts, 
                batch["depth1"], 
                batch["depth0"], 
                batch["T_1to0"], 
                batch["K1"], 
                batch["K0"],
                )   
        batch['mkpts0_f'] = batch['mkpts0_f'][valid_mask0.reshape(-1) * valid_mask1.reshape(-1), :]
        batch['mkpts1_f'] = batch['mkpts1_f'][valid_mask0.reshape(-1) * valid_mask1.reshape(-1), :]
        batch['m_bids'] = batch['m_bids'][valid_mask0.reshape(-1) * valid_mask1.reshape(-1)]
        batch['mconf'] = batch['mconf'][valid_mask0.reshape(-1) * valid_mask1.reshape(-1)]
        # batch['mkpts0_f'] = np.
        ret_dict, rel_pair_names = self._compute_metrics(batch)
        with self.profiler.profile("dump_results"):
            if self.dump_dir is not None:
                # dump results for further analysis
                keys_to_save = {"mkpts0_f", "mkpts1_f", "mconf", "epi_errs"}
                pair_names = list(zip(*batch["pair_names"]))
                bs = batch["image0"].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch["m_bids"] == b_id
                    item["pair_names"] = pair_names[b_id]
                    item["identifier"] = "#".join(rel_pair_names[b_id])
                    for key in keys_to_save:
                        item[key] = batch[key][mask].cpu().numpy()
                    for key in ["R_errs", "t_errs", "inliers"]:
                        item[key] = batch[key][b_id]
                    item['length'] = len(batch["mkpts0_f"].cpu().numpy())
                    item['occlusion_rate_left'] = (occluded_maskw0.sum() / valid_maskw0.sum()).cpu().numpy()
                    item['matching_occlusion_rate_left'] = (occluded_mask0.sum() / valid_mask0.sum()).cpu().numpy()
                    item['occlusion_rate_right'] = (occluded_maskw1.sum() / valid_maskw1.sum()).cpu().numpy()
                    item['matching_occlusion_rate_right'] = (occluded_mask1.sum() / valid_mask1.sum()).cpu().numpy()                    
                    dumps.append(item)
                ret_dict["dumps"] = dumps
        return ret_dict

    def test_epoch_end(self, outputs):
        # metrics: dict of list, numpy
        _metrics = [o["metrics"] for o in outputs]
        metrics = {
            k: flattenList(gather(flattenList([_me[k] for _me in _metrics])))
            for k in _metrics[0]
        }

        # [{key: [{...}, *#bs]}, *#batch]
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o["dumps"] for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(
                f"Prediction and evaluation results will be saved to: {self.dump_dir}"
            )

        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(
                metrics, self.config.TRAINER.EPI_ERR_THR
            )
            logger.info("\n" + pprint.pformat(val_metrics_4tb))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / "LoFTR_pred_eval", dumps)
