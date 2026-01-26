#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm
import pickle

import torch.nn.functional as F

from encoders.feature_extractor import FeatureExtractor
from arguments import ModelParams, OptimizationParams, get_combined_args
from gaussian_renderer import get_render_visible_mask, render_gsplat
from scene import Scene
from scene.gaussian_model import GaussianModel
from scene.salient_sample_detector import SalientSampleDetector
from utils.general_utils import safe_state, seed_everything
from utils.graphics_utils import focal2fov, fov2focal
from utils.image_utils import get_resolution_from_longest_edge
from utils.loss_utils import *

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


@torch.no_grad()
def calculate_match_score(
        gaussians: GaussianModel,
        gt_feature_map,
        pose,
        K,
        render_visible_mask=None,
        img_mask=None,
):
    xyz = gaussians.get_xyz
    feat = gaussians.get_loc_feature.squeeze()

    xyz_homo = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=xyz.device)], dim=-1)
    xyz_cam = (pose @ xyz_homo.T)[:3]
    depths = xyz_cam[2]
    xyz_cam_homo = xyz_cam / depths

    xy = (K @ xyz_cam_homo)[:2].long()

    in_mask = (
            (xy[0] >= 0)
            & (xy[0] < gt_feature_map.shape[2])
            & (xy[1] >= 0)
            & (xy[1] < gt_feature_map.shape[1])
    )

    if render_visible_mask is not None:
        visible_mask = in_mask & render_visible_mask
    else:
        visible_mask = in_mask

    if img_mask is not None:
        visible_xy = xy[:, in_mask]
        img_mask_expand = torch.zeros_like(visible_mask, dtype=torch.bool)
        img_mask_expand[in_mask] = img_mask[0, visible_xy[1], visible_xy[0]]
        visible_mask = visible_mask & img_mask_expand

    xy = xy[:, visible_mask]
    depths = depths[visible_mask]
    feat = feat[visible_mask]

    gs_feats = F.normalize(feat, p=2, dim=1)
    im_feats = gt_feature_map[:, xy[1], xy[0]].T
    score = (gs_feats * im_feats).sum(-1)
    return score, visible_mask


def generate_gt_map(
        points3d,
        gt_feature_map,
        idx_sampled,
        pose,
        K,
        render_visible_mask=None,
):
    if render_visible_mask is not None:
        render_visible_mask = render_visible_mask[idx_sampled]

        idx_sampled = idx_sampled[render_visible_mask]
    sampled_xyz = points3d[idx_sampled]

    gt_map = torch.zeros(
        (1, gt_feature_map.shape[1], gt_feature_map.shape[2]),
        device=gt_feature_map.device,
    )

    xyz_homo = torch.cat(
        [sampled_xyz, torch.ones(sampled_xyz.shape[0], 1, device=sampled_xyz.device)],
        dim=-1,
    )
    xyz_cam = (pose @ xyz_homo.T)[:3]
    depths = xyz_cam[2]
    xyz_cam_norm = xyz_cam / depths
    xy = (K @ xyz_cam_norm)[:2].long()

    in_mask = (
            (xy[0] >= 0)
            & (xy[0] < gt_feature_map.shape[2])
            & (xy[1] >= 0)
            & (xy[1] < gt_feature_map.shape[1])
    )

    xy_pos = xy[:, in_mask]
    gt_map[:, xy_pos[1], xy_pos[0]] = 1
    return gt_map


def random_block_knn_score(points3d, num, score, k, block=1024):
    sample_idx = torch.randperm(points3d.shape[0])[:num]  # [num]
    sample_points3d = points3d[sample_idx]  # [num, 3]

    points3d_cpu = points3d.cpu()
    sample_points3d_cpu = sample_points3d.cpu()
    all_knn_idx = torch.empty((num, k), dtype=torch.long)

    for start in tqdm(range(0, num, block), desc="Salient Sample Gaussians Selection..."):
        end = min(start + block, num)
        q_batch = sample_points3d_cpu[start:end]  # [B, 3]

        # [B, N]
        dist = torch.cdist(q_batch, points3d_cpu)

        # [B, k]
        knn_idx = torch.topk(dist, k, largest=False, dim=-1)[1]
        all_knn_idx[start:end] = knn_idx

    all_knn_idx = all_knn_idx.cuda()

    knn_score = score[all_knn_idx]  # [num, k]

    knn_score_sort_idx = torch.argsort(knn_score, descending=True, dim=-1)  # [num, k]

    final_sample_idx = set()
    for i in range(num):
        for j in knn_score_sort_idx[i]:
            idx = all_knn_idx[i, j].item()
            if idx not in final_sample_idx:
                final_sample_idx.add(idx)
                break

    return torch.tensor(list(final_sample_idx)).cuda()


def matching_oriented_sample(
        scene,
        gaussians,
        feature_extractor,
        render_visible_masks,
        masks=None,
        num=16384,
        k=32,
):
    viewpoint_stack = scene.getTrainCameras().copy()

    score_sum = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.float32, device="cuda")
    score_num = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.int, device="cuda")
    fine_resolution = (
        480, 640
    )

    for viewpoint_cam in tqdm(viewpoint_stack, desc="Match Score"):
        gt_image = viewpoint_cam.original_image.cuda()
        gt_image = F.interpolate(
            gt_image.unsqueeze(0),
            size=(fine_resolution[0], fine_resolution[1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        gt_feature_map = feature_extractor(gt_image[None])["feature_map"]
        gt_feature_map = F.interpolate(
            gt_feature_map,
            size=(fine_resolution[0], fine_resolution[1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        gt_feature_map = F.normalize(gt_feature_map, p=2, dim=0)

        viewmat = viewpoint_cam.world_view_transform.transpose(0, 1).cuda()  # [4, 4]
        focalX = fov2focal(viewpoint_cam.FoVx, gt_feature_map.shape[2])
        focalY = fov2focal(viewpoint_cam.FoVy, gt_feature_map.shape[1])
        K = torch.tensor(
            [
                [focalX, 0.0, gt_feature_map.shape[2] / 2],
                [0.0, focalY, gt_feature_map.shape[1] / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )

        if render_visible_masks.get(viewpoint_cam.image_name, None) is None:
            render_visible_mask = get_render_visible_mask(
                gaussians,
                viewpoint_cam,
                gt_feature_map.shape[2],
                gt_feature_map.shape[1],
            )
            render_visible_masks[viewpoint_cam.image_name] = render_visible_mask

        if masks is not None:
            object_mask = masks[viewpoint_cam.image_name][0].cuda()[None]
            distort_mask = masks[viewpoint_cam.image_name][2].cuda()[None]
            mask = object_mask & distort_mask
            img_mask = (
                    F.interpolate(
                        mask[None].float(),
                        size=(gt_feature_map.shape[1], gt_feature_map.shape[2]),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    > 0.5
            )
        else:
            img_mask = None

        score, mask = calculate_match_score(
            gaussians,
            gt_feature_map,
            viewmat,
            K,
            render_visible_mask=render_visible_masks[viewpoint_cam.image_name],
            img_mask=img_mask,
        )
        score_num[mask] += 1
        score_sum[mask] += score

    score_num[score_num == 0] = 1
    score_avg = score_sum / score_num

    sampled_idx = random_block_knn_score(gaussians.get_xyz, num, score_avg, k=k)
    sampled_idx = torch.unique(sampled_idx)
    return sampled_idx, score_avg, score_num, render_visible_masks


def training_detector_multi(scene_list, args, saving_iterations, tb_writer, train_iteration=30000):
    print("Training detector...")
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    feature_extractor = FeatureExtractor(args.feature_type)
    detector = SalientSampleDetector(feature_extractor.feature_dim).cuda().train()
    optimizer = torch.optim.AdamW(detector.parameters(), lr=0.001)
    grad_accum = 8
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_iteration // grad_accum, eta_min=0.0005)

    m_points3d = []
    m_cameras = []
    m_idx = []
    m_render_masks = []
    m_masks = []
    for scene in tqdm(scene_list, desc="load data"):
        folder_name = Path(scene).name
        save_path = os.path.join(args.model_path, folder_name, "detector")
        points3d = torch.load(os.path.join(save_path, "points3d.pt"), weights_only=True)
        cameras = torch.load(os.path.join(save_path, "salient_cameras.pt"), weights_only=True)
        sampled_idx = pickle.load(
            open(
                os.path.join(save_path, "salient_sampled_idx.pkl"),
                "rb",
            )
        )
        render_visible_masks = pickle.load(
            open(
                os.path.join(save_path, "salient_masks.pkl"),
                "rb",
            )
        )

        masks = None
        mask_path = os.path.join(args.source_path, folder_name, "masks.pkl")
        if os.path.exists(mask_path):
            with open(mask_path, "rb") as f:
                masks = pickle.load(f)

        m_points3d.append(points3d)
        m_idx.append(sampled_idx)
        m_cameras.append(cameras)
        m_render_masks.append(render_visible_masks)
        m_masks.append(masks)
    progress_bar = tqdm(range(0, train_iteration), desc="Scene-Independent Detector")
    first_iter = 1
    for iteration in range(first_iter, train_iteration + 1):
        iter_start.record()

        idx = random.randint(0, len(m_cameras) - 1)
        scene_path = scene_list[idx]
        folder_name = Path(scene_path).name
        points3d = m_points3d[idx]
        cameras = m_cameras[idx]
        sampled_idx = m_idx[idx]
        render_visible_masks = m_render_masks[idx]

        masks = m_masks[idx]

        camera = random.choice(cameras)
        img_name = camera["image_name"]
        img_path = os.path.join(args.source_path, folder_name, "images", img_name)
        original_image = Image.open(img_path)

        image_raw = torch.from_numpy(np.array(original_image)) / 255.0
        if len(image_raw.shape) == 3:
            image_raw = image_raw.permute(2, 0, 1)
        else:
            image_raw = image_raw.unsqueeze(dim=-1).permute(2, 0, 1)
        fovx = camera["fovx"]
        fovy = camera["fovy"]
        pose = camera["pose"]

        fine_resolution = get_resolution_from_longest_edge(
            480, 640)

        gt_image = image_raw.cuda()
        gt_image = F.interpolate(
            gt_image.unsqueeze(0),
            size=(fine_resolution[0], fine_resolution[1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        gt_feature_map = feature_extractor(gt_image[None])["feature_map"]
        gt_feature_map = F.interpolate(
            gt_feature_map,
            size=(fine_resolution[0], fine_resolution[1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        gt_feature_map = F.normalize(gt_feature_map, p=2, dim=0)

        viewmat = pose.cuda()
        focalX = fov2focal(fovx, gt_feature_map.shape[2])
        focalY = fov2focal(fovy, gt_feature_map.shape[1])
        K = torch.tensor(
            [
                [focalX, 0.0, gt_feature_map.shape[2] / 2],
                [0.0, focalY, gt_feature_map.shape[1] / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )

        render_visible_mask = render_visible_masks.get(img_name, None)
        gt_map = generate_gt_map(
            points3d, gt_feature_map, sampled_idx, viewmat, K, render_visible_mask
        )

        if masks is not None:
            object_mask = masks[img_name][0].cuda()[None]
            distort_mask = masks[img_name][2].cuda()[None]
            mask = object_mask & distort_mask
            gt_map_mask = (
                    F.interpolate(
                        mask[None].float(),
                        size=(gt_map.shape[1], gt_map.shape[2]),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    > 0.5
            )
            gt_map *= gt_map_mask

        heat_map = detector(gt_feature_map)
        loss = score_map_bce_loss(heat_map, gt_map)
        loss.backward()
        if iteration % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            loss_val = loss.item()
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{loss_val:.{7}f}",
                    }
                )
                progress_bar.update(10)
            if iteration == train_iteration:
                progress_bar.close()
            if tb_writer:
                tb_writer.add_scalar(
                    "detector_loss_patches/training_loss", loss_val, iteration
                )
                tb_writer.add_scalar(
                    "detector_loss_patches/lr",
                    optimizer.param_groups[0]["lr"],
                    iteration,
                )

        if iteration in saving_iterations:
            save_path = os.path.join(args.model_path, "detector")
            os.makedirs(save_path, exist_ok=True)
            torch.save(detector.state_dict(), save_path + f"/{iteration}_detector.pth")


if __name__ == "__main__":
    seed_everything(2025)

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)

    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10000, 20000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000, 20000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--detector_folder", type=str, default="detector")
    parser.add_argument("--landmark_num", type=int, default=16384)
    parser.add_argument("--landmark_k", type=int, default=32)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iteration)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    safe_state(args.quiet)

    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(os.path.join(args.model_path, args.detector_folder))
    else:
        tb_writer = None

    scene_list = sorted([
        os.path.join(args.source_path, d)
        for d in os.listdir(args.source_path)
        if os.path.isdir(os.path.join(args.source_path, d))
    ])

    training_detector_multi(
        scene_list,
        args,
        saving_iterations=args.save_iterations,
        tb_writer=tb_writer,
        train_iteration=args.iteration,
    )
