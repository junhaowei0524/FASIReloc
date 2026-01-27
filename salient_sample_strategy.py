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
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from tqdm import tqdm
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


def get_sampled_gaussian(gaussians: GaussianModel, idx_sampled):
    sampled_gaussians = GaussianModel(gaussians.max_sh_degree)
    sampled_gaussians._xyz = gaussians._xyz[idx_sampled]
    sampled_gaussians._loc_feature = gaussians._loc_feature[idx_sampled]
    sampled_gaussians._scaling = gaussians._scaling[idx_sampled]
    sampled_gaussians._opacity = gaussians._opacity[idx_sampled]
    sampled_gaussians._rotation = gaussians._rotation[idx_sampled]
    sampled_gaussians._features_dc = gaussians._features_dc[idx_sampled]
    sampled_gaussians._features_rest = gaussians._features_rest[idx_sampled]
    return sampled_gaussians


def generate_gt_map(
        gaussians: GaussianModel,
        gt_feature_map,
        idx_sampled,
        pose,
        K,
        render_visible_mask=None,
):
    if render_visible_mask is not None:
        render_visible_mask = render_visible_mask[idx_sampled]
        idx_sampled = idx_sampled[render_visible_mask]
    sampled_xyz = gaussians.get_xyz[idx_sampled]

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


@torch.no_grad()
def calculate_consistency_score(gaussians: GaussianModel, gt_feature_map, pose, K, render_visible_mask=None,
                                img_mask=None):
    points3d = gaussians.get_xyz

    points3d_homo = torch.cat([points3d, torch.ones(points3d.shape[0], 1, device=points3d.device)], dim=-1)
    points3d_cam = (pose @ points3d_homo.T)[:3]
    depths = points3d_cam[2]
    points3d_cam_homo = points3d_cam / depths
    points2d = (K @ points3d_cam_homo)[:2].long()

    in_mask = ((points2d[0] >= 0) & (points2d[0] < gt_feature_map.shape[2]) & (points2d[1] >= 0) & (
            points2d[1] < gt_feature_map.shape[1]))

    if render_visible_mask is not None:
        visible_mask = in_mask & render_visible_mask
    else:
        visible_mask = in_mask
    if img_mask is not None:
        visible_points2d = points2d[:, in_mask]
        img_mask_expand = torch.zeros_like(visible_mask, dtype=torch.bool)
        img_mask_expand[in_mask] = img_mask[0, visible_points2d[1], visible_points2d[0]]
        visible_mask = visible_mask & img_mask_expand

    x = 2 * (points2d[0, :] / (gt_feature_map.shape[2] - 1)) - 1
    y = 2 * (points2d[1, :] / (gt_feature_map.shape[1] - 1)) - 1
    grid = torch.stack([x, y], dim=-1).unsqueeze(0).unsqueeze(2)
    feat = F.grid_sample(gt_feature_map.unsqueeze(0), grid, mode='bilinear', align_corners=False)
    feat = feat.squeeze(0).squeeze(-1).T  # [N, C]
    feat[~visible_mask] = 0
    return feat, visible_mask


@torch.no_grad()
def calculate_geometry_score(gaussians: GaussianModel, gt_feature_map, pose, K, render_visible_mask=None,
                             img_mask=None):
    points3d = gaussians.get_xyz
    features3d = gaussians.get_loc_feature.squeeze()

    points3d_homo = torch.cat([points3d, torch.ones(points3d.shape[0], 1, device=points3d.device)], dim=-1)
    points3d_cam = (pose @ points3d_homo.T)[:3]
    depths = points3d_cam[2]
    points3d_cam_homo = points3d_cam / depths
    points2d = (K @ points3d_cam_homo)[:2].long()

    in_mask = ((points2d[0] >= 0) & (points2d[0] < gt_feature_map.shape[2]) & (points2d[1] >= 0) & (
            points2d[1] < gt_feature_map.shape[1]))

    if render_visible_mask is not None:
        visible_mask = in_mask & render_visible_mask
    else:
        visible_mask = in_mask
    if img_mask is not None:
        visible_points2d = points2d[:, in_mask]
        img_mask_expand = torch.zeros_like(visible_mask, dtype=torch.bool)
        img_mask_expand[in_mask] = img_mask[0, visible_points2d[1], visible_points2d[0]]  # [N,]
        visible_mask = visible_mask & img_mask_expand

    points2d = points2d[:, visible_mask]
    depths = depths[visible_mask]
    features3d = features3d[visible_mask]

    gs_features = F.normalize(features3d, p=2, dim=1)
    img_features = gt_feature_map[:, points2d[1], points2d[0]].T
    score = (gs_features * img_features).sum(-1)
    return score, visible_mask


def calculate_salient(scene, gaussians, feature_extractor, render_visible_masks, masks=None, num=16384, k=32):
    viewpoint_stack = scene.getTrainCameras().copy()
    img = viewpoint_stack[0].original_image.cuda()
    poses = torch.zeros(len(viewpoint_stack), 4, 4, dtype=torch.float32, device="cuda")
    vis_mask = torch.zeros(len(viewpoint_stack), gaussians.get_xyz.shape[0], dtype=torch.float32, device="cuda")
    idx = 0
    c = feature_extractor(img[None])["feature_map"][0].shape[0]
    v = len(viewpoint_stack)
    con_sum = torch.zeros(gaussians.get_xyz.shape[0], c, dtype=torch.float32, device="cuda")
    geo_sum = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.float32, device="cuda")
    score_num = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.float32, device="cuda")
    fine_resolution = (viewpoint_stack[0].original_image.shape[1],
                       viewpoint_stack[0].original_image.shape[2])

    for viewpoint_camera in tqdm(viewpoint_stack, desc="Salient Score:"):
        gt_image = viewpoint_camera.original_image.cuda()
        gt_feature_map = feature_extractor(gt_image[None])["feature_map"]
        gt_feature_map = F.interpolate(gt_feature_map, size=(fine_resolution[0], fine_resolution[1]),
                                       mode="bilinear", align_corners=False).squeeze(0)
        gt_feature_map = F.normalize(gt_feature_map, p=2, dim=0)

        pose = viewpoint_camera.world_view_transform.transpose(0, 1).cuda()
        poses[idx] = pose

        focalX = fov2focal(viewpoint_camera.FoVx, gt_feature_map.shape[2])
        focalY = fov2focal(viewpoint_camera.FoVy, gt_feature_map.shape[1])
        K = torch.tensor(
            [
                [focalX, 0.0, gt_feature_map.shape[2] / 2],
                [0.0, focalY, gt_feature_map.shape[1] / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32, device="cuda", )

        if render_visible_masks.get(viewpoint_camera.image_name, None) is None:
            render_visible_mask = get_render_visible_mask(
                gaussians, viewpoint_camera, gt_feature_map.shape[2], gt_feature_map.shape[1],
            )
            render_visible_masks[viewpoint_camera.image_name] = render_visible_mask

        if masks is not None:
            object_mask = masks[viewpoint_camera.image_name][0].cuda()[None]
            distort_mask = masks[viewpoint_camera.image_name][2].cuda()[None]
            mask = object_mask & distort_mask
            img_mask = (
                    F.interpolate(mask[None].float(), size=(fine_resolution[0], fine_resolution[1]),
                                  mode="bilinear", align_corners=False, ).squeeze(0) > 0.5
            )
        else:
            img_mask = None

        geo, mask = calculate_geometry_score(gaussians, gt_feature_map, pose, K,
                                             render_visible_mask=render_visible_masks[
                                                 viewpoint_camera.image_name],
                                             img_mask=img_mask)

        con, mask = calculate_consistency_score(gaussians, gt_feature_map, pose, K,
                                                render_visible_mask=render_visible_masks[
                                                    viewpoint_camera.image_name],
                                                img_mask=img_mask)
        score_num[mask] += 1
        geo_sum[mask] += geo
        con_sum += con
        vis_mask[idx] = mask
        idx = idx + 1
    score_num[score_num == 0] = 1
    geo_sum = geo_sum / score_num
    mean_score_con = con_sum / score_num[:, None]
    var_score_con = (con_sum - mean_score_con) ** 2 / score_num[:, None]
    var_scalar = var_score_con.mean(dim=1)
    consistency_score = torch.exp(-1 * var_scalar)
    consistency_score = 2 * consistency_score - 1
    score_num = score_num / v
    return geo_sum, consistency_score, score_num, poses, vis_mask, render_visible_masks


def calculate_generalizability(points3d, poses, vis_mask):
    points3d = points3d.detach().cpu().numpy()  # [N,3]
    poses = poses.detach().cpu().numpy()  # [V,4,4]
    vis_mask = vis_mask.detach().cpu().numpy()  # [V,N]

    N = points3d.shape[0]
    V = poses.shape[0]

    R = poses[:, :3, :3]  # [V,3,3]
    t = poses[:, :3, 3]  # [V,3]
    cam_centers = -np.einsum('vij,vj->vi', R.transpose(0, 2, 1), t)  # [V,3]

    H_accum = np.zeros((N, 3, 3), dtype=np.float32)
    counts = np.zeros(N, dtype=np.int32)

    for v in tqdm(range(V), desc="Generalizability"):
        visible_points = np.nonzero(vis_mask[v])[0]
        if visible_points.size == 0:
            continue

        pts = points3d[visible_points]  # [K,3]
        dirs = pts - cam_centers[v][None, :]  # [K,3]
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs /= np.maximum(norms, 1e-8)

        bbT = np.einsum('ki,kj->kij', dirs, dirs)
        I_minus_bbT = np.eye(3, dtype=np.float32)[None, :, :] - bbT

        np.add.at(H_accum, visible_points, I_minus_bbT)
        np.add.at(counts, visible_points, 1)

    gen_scores = np.zeros(N, dtype=np.float32)
    valid_idx = counts > 1

    H_valid = H_accum[valid_idx] / counts[valid_idx, None, None]
    H_valid = 0.5 * (H_valid + np.transpose(H_valid, (0, 2, 1)))

    eigvals = np.linalg.eigvalsh(H_valid)
    min_eig = eigvals[:, 0]
    max_eig = eigvals[:, -1]

    ratio = np.clip(min_eig / np.maximum(max_eig, 1e-8), 0, 1)
    gen_scores[valid_idx] = np.arccos(1 - 2 * ratio) / np.pi

    gen_scores = torch.from_numpy(gen_scores).to("cuda")
    gen_scores = gen_scores * 2 - 1
    return gen_scores


def random_block_knn_score(points3d, score, num, k, block=1024):
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


def salient_sample(scene, gaussians, feature_extractor, render_visible_masks, masks=None, num=16384, k=32):
    score_geo, score_con, score_stab, poses, vis_mask, render_visible_masks = calculate_salient(scene, gaussians,
                                                                                                feature_extractor,
                                                                                                render_visible_masks,
                                                                                                masks=masks, num=num,
                                                                                                k=k)
    score_gen = calculate_generalizability(gaussians.get_xyz, poses, vis_mask)
    w1, w2, w3, w4 = 1, 0.3, 0.3, 0.3
    salient_score = w1 * score_geo + w2 * score_con + w3 * score_stab + w4 * score_gen
    sample_idx = random_block_knn_score(gaussians.get_xyz, salient_score, num, k)
    sample_idx = torch.unique(sample_idx)

    return sample_idx, render_visible_masks


def evaluate_detector(
        detector,
        feature_extractor,
        gaussians,
        sampled_idx,
        scene,
        masks=None,
        render_visible_masks=None,
        tb_writer=None,
        iteration=0,
):
    torch.cuda.empty_cache()

    landmarks = get_sampled_gaussian(gaussians, sampled_idx)

    bg_color = [1, 1, 1] if scene.args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    validation_configs = (
        {"name": "test", "cameras": scene.getTestCameras()},
        {
            "name": "train",
            "cameras": [
                scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                for idx in range(5, 30, 5)
            ],
        },
    )

    for config in validation_configs:
        if config["cameras"] and len(config["cameras"]) > 0:
            fine_resolution = get_resolution_from_longest_edge(
                config["cameras"][0].original_image.shape[1],
                config["cameras"][0].original_image.shape[2],
                scene.longest_edge,
            )
            loss_sum = 0.0

            for idx, viewpoint_cam in enumerate(config["cameras"]):
                gt_image = viewpoint_cam.original_image.cuda()
                gt_feature_map = feature_extractor(gt_image[None])["feature_map"]
                gt_feature_map = F.interpolate(
                    gt_feature_map,
                    size=(fine_resolution[0], fine_resolution[1]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                gt_feature_map = F.normalize(gt_feature_map, p=2, dim=0)

                viewmat = viewpoint_cam.world_view_transform.transpose(0, 1).cuda()
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

                visible_mask = render_visible_masks.get(
                    viewpoint_cam.image_name, None
                )
                if visible_mask is None:
                    visible_mask = get_render_visible_mask(
                        gaussians,
                        viewpoint_cam,
                        gt_feature_map.shape[2],
                        gt_feature_map.shape[1],
                    )
                    render_visible_masks[viewpoint_cam.image_name] = visible_mask
                else:
                    visible_mask = render_visible_masks[viewpoint_cam.image_name]

                gt_map = generate_gt_map(
                    gaussians,
                    gt_feature_map,
                    sampled_idx,
                    viewmat,
                    K,
                    visible_mask,
                )

                if masks is not None:
                    object_mask = masks[viewpoint_cam.image_name][0].cuda()[None]
                    distort_mask = masks[viewpoint_cam.image_name][2].cuda()[None]

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
                loss_sum += loss.item()

                if tb_writer and idx < 5:
                    render = render_gsplat(
                        viewpoint_cam, gaussians, background, rgb_only=True
                    )["render"]
                    sampled_render = render_gsplat(
                        viewpoint_cam, landmarks, background, rgb_only=True
                    )["render"]
                    heat_map = (heat_map - heat_map.min()) / (
                            heat_map.max() - heat_map.min()
                    )
                    tb_writer.add_images(
                        f"detector_vis_{config['name']}/gt_map_{idx}",
                        gt_map[None],
                        iteration,
                    )
                    tb_writer.add_images(
                        f"detector_vis_{config['name']}/heat_map{idx}",
                        heat_map[None],
                        iteration,
                    )
                    tb_writer.add_images(
                        f"detector_vis_{config['name']}/render_{idx}",
                        render[None],
                        iteration,
                    )
                    tb_writer.add_images(
                        f"detector_vis_{config['name']}/sampled_render_{idx}",
                        sampled_render[None],
                        iteration,
                    )

            loss_sum /= len(config["cameras"])
            print(
                f"\n[ITER {iteration}] Evaluating detector: {config['name']} loss {loss_sum}"
            )
            if tb_writer:
                tb_writer.add_scalar(
                    f"detector_loss_patches/{config['name']}_loss",
                    loss_sum,
                    iteration,
                )


def save_data(scene, gaussians, sampled_idx, render_visible_masks, detector_folder):
    save_path = os.path.join(scene.model_path, detector_folder)
    os.makedirs(save_path, exist_ok=True)
    pickle.dump(sampled_idx, open(os.path.join(save_path, "salient_sampled_idx.pkl"), "wb"))
    pickle.dump(render_visible_masks, open(os.path.join(save_path, "salient_masks.pkl"), "wb"))
    torch.save(gaussians.get_xyz, os.path.join(save_path, "points3d.pt"))

    camera_infos = []
    viewpoint_stack = scene.getTrainCameras().copy()
    for camera in viewpoint_stack:
        cam_dict = {
            "image_name": camera.image_name,
            "fovx": camera.FoVx,
            "fovy": camera.FoVy,
            "pose": camera.world_view_transform.transpose(0, 1),
        }
        camera_infos.append(cam_dict)
    torch.save(camera_infos, os.path.join(save_path, "salient_cameras.pt"))


def training_detector(
        gaussians,
        scene: Scene,
        masks,
        detector_folder="",
        landmark_num=16384,
        landmark_k=32,
):
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
    feature_extractor = FeatureExtractor(scene.feature_type)
    render_visible_masks = {}

    print("salient sampling...")
    sampled_idx, render_visible_masks = salient_sample(
        scene,
        gaussians,
        feature_extractor,
        render_visible_masks,
        masks=masks,
        num=landmark_num,
        k=landmark_k,
    )
    save_data(scene, gaussians, sampled_idx, render_visible_masks, detector_folder)


if __name__ == "__main__":
    seed_everything(2025)
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[10000, 20000, 30000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[10000, 20000, 30000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--detector_folder", type=str, default="detector")
    parser.add_argument("--landmark_num", type=int, default=16384)
    parser.add_argument("--landmark_k", type=int, default=32)

    args = get_combined_args(parser)
    args.save_iterations.append(args.iterations)

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset = lp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)

    masks = None
    if os.path.exists(os.path.join(dataset.source_path, "masks.pkl")):
        import pickle

        masks = pickle.load(open(os.path.join(dataset.source_path, "masks.pkl"), "rb"))

    scene = Scene(dataset, gaussians, load_iteration=args.iteration)

    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(
            os.path.join(dataset.model_path, args.detector_folder)
        )
    else:
        tb_writer = None

    training_detector(
        gaussians,
        scene,
        masks,
        detector_folder=args.detector_folder,
        landmark_num=args.landmark_num,
        landmark_k=args.landmark_k,
    )

    # All done
    print("\n salient sampling strategy complete.")
