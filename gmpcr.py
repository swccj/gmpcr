"""Core implementation of the selected GMPCR Rayleigh-confidence method."""

import heapq
import random
from pathlib import Path

import numpy as np
import torch
from igraph import Graph
from tqdm import tqdm

from TransSync.Laplacian_TS import pair2globalT_cycle
from dataops.dataset import get_dataset
from utils.T_eval import (
    evaluate_registration,
    read_pre_trajectory,
    read_trajectory,
    read_trajectory_info,
)
from utils.knn_search import KNNMatcher
from utils.r_eval import compute_R_diff

R_SPLITS = (3, 5, 10, 30, 45)
T_SPLITS = (0.05, 0.1, 0.25, 0.5, 0.75)
TRANSLATION_RESIDUAL_DATASETS = frozenset(
    ("3dmatch", "3dLomatch", "scannet", "ETH")
)
DECOUPLED_TRANSLATION_DATASETS = frozenset(
    ("3dmatch", "3dLomatch", "scannet")
)
@torch.no_grad()
def rayleigh_scg_confidence(scg, spectral_vector):
    """Return the SMVR-style Rayleigh confidence of a refined SCG."""
    if scg.ndim != 2 or scg.shape[0] != scg.shape[1]:
        raise ValueError(f"SCG must be square, found shape {tuple(scg.shape)}")
    if spectral_vector.shape[0] != scg.shape[0]:
        raise ValueError(
            "SCG and spectral vector size mismatch: "
            f"{tuple(scg.shape)} versus {tuple(spectral_vector.shape)}"
        )
    if scg.numel() == 0:
        return scg.new_zeros(())

    symmetric_scg = 0.5 * (scg + scg.transpose(0, 1))
    vector = spectral_vector.reshape(-1)
    vector_norm = torch.linalg.vector_norm(vector)
    if vector_norm.item() == 0.0:
        return scg.new_zeros(())
    unit_vector = vector / vector_norm
    return torch.dot(unit_vector, symmetric_scg @ unit_vector)


def normalize_scan_pair_scores(score_matrix):
    """L2-normalize finite unordered scan-pair scores as in SMVR."""
    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(
            f"Scan-pair score matrix must be square, found shape {scores.shape}"
        )

    normalized = scores.copy()
    finite = np.isfinite(normalized)
    unordered = np.triu(finite, k=1)
    score_norm = float(np.linalg.norm(normalized[unordered]))
    if score_norm > 0.0:
        normalized[finite] /= score_norm
    else:
        normalized[finite] = 0.0
    return normalized, score_norm


def combine_pair_confidence_with_inlier_weights(
    inlier_weights,
    pair_scores,
    *,
    confidence_floor=1.0,
    confidence_power=1.0,
):
    """Fuse scan-pair confidence into initial pose-graph edge weights.

    Pair confidence is max-normalized over successfully estimated retained
    edges. A floor of one exactly preserves the previous IR-support weights;
    a floor of zero and power of one gives the SMVR-style product up to a
    scene-wide positive scale.
    """
    weights = np.asarray(inlier_weights, dtype=np.float64)
    scores = np.asarray(pair_scores, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError(f"Inlier-weight matrix must be square, found {weights.shape}")
    if scores.shape != weights.shape:
        raise ValueError(
            f"Pair-score shape {scores.shape} does not match weights {weights.shape}"
        )
    if not 0.0 <= confidence_floor <= 1.0:
        raise ValueError("Pair-confidence weight floor must be in [0, 1]")
    if confidence_power <= 0.0:
        raise ValueError("Pair-confidence weight power must be positive")

    fused = weights.copy()
    if confidence_floor == 1.0:
        return fused, np.zeros_like(weights)

    valid = (weights > 0.0) & np.isfinite(scores)
    if not np.any(valid):
        return fused, np.zeros_like(weights)

    nonnegative_scores = np.maximum(scores, 0.0)
    score_scale = float(np.max(nonnegative_scores[valid]))
    if score_scale <= np.finfo(np.float64).eps:
        return fused, np.zeros_like(weights)

    normalized_scores = np.zeros_like(weights)
    normalized_scores[valid] = np.clip(
        nonnegative_scores[valid] / score_scale,
        0.0,
        1.0,
    )
    confidence_multiplier = confidence_floor + (1.0 - confidence_floor) * (
        normalized_scores**confidence_power
    )
    fused[valid] *= confidence_multiplier[valid]
    return fused, normalized_scores


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def topk_smallest_indices(values, k):
    k = min(int(k), values.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.intp)
    if k == values.shape[0]:
        return np.argsort(values)

    top_indices = np.argpartition(values, k - 1)[:k]
    return top_indices[np.argsort(values[top_indices])]


def mean_or_nan(values):
    return float(np.mean(values)) if len(values) else float("nan")


def farthest_point_sample(xyz, num_samples):
    num_points = xyz.shape[0]
    if num_points <= num_samples:
        return np.arange(num_points)

    centroids = np.zeros(num_samples, dtype=np.int_)
    distances = np.full(num_points, np.inf)

    farthest = np.random.randint(0, num_points)

    for i in range(num_samples):
        centroids[i] = farthest
        centroid = xyz[farthest, :]

        dist = np.sum((xyz - centroid) ** 2, axis=-1)

        mask = dist < distances
        distances[mask] = dist[mask]

        farthest = np.argmax(distances)

    return centroids


def get_des(dataset, sid, data_root):
    return np.load(Path(data_root) / dataset.name / "yoho_desc" / f"{sid}.npy")


def configure_eval_metrics(args):
    args.rr = args.dataset in ("3dmatch", "3dLomatch", "ETH")
    args.ecdf = args.dataset == "scannet"


def resolve_translation_residual_scale(args):
    """Keep legacy synchronization unless a dataset has a verified joint model."""
    if args.dataset not in TRANSLATION_RESIDUAL_DATASETS:
        return 0.0
    return max(float(args.translation_residual_scale), 0.0)


class Gmpcr:
    def __init__(self, args):
        self.args = args
        self.data_root = Path(args.data_dir).expanduser().resolve()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.KNN = KNNMatcher()
        self.topk = self.args.topk
        self.num_iterations = self.args.num_iterations
        self.inlier_threshold = self.args.inlier_threshold
        self.inlier_threshold_sq = self.inlier_threshold**2
        self.max_cliques = self.args.max_cliques
        self.translation_residual_scale = resolve_translation_residual_scale(args)
        self.decouple_translation_residual = (
            args.dataset in DECOUPLED_TRANSLATION_DATASETS
            and self.translation_residual_scale > 0
        )
        self.prelog_dir = Path(self.args.save_dir) / "cycle_results" / "cycle_prelogs"
        self.prepose_dir = Path(self.args.save_dir) / "cycle_results" / "pcposes"
        self.last_pair_scores = None

    def savepose(self, dataset, poses):
        pose_dir = self.prepose_dir / dataset.name
        pose_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(pose_dir / "pose.txt", poses.reshape(-1, 4), delimiter=",")

    def savelog(self, dataset, trans):
        log_dir = self.prelog_dir / dataset.name
        log_dir.mkdir(parents=True, exist_ok=True)
        pair_num = int(len(dataset.pc_ids))
        with open(log_dir / "pre.log", "w") as writer:
            for i, pair in enumerate(dataset.pair_ids):
                pc0, pc1 = pair
                transform_pr = trans[i]
                writer.write(f"{int(pc0)}\t{int(pc1)}\t{pair_num}\n")
                writer.write(
                    f"{transform_pr[0][0]}\t{transform_pr[0][1]}\t{transform_pr[0][2]}\t{transform_pr[0][3]}\n"
                )
                writer.write(
                    f"{transform_pr[1][0]}\t{transform_pr[1][1]}\t{transform_pr[1][2]}\t{transform_pr[1][3]}\n"
                )
                writer.write(
                    f"{transform_pr[2][0]}\t{transform_pr[2][1]}\t{transform_pr[2][2]}\t{transform_pr[2][3]}\n"
                )
                writer.write(f"{0.0}\t{0.0}\t{0.0}\t{1.0}\n")

    def official_RR(self, dataset):
        gt_dir = Path(dataset.gt_dir)
        gt_pairs, gt_traj = read_trajectory(gt_dir.with_suffix(".log"))
        _, gt_traj_cov = read_trajectory_info(gt_dir.with_suffix(".info"))

        n_fragments = len(dataset.pc_ids)
        gt_indices = np.asarray(gt_pairs[:, :2], dtype=int)
        retained = np.all(gt_indices < n_fragments, axis=1)
        gt_pairs = gt_pairs[retained]
        gt_traj = gt_traj[retained]
        gt_traj_cov = gt_traj_cov[retained]

        pre_dir = self.prelog_dir / dataset.name
        est_pairs, est_traj = read_pre_trajectory(pre_dir / "pre.log")

        _, recall, _, _ = evaluate_registration(
            n_fragments,
            est_traj,
            est_pairs,
            gt_pairs,
            gt_traj,
            gt_traj_cov,
            err2=self.args.tau_2,
            nonconsecutive=True,
        )
        return recall

    def ecdf(self, dataset, r_splits=R_SPLITS, t_splits=T_SPLITS):
        pre_dir = self.prelog_dir / dataset.name
        est_pairs, est_traj = read_pre_trajectory(pre_dir / "pre.log")
        Rdiffs, tdiffs = [], []
        for pair, est in zip(est_pairs, est_traj):
            id0, id1, _ = pair
            gt = dataset.get_transform(id0, id1)
            Rdiff = compute_R_diff(gt[0:3, 0:3], est[0:3, 0:3])
            tdiff = np.sqrt(np.sum(np.square(gt[0:3, -1] - est[0:3, -1])) + 1e-8)
            Rdiffs.append(Rdiff)
            tdiffs.append(tdiff)
        if not Rdiffs:
            return np.zeros(len(r_splits)), 0.0, 0.0, np.zeros(len(t_splits)), 0.0, 0.0
        Rdiffs = np.array(Rdiffs)
        tdiffs = np.array(tdiffs)

        ecdf_r = [np.mean(Rdiffs < rthes) for rthes in r_splits]
        mean_r = np.mean(Rdiffs)
        med_r = np.median(Rdiffs)

        ecdf_t = [np.mean(tdiffs < tthes) for tthes in t_splits]
        mean_t = np.mean(tdiffs)
        med_t = np.median(tdiffs)
        return np.array(ecdf_r), mean_r, med_r, np.array(ecdf_t), mean_t, med_t

    @torch.no_grad()
    def match(self, des0, des1):
        feats0 = torch.from_numpy(np.asarray(des0, dtype=np.float32).T[None]).to(
            self.device
        )
        feats1 = torch.from_numpy(np.asarray(des1, dtype=np.float32).T[None]).to(
            self.device
        )
        d_0_in_1, argmin_of_0_in_1 = self.KNN(feats1, feats0)
        d_0_in_1 = d_0_in_1[0, 0].cpu().numpy()
        argmin_of_0_in_1 = argmin_of_0_in_1[0, 0].cpu().numpy()
        _, argmin_of_1_in_0 = self.KNN(feats0, feats1)
        argmin_of_1_in_0 = argmin_of_1_in_0[0, 0].cpu().numpy()
        src_indices = np.arange(argmin_of_0_in_1.shape[0])
        mutual_mask = argmin_of_1_in_0[argmin_of_0_in_1] == src_indices
        if not np.any(mutual_mask):
            return np.zeros((0, 2), dtype=int)
        match_pps = np.stack(
            [src_indices[mutual_mask], argmin_of_0_in_1[mutual_mask]], axis=1
        ).astype(int)
        match_distances = d_0_in_1[mutual_mask]
        top_indices = topk_smallest_indices(match_distances, self.args.top_overlap)
        match_pps = match_pps[top_indices]
        return match_pps

    @torch.no_grad()
    def build_fcg(self, Keys_m0, Keys_m1):
        src_pts = torch.from_numpy(Keys_m0).to(device=self.device, dtype=torch.float32)
        tgt_pts = torch.from_numpy(Keys_m1).to(device=self.device, dtype=torch.float32)
        comp_threshold = self.args.comp_threshold
        src_dist = torch.cdist(src_pts, src_pts)
        tgt_dist = torch.cdist(tgt_pts, tgt_pts)
        cross_dis = torch.abs(src_dist - tgt_dist)
        FCG = 1 - cross_dis.square() / (comp_threshold**2)
        FCG.clamp_(min=0)
        FCG.fill_diagonal_(0)
        FCG.masked_fill_(FCG < self.args.fcg_threshold, 0)
        return FCG

    @torch.no_grad()
    def build_scg_from_fcg(self, FCG):
        SCG = torch.matmul(FCG, FCG) * FCG
        leading_eigenvector = self.cal_leading_eigenvector(self.num_iterations, SCG)
        return SCG, leading_eigenvector

    def build_scg(self, Keys_m0, Keys_m1):
        FCG = self.build_fcg(Keys_m0, Keys_m1)
        SCG, leading_eigenvector = self.build_scg_from_fcg(FCG)
        return FCG, SCG, leading_eigenvector

    def select_consistent_by_ratio(self, leading_eigenvector):
        num_corr = leading_eigenvector.shape[0]
        if num_corr <= 3:
            return np.arange(num_corr)

        keep_ratio = min(max(self.args.scg_keep_ratio, 0.0), 1.0)
        keep_num = min(max(3, int(np.ceil(num_corr * keep_ratio))), num_corr)
        if keep_num == num_corr:
            return np.arange(num_corr)
        consistency_scores = torch.abs(leading_eigenvector).view(-1)
        return (
            torch.topk(consistency_scores, keep_num, largest=True).indices.cpu().numpy()
        )

    def prune(self, n, scene):
        scoremat = np.full((n, n), -np.inf)
        mask = np.zeros((n, n), dtype=bool)
        all_matched_keys = {}
        all_SCGs = {}
        all_leading_eigenvectors = {}

        kps_list = [scene.get_kps(i) for i in range(n)]
        des_list = [get_des(scene, i, self.data_root) for i in range(n)]

        fps_indices_list = [
            farthest_point_sample(des, self.args.num_sample) for des in des_list
        ]
        kps_list = [kps[indices] for kps, indices in zip(kps_list, fps_indices_list)]
        des_list = [des[indices] for des, indices in zip(des_list, fps_indices_list)]

        for i in range(n):
            for j in range(i + 1, n):
                match = self.match(des_list[i], des_list[j])

                if len(match) == 0:
                    continue

                Keys_m0 = kps_list[i][match[:, 0]]
                Keys_m1 = kps_list[j][match[:, 1]]

                FCG, SCG, leading_eigenvector = self.build_scg(Keys_m0, Keys_m1)
                keep_indices = self.select_consistent_by_ratio(leading_eigenvector)
                Keys_m0 = Keys_m0[keep_indices]
                Keys_m1 = Keys_m1[keep_indices]
                keep_indices_t = torch.from_numpy(keep_indices).to(device=FCG.device)
                FCG = FCG[keep_indices_t][:, keep_indices_t]
                SCG, leading_eigenvector = self.build_scg_from_fcg(FCG)

                all_matched_keys[(i, j)] = (Keys_m0, Keys_m1)

                all_SCGs[(i, j)] = SCG.cpu()
                all_leading_eigenvectors[(i, j)] = leading_eigenvector.cpu()

                current_conf_scalar = rayleigh_scg_confidence(
                    SCG,
                    leading_eigenvector,
                ).item()

                scoremat[j, i] = current_conf_scalar
                scoremat[i, j] = current_conf_scalar

        scoremat, _ = normalize_scan_pair_scores(scoremat)

        self.last_pair_scores = scoremat.copy()
        topk = min(self.topk, n)
        if topk <= 0:
                return mask, all_matched_keys, all_SCGs, all_leading_eigenvectors

        partition_idx = np.argsort(-np.nan_to_num(scoremat, nan=-np.inf), axis=1)[
            :, :topk
        ]
        row_indices = np.arange(n)[:, None]
        mask[row_indices, partition_idx] = True

        return mask, all_matched_keys, all_SCGs, all_leading_eigenvectors

    def max_clique_search(self, SCG):
        SCG = SCG.detach().cpu().numpy()
        graph = Graph.Adjacency((SCG > 0).tolist())
        graph.to_undirected()
        return graph.maximal_cliques(min=3)

    def clique_selection(self, macs, SCG):
        if not macs:
            return []
        weighted_cliques = []
        for clique in macs:
            clique_tensor = torch.as_tensor(clique, device=SCG.device)
            sub_matrix = SCG[clique_tensor][:, clique_tensor]
            num_edges = len(clique) * (len(clique) - 1)
            weighted_cliques.append(((sub_matrix.sum() / num_edges).item(), clique))

        return [
            clique
            for _, clique in heapq.nlargest(
                self.max_cliques,
                weighted_cliques,
                key=lambda item: item[0],
            )
        ]

    def cal_leading_eigenvector(self, num_iterations, M):
        leading_eig = torch.ones_like(M[:, 0:1])
        leading_eig_last = leading_eig
        for _ in range(num_iterations):
            leading_eig = M @ leading_eig
            leading_eig = leading_eig / (
                torch.norm(leading_eig, dim=0, keepdim=True) + 1e-6
            )
            if torch.allclose(leading_eig, leading_eig_last):
                break
            leading_eig_last = leading_eig
        return leading_eig

    def weighted_svd(self, pts0, pts1, weights):
        weight_sum = weights.sum() + 1e-8
        weights_normalized = weights / weight_sum

        centroid0 = torch.sum(pts0 * weights_normalized, dim=0, keepdim=True)
        centroid1 = torch.sum(pts1 * weights_normalized, dim=0, keepdim=True)

        centered_pts0 = pts0 - centroid0
        centered_pts1 = pts1 - centroid1

        H = torch.matmul(centered_pts1.transpose(0, 1), centered_pts0 * weights)

        U, _, Vh = torch.linalg.svd(H)

        R = torch.matmul(Vh.transpose(0, 1), U.transpose(0, 1))

        if torch.det(R) < 0:
            Vh_corrected = Vh.clone()
            Vh_corrected[-1, :] *= -1
            R = torch.matmul(Vh_corrected.transpose(0, 1), U.transpose(0, 1))

        t = centroid0.transpose(0, 1) - torch.matmul(R, centroid1.transpose(0, 1))
        t = t.squeeze()

        T = torch.eye(4, dtype=R.dtype, device=R.device)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def dataseteval(self, dataset, Tpres):
        pair_ids = np.asarray(dataset.pair_ids, dtype=int)
        if pair_ids.size == 0:
            Tpairs = np.empty((0, 4, 4), dtype=Tpres.dtype)
            self.savelog(dataset, Tpairs)
            return

        Tpres_inv = np.linalg.inv(Tpres)
        Tpairs = Tpres_inv[pair_ids[:, 0]] @ Tpres[pair_ids[:, 1]]
        self.savelog(dataset, Tpairs)

    def run(self):
        dataset = get_dataset(self.args.dataset, self.data_root)
        if self.decouple_translation_residual:
            synchronization_mode = "decoupled rotation and translation residuals"
        elif self.translation_residual_scale > 0:
            synchronization_mode = "joint rotation-translation residual"
        else:
            synchronization_mode = "legacy rotation-only residual"
        print(
            f"Synchronization mode for {self.args.dataset}: "
            f"{synchronization_mode}"
        )
        if self.args.scene is not None and self.args.scene not in dataset:
            available_scenes = [
                name for name, value in dataset.items() if not isinstance(value, str)
            ]
            raise ValueError(
                f"Unknown scene {self.args.scene!r} for {self.args.dataset}. "
                f"Available scenes: {available_scenes}"
            )
        Total_N_pair = 0
        RR = []
        IRs = []
        ecdf_rs, mean_rs, med_rs, ecdf_ts, mean_ts, med_ts = [], [], [], [], [], []
        setname = dataset["wholesetname"]
        disable_progress = os.environ.get("GMPCR_DISABLE_PROGRESS") == "1"
        for name, scene in tqdm(dataset.items(), disable=disable_progress):
            if isinstance(scene, str):
                continue
            if self.args.scene is not None and name != self.args.scene:
                continue
            print("construct graph:", name)
            n = len(scene.pc_ids)
            Ts = np.zeros([n, n, 4, 4])
            irs = np.zeros([n, n])
            N_pair = 0
            mask, all_matched_keys, all_SCGs, all_leading_eigenvectors = self.prune(
                n, scene
            )
            candidate_pairs = np.argwhere(np.triu(mask | mask.T, k=1))
            for i, j in candidate_pairs:
                pair_key = (int(i), int(j))
                if pair_key not in all_matched_keys:
                    continue

                Keys_m0, Keys_m1 = all_matched_keys[pair_key]
                SCG = all_SCGs[pair_key]
                leading_eigenvector = all_leading_eigenvectors[pair_key]

                macs = self.max_clique_search(SCG)
                macs_filterd = self.clique_selection(macs, SCG)

                best_inlier = -1.0
                best_T = None
                N_matches = -1

                pts0_full = torch.from_numpy(Keys_m0).to(
                    SCG.device, dtype=torch.float32
                )
                pts1_full = torch.from_numpy(Keys_m1).to(
                    SCG.device, dtype=torch.float32
                )

                for mac in macs_filterd:
                    if len(mac) < 3:
                        continue

                    mac_tensor = torch.as_tensor(mac, device=SCG.device)
                    pts0 = pts0_full[mac_tensor]
                    pts1 = pts1_full[mac_tensor]
                    weights_svd = torch.abs(leading_eigenvector[mac_tensor]).view(-1, 1)

                    T_candidate = self.weighted_svd(pts0, pts1, weights_svd)

                    R = T_candidate[:3, :3].float()
                    t = T_candidate[:3, 3].float()
                    Keys_m1_transformed = torch.matmul(pts1_full, R.transpose(0, 1)) + t
                    distances_sq = torch.sum(
                        (pts0_full - Keys_m1_transformed) ** 2, dim=-1
                    )

                    inlier_mask = distances_sq < self.inlier_threshold_sq
                    inlier_ratio = inlier_mask.float().mean().item()

                    if inlier_ratio > best_inlier:
                        best_inlier = inlier_ratio
                        best_T = T_candidate.cpu().numpy()
                        N_matches = len(mac)

                if best_T is not None:
                    irs[i, j] = best_inlier * N_matches / 100.0
                    irs[j, i] = best_inlier * N_matches / 100.0
                    Ts[i, j] = best_T
                    Ts[j, i] = np.linalg.inv(best_T)
                    N_pair += 1
                    IRs.append(best_inlier)
            print(f"Estimate {N_pair} pairs")

            final_weights, _ = combine_pair_confidence_with_inlier_weights(
                irs,
                self.last_pair_scores,
                confidence_floor=self.args.pair_confidence_weight_floor,
                confidence_power=self.args.pair_confidence_weight_power,
            )
            if N_pair == 0 or not np.any(final_weights > 1e-8):
                print(
                    f"Warning: No valid connections for scene {name}. Outputting Identity poses."
                )
                Tglobals = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
            else:
                Tglobals, _ = pair2globalT_cycle(
                    final_weights,
                    Ts,
                    max_iterations=self.args.weight_iterations,
                    lambda_min=self.args.lambda_min,
                    lambda_max=self.args.lambda_max,
                    kappa=self.args.kappa,
                    rho=self.args.rho,
                    w_min=self.args.w_min,
                    w_max=self.args.w_max,
                    residual_scale=self.args.residual_scale,
                    translation_residual_scale=self.translation_residual_scale,
                    decouple_translation_residual=self.decouple_translation_residual,
                    translation_max_iterations=self.args.translation_weight_iterations,
                    translation_kappa=self.args.translation_kappa,
                    translation_rho=self.args.translation_rho,
                )
            self.savepose(scene, Tglobals)
            Total_N_pair += N_pair

            self.dataseteval(scene, Tglobals)
            if self.args.rr:
                rr_one = self.official_RR(scene)
                RR.append(rr_one)
                print(f"{scene.name} rr: {rr_one}")
            if self.args.ecdf:
                ecdf_r, mean_r, med_r, ecdf_t, mean_t, med_t = self.ecdf(scene)
                ecdf_rs.append(ecdf_r[None])
                mean_rs.append(mean_r)
                med_rs.append(med_r)
                ecdf_ts.append(ecdf_t[None])
                mean_ts.append(mean_t)
                med_ts.append(med_t)
                print(f"{scene.name} ecdf_r:{ecdf_r}, ecdf_t:{ecdf_t}")
        print(f"Conduct {Total_N_pair} pairwise registrations in total.")
        print("IR of the selected pairwise transformations:", mean_or_nan(IRs))
        if self.args.rr:
            print(f"RR of {setname} - Avg: ", mean_or_nan(RR))
        if self.args.ecdf:
            if not ecdf_rs:
                print(f"No valid scenes were evaluated for {setname}.")
                return
            ecdf_rs = np.concatenate(ecdf_rs, axis=0)
            ecdf_ts = np.concatenate(ecdf_ts, axis=0)
            ecdf_rs = np.mean(ecdf_rs, axis=0)
            ecdf_ts = np.mean(ecdf_ts, axis=0)
            print(f"ECDF_R of {setname} - Statistic: ", ecdf_rs)
            print(
                f"ECDF_R of {setname} - Mean/Med: ",
                mean_or_nan(mean_rs),
                "/",
                mean_or_nan(med_rs),
            )
            print(f"ECDF_T of {setname} - Statistic: ", ecdf_ts)
            print(
                f"ECDF_T of {setname} - Mean/Med: ",
                mean_or_nan(mean_ts),
                "/",
                mean_or_nan(med_ts),
            )
