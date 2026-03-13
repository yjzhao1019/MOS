# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
import torch
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss

def modality_alignment_loss(feat, img_paths):
    # 判断模态
    modality_labels = []
    for path in img_paths:
        if 'rgb' in path.lower() or 'optical' in path.lower():
            modality_labels.append(0)
        elif 'sar' in path.lower():
            modality_labels.append(1)
        else:
            modality_labels.append(-1)  # 未知类别，可忽略

    modality_labels = torch.tensor(modality_labels, device=feat.device)

    # 分离模态特征
    optical_feats = feat[modality_labels == 0]
    sar_feats = feat[modality_labels == 1]

    if len(optical_feats) == 0 or len(sar_feats) == 0:
        return torch.tensor(0.0, device=feat.device)

    mu_opt = optical_feats.mean(dim=0)
    mu_sar = sar_feats.mean(dim=0)
    mean_loss = F.mse_loss(mu_opt, mu_sar)

    # 方差（或标准差）
    var_opt = optical_feats.var(dim=0, unbiased=False)
    var_sar = sar_feats.var(dim=0, unbiased=False)
    var_loss = F.mse_loss(var_opt, var_sar)

    # 加权组合（可调超参）
    alpha = 1.0
    beta = 1.0  # 方差损失权重通常小一些
    total_loss = alpha * mean_loss + beta * var_loss

    return total_loss

def classwise_modality_alignment_loss(feat, labels, img_paths):
    """
    按 ID 对齐的 Optical-SAR 模态距离损失。
    feat: Tensor [B, D]
    labels: Tensor [B] -> 每个样本的行人/舰船ID (pid)
    img_paths: list[str] -> 每个样本对应路径，用于判断模态(opt/sar)
    """
    # 判断模态 (0: Optical, 1: SAR)
    modality_labels = []
    for path in img_paths:
        path_lower = path.lower()
        if "rgb" in path_lower or "optical" in path_lower:
            modality_labels.append(0)
        elif "sar" in path_lower:
            modality_labels.append(1)
        else:
            modality_labels.append(-1)
    modality_labels = torch.tensor(modality_labels, device=feat.device)
    feat = F.normalize(feat, p=2, dim=1)

    ids = labels.unique()
    losses = []
    mean_losses = []
    var_losses = []

    for pid in ids:
        pid_mask = (labels == pid)
        opt_feats = feat[(pid_mask) & (modality_labels == 0)]
        sar_feats = feat[(pid_mask) & (modality_labels == 1)]

        if len(opt_feats) == 0 or len(sar_feats) == 0:
            continue  # 若该ID只在一个模态中出现，则跳过

        mu_opt = opt_feats.mean(dim=0)
        mu_sar = sar_feats.mean(dim=0)
        mean_loss = torch.norm(mu_opt - mu_sar, p=2)

        # 方差对齐（逐维度方差）
        var_opt = opt_feats.var(dim=0, unbiased=False)  # [D]
        var_sar = sar_feats.var(dim=0, unbiased=False)  # [D]
        var_loss = torch.norm(var_opt - var_sar, p=2)

        mean_losses.append(mean_loss)
        var_losses.append(var_loss)

        # losses.append(F.mse_loss(mu_opt, mu_sar))
    if len(mean_losses) == 0:
        return torch.tensor(0.0, device=feat.device)

    
    alpha = 1.0  # 均值权重
    beta = 1.0   # 方差权重（可调超参）
    total_loss = alpha * (sum(mean_losses) / len(mean_losses)) + \
                 beta * (sum(var_losses) / len(var_losses))

    return total_loss


def make_loss(cfg, num_classes):  # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if "triplet" in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print("expected METRIC_LOSS_TYPE should be triplet" "but got {}".format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == "on":
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        # print("label smooth on, num_classes: ", num_classes)

    if sampler == "softmax":

        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif sampler == "softmax_triplet":  # 进入这个分支

        def loss_func(score, feat, target, target_cam, img_paths):
            if cfg.MODEL.METRIC_LOSS_TYPE == "triplet":
                if cfg.MODEL.IF_LABELSMOOTH == "on":
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else: # 进入这个分支
                    if isinstance(score, list):
                        print("score is list")
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        # print("score is not list")
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                        print("feat is list")
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                        # print("feat is not list")
                        TRI_LOSS = triplet(feat, target)[0]
                    
                    GLOBAL_MODALITY_LOSS = modality_alignment_loss(feat, img_paths)
                    LOCAL_MODALITY_LOSS = classwise_modality_alignment_loss(feat, target, img_paths)
                    # print(f"ID_LOSS: {ID_LOSS.item():.4f}, TRI_LOSS: {TRI_LOSS.item():.4f}, GLOBAL_MODALITY_LOSS: {GLOBAL_MODALITY_LOSS.item():.4f}, LOCAL_MODALITY_LOSS: {LOCAL_MODALITY_LOSS.item():.4f}")
                    # print(f" Weights -> ID: {cfg.MODEL.ID_LOSS_WEIGHT}, TRI: {cfg.MODEL.TRIPLET_LOSS_WEIGHT}, MODALITY: {cfg.MODEL.MODALITY_LOSS_WEIGHT}")

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS  + cfg.MODEL.MODALITY_LOSS_WEIGHT * LOCAL_MODALITY_LOSS
            else:
                print("expected METRIC_LOSS_TYPE should be triplet" "but got {}".format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print("expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center" "but got {}".format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion
