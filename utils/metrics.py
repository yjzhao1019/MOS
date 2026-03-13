import torch
import numpy as np
import os
from utils.reranking import re_ranking
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat




def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_img_paths, g_img_paths, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_query_results = []
    num_valid_q = 0.  # number of valid query
    cnt = 0
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        sorted_g_paths = [g_img_paths[i] for i in order]
        # remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        # keep = np.invert(remove)
        keep = np.ones_like(g_pids, dtype=bool)

        sorted_g_pids = g_pids[order]
        sorted_matches = (sorted_g_pids == q_pid).astype(np.int32)
        if sorted_matches[0] == 1:
            cnt += 1

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval_queryAdd_galleryAdd():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False, alpha=0.5):
        super(R1_mAP_eval_queryAdd_galleryAdd, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.alpha = alpha
        
    def reset(self):
        self.feats = []
        self.pids = []
        self.queryAdd_feats = []
        self.galleryAdd_feats = []
        self.camids = []
        self.img_paths = []
        
    def update(self, output):  # called once for each batch
        feat, pid, camid, img_path = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.img_paths.extend(img_path)
        self.camids.extend(np.asarray(camid))
    
    def update_queryAdd(self, output):  # called once for each batch
        feat, pid, camid = output
        self.queryAdd_feats.append(feat.cpu())
    
    def update_galleryAdd(self, output):  # called once for each batch
        feat, pid, camid = output
        self.galleryAdd_feats.append(feat.cpu())
        


    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_img_paths = self.img_paths[:self.num_query]
        # gallery
        gf = feats[self.num_query:]
        print(f'gf shape: {gf.shape}')
        g_pids = np.asarray(self.pids[self.num_query:])
        g_img_paths = self.img_paths[self.num_query:]
        
        # queryAdd
        queryAdd_f = torch.cat(self.queryAdd_feats, dim=0)
        if self.feat_norm:
            queryAdd_f = torch.nn.functional.normalize(queryAdd_f, dim=1, p=2)
        print(f'queryAdd_f shape: {queryAdd_f.shape}')

        # switch queryAdd_f to list
        queryAdd_f_list = queryAdd_f.cpu().numpy().tolist()
        print(f'queryAdd_f_list length: {len(queryAdd_f_list)}')
        new_qaf = []
        qaf_iter = iter(queryAdd_f_list)
        for img_path in q_img_paths:
            if img_path.endswith('RGB.tif'):
                group = []
                for _ in range(5):
                    group.append(next(qaf_iter))
                new_qaf.extend(group)
            else:
                group = []
                for _ in range(5):
                    group.append(torch.zeros(len(queryAdd_f_list[0])))
                new_qaf.extend(group)
            # print(f'Processed img_path: {img_path}, group size: {len(group)}')

        queryAdd_f = torch.tensor(new_qaf)
        print(f'queryAdd_f shape after processing: {queryAdd_f.shape}')
        assert queryAdd_f.shape[0] % qf.shape[0] == 0, "样本数量必须是整数倍"
        ratio = queryAdd_f.shape[0] // qf.shape[0]
        queryAdd_f = queryAdd_f.view(qf.shape[0], ratio, -1)
        print(f'queryAdd_f reshaped: {queryAdd_f.shape}')
        queryAdd_f = torch.mean(queryAdd_f, dim=1)
        print(f'queryAdd_f averaged: {queryAdd_f.shape}')

        # galleryAdd
        galleryAdd_f = torch.cat(self.galleryAdd_feats, dim=0)
        if self.feat_norm:
            galleryAdd_f = torch.nn.functional.normalize(galleryAdd_f, dim=1, p=2)
        print(f'galleryAdd_f shape: {galleryAdd_f.shape}')

        # switch queryAdd_f to list
        galleryAdd_f_list = galleryAdd_f.cpu().numpy().tolist()
        print(f'galleryAdd_f_list length: {len(galleryAdd_f_list)}')
        new_gaf = []
        gaf_iter = iter(galleryAdd_f_list)
        for img_path in g_img_paths:
            if img_path.endswith('RGB.tif'):
                group = []
                for _ in range(5):
                    group.append(next(gaf_iter))
                new_gaf.extend(group)
            else:
                group = []
                for _ in range(5):
                    group.append(torch.zeros(len(galleryAdd_f_list[0])).tolist())
                new_gaf.extend(group)

        galleryAdd_f = torch.tensor(new_gaf)
        print(f'galleryAdd_f shape after processing: {galleryAdd_f.shape}')


        assert galleryAdd_f.shape[0] % gf.shape[0] == 0, "样本数量必须是整数倍"
        ratio = galleryAdd_f.shape[0] // gf.shape[0]
        galleryAdd_f = galleryAdd_f.view(gf.shape[0], ratio, -1)
        print(f'galleryAdd_f reshaped: {galleryAdd_f.shape}')
        galleryAdd_f = torch.mean(galleryAdd_f, dim=1)
        print(f'galleryAdd_f averaged: {galleryAdd_f.shape}')

        alpha = self.alpha
        new_qf = (1 - alpha) * qf + alpha * queryAdd_f
        new_qf = torch.nn.functional.normalize(new_qf, dim=1, p=2)
        new_gf = (1 - alpha) * gf + alpha * galleryAdd_f
        new_gf = torch.nn.functional.normalize(new_gf, dim=1, p=2)

        all_feats = torch.cat([new_qf, new_gf], dim=0)
        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(new_qf, new_gf, k1=50, k2=15, lambda_value=0.3)

        else:
            # print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(new_qf, new_gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_img_paths, g_img_paths)

        
        print(f'alpha: {alpha:.2f}, mAP: {mAP:.4f}, Rank-1: {cmc[0]:.4f}, Rank-5: {cmc[4]:.4f}, Rank-10: {cmc[9]:.4f}')

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf


class R1_mAP_eval_galleryAdd():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False, alpha=0.5):
        super(R1_mAP_eval_galleryAdd, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.alpha = alpha

    def reset(self):
        self.feats = []
        self.pids = []
        self.galleryAdd_feats = []
        self.camids = []
        self.img_paths = []

    def update(self, output):  # called once for each batch
        feat, pid, camid, img_path = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.img_paths.extend(img_path)
    
    def update_galleryAdd(self, output):  # called once for each batch
        feat, pid, camid = output
        self.galleryAdd_feats.append(feat.cpu())

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_img_paths = self.img_paths[:self.num_query]
        # gallery
        gf = feats[self.num_query:]
        print(f'gf shape: {gf.shape}')
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_img_paths = self.img_paths[self.num_query:]

        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            # print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)

        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_img_paths, g_img_paths)
        print(f'alpha: {self.alpha:.2f}, mAP: {mAP:.4f}, Rank-1: {cmc[0]:.4f}, Rank-5: {cmc[4]:.4f}, Rank-10: {cmc[9]:.4f}')


        # galleryAdd
        galleryAdd_f = torch.cat(self.galleryAdd_feats, dim=0)
        if self.feat_norm:
            galleryAdd_f = torch.nn.functional.normalize(galleryAdd_f, dim=1, p=2)
        print(f'galleryAdd_f shape: {galleryAdd_f.shape}')


        # switch queryAdd_f to list
        galleryAdd_f_list = galleryAdd_f.cpu().numpy().tolist()
        print(f'galleryAdd_f_list length: {len(galleryAdd_f_list)}')
        new_gaf = []
        gaf_iter = iter(galleryAdd_f_list)
        for img_path in g_img_paths:
            if img_path.endswith('RGB.tif'):
                group = []
                for _ in range(5):
                    group.append(next(gaf_iter))
                new_gaf.extend(group)
            else:
                group = []
                for _ in range(5):
                    group.append(torch.zeros(len(galleryAdd_f_list[0])).tolist())
                new_gaf.extend(group)
            # print(f'new_gaf length: {len(new_gaf)}')

        galleryAdd_f = torch.tensor(new_gaf)
        print(f'galleryAdd_f shape after processing: {galleryAdd_f.shape}')

        assert galleryAdd_f.shape[0] % gf.shape[0] == 0, "样本数量必须是整数倍"
        ratio = galleryAdd_f.shape[0] // gf.shape[0]
        galleryAdd_f = galleryAdd_f.view(gf.shape[0], ratio, -1)
        print(f'galleryAdd_f reshaped: {galleryAdd_f.shape}')
        galleryAdd_f = torch.mean(galleryAdd_f, dim=1)
        print(f'galleryAdd_f averaged: {galleryAdd_f.shape}')

        
        alpha = self.alpha
        new_gf = (1 - alpha) * gf + alpha * galleryAdd_f
        new_gf = torch.nn.functional.normalize(new_gf, dim=1, p=2)
    

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, new_gf, k1=50, k2=15, lambda_value=0.3)

        else:
            # print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, new_gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_img_paths, g_img_paths)
        print(f'alpha: {alpha:.2f}, mAP: {mAP:.4f}, Rank-1: {cmc[0]:.4f}, Rank-5: {cmc[4]:.4f}, Rank-10: {cmc[9]:.4f}')
        return cmc, mAP, distmat, self.pids, self.camids, qf, gf


class R1_mAP_eval_queryAdd():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False, alpha=0.5):
        super(R1_mAP_eval_queryAdd, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.alpha = alpha
    def reset(self):
        self.feats = []
        self.pids = []
        self.queryAdd_feats = []
        self.camids = []
        self.img_paths = []

    def update(self, output):  # called once for each batch
        feat, pid, camid, img_path = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.img_paths.extend(img_path)
    
    def update_queryAdd(self, output):  # called once for each batch
        feat, pid, camid = output
        self.queryAdd_feats.append(feat.cpu())

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        print(f'qf shape: {qf.shape}')
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_img_paths = self.img_paths[:self.num_query]
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_img_paths = self.img_paths[self.num_query:]
        g_camids = np.asarray(self.camids[self.num_query:])

        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            # print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)

        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_img_paths, g_img_paths)
        print(f'original mAP: {mAP:.4f}, Rank-1: {cmc[0]:.4f}, Rank-5: {cmc[4]:.4f}, Rank-10: {cmc[9]:.4f}')


        # queryAdd
        queryAdd_f = torch.cat(self.queryAdd_feats, dim=0)
        if self.feat_norm:
            queryAdd_f = torch.nn.functional.normalize(queryAdd_f, dim=1, p=2)
        print(f'queryAdd_f shape: {queryAdd_f.shape}')

        # switch queryAdd_f to list
        queryAdd_f_list = queryAdd_f.cpu().numpy().tolist()
        print(f'queryAdd_f_list length: {len(queryAdd_f_list)}')
        new_qaf = []
        qaf_iter = iter(queryAdd_f_list)
        for img_path in q_img_paths:
            if img_path.endswith('RGB.tif'):
                group = []
                for _ in range(5):
                    group.append(next(qaf_iter))
                new_qaf.extend(group)
            else:
                group = []
                for _ in range(5):
                    group.append(torch.zeros(len(queryAdd_f_list[0])))
                new_qaf.extend(group)
            # print(f'new_qaf length: {len(new_qaf)}')

        queryAdd_f = torch.tensor(new_qaf)
        print(f'queryAdd_f shape after processing: {queryAdd_f.shape}')

        assert queryAdd_f.shape[0] % qf.shape[0] == 0, "样本数量必须是整数倍"
        ratio = queryAdd_f.shape[0] // qf.shape[0]
        queryAdd_f = queryAdd_f.view(qf.shape[0], ratio, -1)
        print(f'queryAdd_f reshaped: {queryAdd_f.shape}')
        queryAdd_f = torch.mean(queryAdd_f, dim=1)
        print(f'queryAdd_f averaged: {queryAdd_f.shape}')

        alpha = self.alpha
        new_qf = (1 - alpha) * qf + alpha * queryAdd_f
        new_qf = torch.nn.functional.normalize(new_qf, dim=1, p=2)
    

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(new_qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            # print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(new_qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_img_paths, g_img_paths)
        print(f'alpha: {alpha:.2f}, mAP: {mAP:.4f}, Rank-1: {cmc[0]:.4f}, Rank-5: {cmc[4]:.4f}, Rank-10: {cmc[9]:.4f}')
        return cmc, mAP, distmat, self.pids, self.camids, qf, gf


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.img_paths = []

    def update(self, output):  # called once for each batch
        feat, pid, camid, img_path = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.img_paths.extend(img_path)

    def compute(self):  # called after each epoch
        # self.tsne_opt_sar()
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_img_paths = self.img_paths[:self.num_query]
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_img_paths = self.img_paths[self.num_query:]

        g_camids = np.asarray(self.camids[self.num_query:])

        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_img_paths, g_img_paths)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf



