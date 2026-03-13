import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP_eval_queryAdd, R1_mAP_eval_galleryAdd, R1_mAP_eval_queryAdd_galleryAdd
from torch.cuda import amp
import torch.distributed as dist
from loss import clip_loss



def do_train(cfg, model, center_criterion, train_loader, val_loader, o2s_val_loader, s2o_val_loader, optimizer, optimizer_center, scheduler, loss_fn, num_query, num_o2s_query, num_s2o_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info("start training")
    _LOCAL_PROCESS_GROUP = None

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    # train
    if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
        model.module.train_with_single()
    else:
        model.train_with_single()
    best_mAP = 0.0
    best_rank1 = 0.0
    best_mAP_tuple = (0.0, 0.0, 0.0, 0.0)
    best_rank1_tuple = (0.0, 0.0, 0.0, 0.0)
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view, img_paths, img_wh) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            img_wh = img_wh.to(device)
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, img_wh=img_wh)
                loss = loss_fn(score, feat, target, target_cam, img_paths)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if "center" in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= 1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                        epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]
                    )
                )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch, time_per_batch, train_loader.batch_size / time_per_batch
                )
            )

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)))
            else:
                torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _, img_wh) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            img_wh = img_wh.to(device)
                            feat = model(img, cam_label=camids, img_wh=img_wh)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, img_paths, img_wh) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        img_wh = img_wh.to(device)
                        feat = model(img, cam_label=camids, img_wh=img_wh)
                        evaluator.update((feat, vid, camid, img_paths))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()

                o2s_evaluator = R1_mAP_eval(num_o2s_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
                o2s_evaluator.reset()
                for n_iter, (img, vid, camid, camids, target_view, img_paths, img_wh) in enumerate(o2s_val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        img_wh = img_wh.to(device)
                        feat = model(img, cam_label=camids, img_wh=img_wh)
                        o2s_evaluator.update((feat, vid, camid, img_paths))
                o2s_cmc, o2s_mAP, _, _, _, _, _ = o2s_evaluator.compute()

                s2o_evaluator = R1_mAP_eval(num_s2o_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
                s2o_evaluator.reset()
                for n_iter, (img, vid, camid, camids, target_view, img_paths, img_wh) in enumerate(s2o_val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        img_wh = img_wh.to(device)
                        feat = model(img, cam_label=camids, img_wh=img_wh)
                        s2o_evaluator.update((feat, vid, camid, img_paths))
                s2o_cmc, s2o_mAP, _, _, _, _, _ = s2o_evaluator.compute()

                if mAP > best_mAP:
                    best_mAP = mAP
                    best_mAP_tuple = (mAP, cmc[0], cmc[4], cmc[9])
                    if cfg.MODEL.DIST_TRAIN:
                        torch.save(model.module.state_dict(), os.path.join(cfg.OUTPUT_DIR, "best_model_mAP.pth"))
                    else:
                        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "best_model_mAP.pth"))
                if cmc[0] > best_rank1:
                    best_rank1 = cmc[0]
                    best_rank1_tuple = (mAP, cmc[0], cmc[4], cmc[9])
                    if cfg.MODEL.DIST_TRAIN:
                        torch.save(model.module.state_dict(), os.path.join(cfg.OUTPUT_DIR, "best_model_rank1.pth"))
                    else:
                        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "best_model_rank1.pth"))
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info(f"all query-gallery: mAP: {mAP:.1%}, rank-1: {cmc[0]:.1%}, rank-5: {cmc[4]:.1%}, rank-10: {cmc[9]:.1%}")
                logger.info(f"o2s query-gallery: mAP: {o2s_mAP:.1%}, rank-1: {o2s_cmc[0]:.1%}, rank-5: {o2s_cmc[4]:.1%}, rank-10: {o2s_cmc[9]:.1%}")
                logger.info(f"s2o query-gallery: mAP: {s2o_mAP:.1%}, rank-1: {s2o_cmc[0]:.1%}, rank-5: {s2o_cmc[4]:.1%}, rank-10: {s2o_cmc[9]:.1%}")
                # logger.info("mAP: {:.1%}".format(mAP))
                # for r in [1, 5, 10]:
                #     logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()
        logger.info(f"Current best mAP set: {best_mAP_tuple[0]:.1%}, {best_mAP_tuple[1]:.1%}, {best_mAP_tuple[2]:.1%}, {best_mAP_tuple[3]:.1%}")
        logger.info(f"Current best rank1 set: {best_rank1_tuple[0]:.1%}, {best_rank1_tuple[1]:.1%}, {best_rank1_tuple[2]:.1%}, {best_rank1_tuple[3]:.1%}")

    logger.info(f"best mAP set: {best_mAP_tuple[0]:.1%}, {best_mAP_tuple[1]:.1%}, {best_mAP_tuple[2]:.1%}, {best_mAP_tuple[3]:.1%}")
    logger.info(f"best rank1 set: {best_rank1_tuple[0]:.1%}, {best_rank1_tuple[1]:.1%}, {best_rank1_tuple[2]:.1%}, {best_rank1_tuple[3]:.1%}")



def do_inference_queryAdd_galleryAdd(cfg, model, val_loader, queryAdd_loader, galleryAdd_loader, num_query, alpha):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing with queryAdd and galleryAdd")
    
    evaluator = R1_mAP_eval_queryAdd_galleryAdd(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, alpha=alpha)
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for inference".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath, img_wh) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            img_wh = img_wh.to(device)
            feat = model(img, cam_label=camids, img_wh=img_wh)
            evaluator.update((feat, pid, camid, imgpath))
            img_path_list.extend(imgpath)
    
    queryAdd_path_list = []
    for n_iter, (img, pid, camid, camids, target_view, imgpath, img_wh) in enumerate(queryAdd_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            img_wh = img_wh.to(device)
            feat = model(img, cam_label=camids, img_wh=img_wh)
            evaluator.update_queryAdd((feat, pid, camid))
            queryAdd_path_list.extend(imgpath)
            
    galleryAdd_path_list = []
    for n_iter, (img, pid, camid, camids, target_view, imgpath, img_wh) in enumerate(galleryAdd_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            img_wh = img_wh.to(device)
            feat = model(img, cam_label=camids, img_wh=img_wh)
            evaluator.update_galleryAdd((feat, pid, camid))
            galleryAdd_path_list.extend(imgpath)
            
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
    
    
    


def do_inference_galleryAdd(cfg, model, val_loader, galleryAdd_loader, num_query, alpha):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing with galleryAdd")

    evaluator = R1_mAP_eval_galleryAdd(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, alpha=alpha)
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for inference".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath, img_wh) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            img_wh = img_wh.to(device)
            feat = model(img, cam_label=camids, img_wh=img_wh)
            evaluator.update((feat, pid, camid, imgpath))
            img_path_list.extend(imgpath)

    galleryAdd_path_list = []
    for n_iter, (img, pid, camid, camids, target_view, imgpath, img_wh) in enumerate(galleryAdd_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            img_wh = img_wh.to(device)
            feat = model(img, cam_label=camids, img_wh=img_wh)
            evaluator.update_galleryAdd((feat, pid, camid))
            galleryAdd_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]



def do_inference_queryAdd(cfg, model, val_loader, queryAdd_loader, num_query, alpha):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing with queryAdd")

    evaluator = R1_mAP_eval_queryAdd(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, alpha=alpha)
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for inference".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath, img_wh) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            img_wh = img_wh.to(device)
            feat = model(img, cam_label=camids, img_wh=img_wh)
            evaluator.update((feat, pid, camid, imgpath))
            img_path_list.extend(imgpath)
    
    queryAdd_path_list = []
    for n_iter, (img, pid, camid, camids, target_view, imgpath, img_wh) in enumerate(queryAdd_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            img_wh = img_wh.to(device)
            feat = model(img, cam_label=camids, img_wh=img_wh)
            evaluator.update_queryAdd((feat, pid, camid))
            queryAdd_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]



class ModelAdapter(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, camids, img_wh):
        return self.model(x, cam_label=camids, img_wh=img_wh)




def do_inference(cfg, model, val_loader, num_query):

    # benchmark_model(cfg, model, val_loader)
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for inference".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath, img_wh) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            img_wh = img_wh.to(device)
            feat = model(img, cam_label=camids, img_wh=img_wh)
            evaluator.update((feat, pid, camid, imgpath))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


