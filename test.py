import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference, do_inference_queryAdd, do_inference_galleryAdd,do_inference_queryAdd_galleryAdd
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TransOSS Testing")
    parser.add_argument("--config_file", default="configs/HOSS/hoss_transoss.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, o2s_val_loader, num_o2s_query, s2o_val_loader, num_s2o_query, num_classes, camera_num, queryAdd_loader, galleryAdd_loader = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num)
    model.load_param(cfg.TEST.WEIGHT)
    # do_inference(cfg, model, val_loader, num_query)
    print('o2s inference')
    # do_inference(cfg, model, o2s_val_loader, num_o2s_query)
    # print('s2o inference')
    do_inference(cfg, model,val_loader, num_query)
    alpha = 0.02
    do_inference_queryAdd(cfg, model, o2s_val_loader, queryAdd_loader, num_o2s_query, alpha)
    do_inference_galleryAdd(cfg, model, s2o_val_loader, galleryAdd_loader, num_s2o_query, alpha)
    do_inference_queryAdd_galleryAdd(cfg, model, val_loader, queryAdd_loader, galleryAdd_loader, num_query, alpha)
