import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper, evaluate_result
def evaluate_video_level_auc(vid2abnormality, anno_file):
    from sklearn.metrics import roc_auc_score
    import os
    import numpy as np

    y_true = []
    y_score = []
    pred_map = {}

    for vid, value in vid2abnormality.items():
        base = os.path.basename(vid)
        arr = np.array(value).reshape(-1)

        if arr.size == 0:
            continue

        pred_map[base] = float(np.max(arr))

    for line in open(anno_file, 'r', encoding='utf-8'):
        parts = line.strip().split()

        if len(parts) < 3:
            continue

        video_name = os.path.basename(parts[0])

        try:
            label = int(parts[-1])
        except:
            continue

        if video_name not in pred_map:
            print('The video %s is excluded on the result!' % video_name)
            continue

        y_true.append(0 if label == 0 else 1)
        y_score.append(pred_map[video_name])

    if len(y_true) == 0:
        raise RuntimeError('No matched videos between prediction and annotation file.')

    if len(set(y_true)) < 2:
        raise RuntimeError('AUC needs both Normal and Anomaly videos, but only one class was found.')

    auc = roc_auc_score(y_true, y_score)
    return auc, auc

from utils.cluster import ClusterLoss, Normalize, BCE, PairEnum
from datasets.build import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
import mmcv
try:
    from apex import amp
except ImportError:
    amp = None
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending
from utils.config import get_config
from models import xclip
from einops import rearrange
import torch.nn.functional as F

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/k400/32_8.yaml')
    parser.add_argument("--opts",help="Modify config options by adding 'KEY VALUE' pairs. ",default=None,nargs='+',)
    parser.add_argument('--output', type=str, default="exp")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)
    # model parameters
    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    parser.add_argument('--w-smooth', default=0.01, type=float, help='weight of smooth loss')
    parser.add_argument('--w-sparse', default=0.001, type=float, help='weight of sparse loss')

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    train_data, val_data, test_data, train_loader, val_loader, test_loader, val_loader_train,_ = build_dataloader(logger, config)
    print("DEBUG len(train_data):", len(train_data))
    print("DEBUG len(val_data):", len(val_data))
    print("DEBUG len(test_data):", len(test_data))
    print("DEBUG len(train_loader):", len(train_loader))
    print("DEBUG len(val_loader):", len(val_loader))
    print("DEBUG len(test_loader):", len(test_loader))
    model, _ = xclip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH, 
                         device="cpu", jit=False, 
                         T=config.DATA.NUM_FRAMES,
                         droppath=config.MODEL.DROP_PATH_RATE, 
                         use_checkpoint=config.TRAIN.USE_CHECKPOINT, 
                         use_cache=config.MODEL.FIX_TEXT,
                         logger=logger,
                        )
    model = model.cuda()


    optimizer, _ = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    if config.TRAIN.OPT_LEVEL != 'O0' and amp is not None:
          model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.TRAIN.OPT_LEVEL)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True
    )
    start_epoch, best_epoch, max_auc = 0, 0, 0.0
    is_best = False
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        start_epoch, max_accuracy = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)

    text_labels = generate_text(train_data)
    
    if config.TEST.ONLY_TEST:
        out_path = os.path.join(config.OUTPUT, "test_scores.pkl")

        if os.path.exists(out_path):
            scores_dict = mmcv.load(out_path)
        else:
            scores_dict = validate(val_loader, text_labels, model, config, out_path)

        tmp_dict = {}

        for v_name in scores_dict["prd"].keys():
            video_name = os.path.basename(v_name)
            prd = np.array(scores_dict["prd"][v_name])

            if prd.ndim == 3:
                prd = prd.reshape(-1, prd.shape[-1])
            elif prd.ndim == 2:
                prd = prd.reshape(-1, prd.shape[-1])
            elif prd.ndim == 1:
                prd = prd.reshape(1, -1)

            if prd.shape[1] > 2:
                anomaly_score = np.max(prd[:, 1:], axis=1)
            else:
                anomaly_score = prd[:, 1]

            tmp_dict[video_name] = [anomaly_score]

        try:
            auc_all_p, auc_ano_p = evaluate_video_level_auc(tmp_dict, config.DATA.VAL_FILE)
            logger.info(f'AUC: [{auc_all_p:.3f}/{auc_ano_p:.3f}]\t')
        except Exception as e:
            logger.info(f"Skipping AUC evaluation because: {e}")

        return

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        train_one_epoch(epoch, model, optimizer, lr_scheduler, train_loader, text_labels, config)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_path = os.path.join(config.OUTPUT, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "max_auc": max_auc,
            }, save_path)
            logger.info(f"Saved checkpoint to {save_path}")

            epoch_saving(
                config,
                epoch,
                model.module,
                max_auc,
                optimizer,
                lr_scheduler,
                None,
                None,
                logger,
                config.OUTPUT,
                is_best
            )

def build_class_weights(device):
    class_weights = torch.tensor(
        [
            1.00,  # Normal
            1.20,   # Abuse
            1.40,   # Arrest
            1.50,   # Arson
            1.30,   # Assault
            1.30,   # Burglary
            1.20,   # Explosion
            1.60,   # Fighting
            0.90,   # RoadAccidents
            0.90,   # Robbery
            2.50,   # Shooting
            1.20,   # Shoplifting
            1.20,   # Stealing
            2.50,   # Vandalism
        ],
        dtype=torch.float32,
        device=device,
    )

    return class_weights
class FocalLoss(torch.nn.Module):
    def __init__(
        self,
        weight=None,
        gamma=2.0,
        label_smoothing=0.0,
        reduction="mean",
    ):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()

        if self.reduction == "sum":
            return focal_loss.sum()

        return focal_loss
def train_one_epoch(epoch, model, optimizer, lr_scheduler, train_loader, text_labels, config, data_dict=None):
    model.train()
    device = torch.device("cuda", torch.cuda.current_device())

    class_weights = torch.ones(
        config.DATA.NUM_CLASSES,
        device=device,
    )

    class_weights[10] = 2.0  # Shooting
    class_weights[13] = 2.0  # Vandalism

    criterion_multiclass = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.03,
    )

    optimizer.zero_grad()
    
    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()
    mil_loss_meter = AverageMeter()
    sm_loss_meter = AverageMeter()
    sp_loss_meter = AverageMeter()

    start = time.time()
    end = time.time()

    texts = text_labels.cuda(non_blocking=True)
    
    for idx, batch_data in enumerate(train_loader):
        images = batch_data["imgs"].cuda(non_blocking=True)[:,:1]
        label_id = batch_data["label"].cuda(non_blocking=True)[:,:1]
        label_id = label_id.reshape(-1)
        bz = images.shape[0]
        a_aug = images.shape[1]
        n_clips = images.shape[2]

        images = rearrange(images, 'b a k c t h w -> (b a k) t c h w')# bz*num_aug*num_clips,num_frames,h,w

        if texts.shape[0] == 1:
            texts = texts.view(1, -1)

        output = model(images, texts)

        # MIL loss on max scores among bags
        logits = rearrange(
            output["y"],
            "(b a k) c -> (b a) k c",
            b=bz,
            a=a_aug,
        )

        scores = F.softmax(logits, dim=-1)

        # ---------------------------------------------------------
        # Check model output before computing loss
        # ---------------------------------------------------------
        if not torch.isfinite(logits).all():
            print(
                f"\n[SKIP] Non-finite logits "
                f"at epoch={epoch}, batch={idx}"
            )
            optimizer.zero_grad()
            continue

        if not torch.isfinite(scores).all():
            print(
                f"\n[SKIP] Non-finite scores "
                f"at epoch={epoch}, batch={idx}"
            )
            optimizer.zero_grad()
            continue

        # Normal class is class 0
        scores_nor = scores[:, :, 0]

        # All classes 1 تا 13 are anomaly classes
        scores_ano = torch.max(
            scores[:, :, 1:],
            dim=-1
        ).values

      
        max_prob_ano, _ = torch.max(
            scores_ano,
            dim=-1
        )
 
        max_prob_nor, _ = torch.max(
            scores_nor,
            dim=-1
        )

        num_clips = logits.size(1) 
        topk_k = min(4, num_clips)

        topk_scores, topk_indices = torch.topk(
            scores,
            k=topk_k,
            dim=1
        )

        topk_logits = torch.gather(
            logits,
            dim=1,
            index=topk_indices
        )

        logits_video = topk_logits.mean(dim=1)

        selected_scores = topk_scores.mean(dim=1)

        max_prob_video = torch.max(
            selected_scores,
            dim=-1
        ).values

        if not torch.isfinite(logits_video).all():
            print(
                f"\n[SKIP] Non-finite logits_video "
                f"at epoch={epoch}, batch={idx}"
            )
            optimizer.zero_grad()
            continue

        if idx == 0 and epoch == 0:
            print("DEBUG logits shape:", logits.shape)
            print("DEBUG scores shape:", scores.shape)
            print("DEBUG topk_indices shape:", topk_indices.shape)
            print("DEBUG topk_logits shape:", topk_logits.shape)
            print("DEBUG logits_video shape:", logits_video.shape)
            print("DEBUG selected_scores shape:", selected_scores.shape)
            print("DEBUG label_id shape:", label_id.shape)
            print("DEBUG label_id min/max:",
                label_id.min().item(),
                label_id.max().item()
            )
            print("DEBUG NUM_CLASSES:",
                config.DATA.NUM_CLASSES
            )
            print("DEBUG TOP-K:", topk_k)
        # ---------------------------------------------------------
        # MIL loss
        # ---------------------------------------------------------
        if logits_video.ndim != 2:
            raise ValueError(
                f"logits_video must have shape [B, C], "
                f"but received {tuple(logits_video.shape)}"
            )



        label_id = label_id.long().reshape(-1)

        if logits_video.shape[0] != label_id.shape[0]:
            raise ValueError(
                f"Batch mismatch: logits={tuple(logits_video.shape)}, "
                f"labels={tuple(label_id.shape)}"
            )

        if logits_video.shape[1] != config.DATA.NUM_CLASSES:
            raise ValueError(
                f"Expected {config.DATA.NUM_CLASSES} classes, "
                f"but model produced {logits_video.shape[1]}"
            )

        if label_id.min().item() < 0:
            raise ValueError("Negative label detected")

        if label_id.max().item() >= config.DATA.NUM_CLASSES:
            raise ValueError(
                f"Invalid label {label_id.max().item()} for "
                f"{config.DATA.NUM_CLASSES} classes"
            )
        if epoch == 0 and idx == 0:
            print("\n===== MULTICLASS SANITY CHECK =====")
            print("logits_video shape:", logits_video.shape)
            print("label_id shape:", label_id.shape)
            print("label values:", label_id.detach().cpu().tolist())
            print("NUM_CLASSES:", config.DATA.NUM_CLASSES)
            print("===================================\n")


        loss_mil = criterion_multiclass(
            logits_video,
            label_id,
        )

        # Overall anomaly probability across classes 1 تا 13
        anomaly_scores = torch.max(
            scores[:, :, 1:],
            dim=-1
        ).values

        smoothed_scores = (
            anomaly_scores[:, 1:]
            - anomaly_scores[:, :-1]
        )

        smoothed_loss = (
            smoothed_scores
            .pow(2)
            .sum(dim=-1)
            .mean()
        )

        sparsity_loss = (
            anomaly_scores
            .sum(dim=-1)
            .mean()
        )

        w_smooth = args.w_smooth
        w_sparse = args.w_sparse

        total_loss = (
            loss_mil
            + smoothed_loss * w_smooth
            + sparsity_loss * w_sparse
        )

        total_loss = (
            total_loss
            / config.TRAIN.ACCUMULATION_STEPS
        )

        # ---------------------------------------------------------
        # Check losses
        # ---------------------------------------------------------
        if not torch.isfinite(loss_mil):
            print(
                f"\n[SKIP] Non-finite loss_mil "
                f"at epoch={epoch}, batch={idx}"
            )
            optimizer.zero_grad()
            continue

        if not torch.isfinite(smoothed_loss):
            print(
                f"\n[SKIP] Non-finite smoothed_loss "
                f"at epoch={epoch}, batch={idx}"
            )
            optimizer.zero_grad()
            continue

        if not torch.isfinite(sparsity_loss):
            print(
                f"\n[SKIP] Non-finite sparsity_loss "
                f"at epoch={epoch}, batch={idx}"
            )
            optimizer.zero_grad()
            continue

        if not torch.isfinite(total_loss):
            print(
                f"\n[SKIP] Non-finite total_loss "
                f"at epoch={epoch}, batch={idx}"
            )
            optimizer.zero_grad()
            continue

        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()

        if config.TRAIN.OPT_LEVEL != "O0" and amp is not None:
            with amp.scale_loss(
                total_loss,
                optimizer
            ) as scaled_loss:
                scaled_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                amp.master_params(optimizer),
                max_norm=1.0
            )

        else:
            total_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

    


        if not torch.isfinite(torch.as_tensor(grad_norm)):
            print(
                f"\n[SKIP] Non-finite gradient norm "
                f"at epoch={epoch}, batch={idx}: "
                f"{grad_norm}"
            )
            optimizer.zero_grad()
            continue

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (
                (idx + 1) %
                config.TRAIN.ACCUMULATION_STEPS
                == 0
            ):
                optimizer.step()
                optimizer.zero_grad()

                lr_scheduler.step_update(
                    epoch * num_steps + idx
                )
        else:
          optimizer.step()

          lr_scheduler.step_update(
              epoch * num_steps + idx
          )

        torch.cuda.synchronize()
        
        tot_loss_meter.update(total_loss.item(), len(label_id))
        mil_loss_meter.update(loss_mil.item(), len(label_id))
        sm_loss_meter.update((smoothed_loss * w_smooth).item(), len(label_id))
        sp_loss_meter.update((sparsity_loss * w_sparse).item(), len(label_id))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'tot {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mil {mil_loss_meter.val:.4f} ({mil_loss_meter.avg:.4f})\t'
                f'sm {sm_loss_meter.val:.4f} ({sm_loss_meter.avg:.4f})\t'
                f'sp {sp_loss_meter.val:.4f} ({sp_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(data_loader, text_labels, model, config, out_path):
    model.eval()
    print("######## ENTERED VALIDATE ########", flush=True)
    vid_list = []

    anno_file = config.DATA.VAL_FILE

    with open(anno_file, "r", encoding="utf-8") as fin:
        for line in fin:
            line_split = line.strip().split()

            if not line_split:
                continue

            filename = line_split[0].split("/")[-1]
            vid_list.append(filename)

    text_inputs = text_labels.cuda(non_blocking=True)

    logger.info(
        f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference"
    )

    scores_dict = {
        "prd": {}
    }

    for idx, batch_data in enumerate(data_loader):
        if idx < 3:
            print("\nDEBUG VALIDATION BATCH:", idx)
            print("batch_data keys:", batch_data.keys())
            print("vid:", batch_data["vid"])
            print("vid type:", type(batch_data["vid"]))
            print("imgs shape:", batch_data["imgs"].shape)
            print("label shape:", batch_data["label"].shape)
        _image = batch_data["imgs"].cuda(
            non_blocking=True
        )

        label_id = batch_data["label"]
        label_id = label_id.reshape(-1)

        b, n, c, t, h, w = _image.size()

        _image = rearrange(
            _image,
            "b n c t h w -> (b n) t c h w"
        )

        output = model(
            _image,
            text_inputs
        )

        if not torch.isfinite(output["y"]).all():
            raise FloatingPointError(
                f"Validation model output contains NaN/Inf "
                f"at batch {idx}"
            )

        scores_prd = F.softmax(
            output["y"],
            dim=-1
        )

        if not torch.isfinite(scores_prd).all():
            raise FloatingPointError(
                f"Validation probabilities contain NaN/Inf "
                f"at batch {idx}"
            )

        scores_prd = rearrange(
            scores_prd,
            "(b n) c -> b n c",
            b=b
        )

        scores_np_prd = (
            scores_prd
            .detach()
            .cpu()
            .numpy()
        )
        if idx < 3:
            print("scores_np_prd shape:", scores_np_prd.shape)
            print("batch size b:", b)
            print("num views n:", n)
            print("======================================")

        for ind in range(scores_np_prd.shape[0]):
            v_name = vid_list[
                batch_data["vid"][ind]
            ]

            if v_name not in scores_dict["prd"]:
                scores_dict["prd"][v_name] = []

            scores_dict["prd"][v_name].append(
                scores_np_prd[ind]
            )

        if idx % 100 == 0 and len(data_loader) >= 100:
            logger.info(
                f"Test: [{idx}/{len(data_loader)}]\t"
            )

    tmp_dict = {}

    for v_name in scores_dict["prd"]:
        video_name = os.path.basename(v_name)

        prd = np.array(
            scores_dict["prd"][v_name]
        )

        if prd.ndim == 3:
            prd = prd.reshape(
                -1,
                prd.shape[-1]
            )
        elif prd.ndim == 2:
            prd = prd.reshape(
                -1,
                prd.shape[-1]
            )
        elif prd.ndim == 1:
            prd = prd.reshape(
                1,
                -1
            )

        if prd.shape[1] > 2:
            anomaly_score = np.max(
                prd[:, 1:],
                axis=1
            )
        else:
            anomaly_score = prd[:, 1]

        tmp_dict[video_name] = [
            anomaly_score
        ]

    logger.info(
        f"writing results to {out_path}"
    )

    mmcv.dump(
        scores_dict,
        out_path
    )

    return scores_dict



if __name__ == '__main__':
    # prepare config
    args, config = parse_option()

    # init_distributed
    # init_distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = 0
        world_size = 1

    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.local_rank < 0:
        args.local_rank = 0

    print("DEBUG local_rank after fix:", args.local_rank)

    torch.cuda.set_device(args.local_rank)

    torch.distributed.init_process_group(
        backend='gloo',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)
    
    # logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")
    
    # save config 
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)

    main(config)
