import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import model
import train_config
from dataset import CUDAPrefetcher, ImageDataset
from utils import accuracy, load_pretrained_state_dict, load_resume_state_dict, make_directory, save_checkpoint, \
    Summary, AverageMeter, ProgressMeter
from test import test

# --------------------------
# Debug helpers / toggles
# --------------------------
DEBUG = os.getenv("DEBUG_TRAIN", "1") == "1"
DEBUG_GRAD_NORM = os.getenv("DEBUG_GRAD_NORM", "1") == "1"  # can be heavy; prints infrequently
DEBUG_PRINT_FREQ = int(os.getenv("DEBUG_PRINT_FREQ", str(getattr(train_config, "train_print_frequency", 50))))

def dprint(*args, rank=0, this_rank=None, **kwargs):
    """Conditional debug print. Defaults to printing only on rank 0 unless 'rank=\"all\"' is passed."""
    if not DEBUG:
        return
    if rank == "all":
        print(*args, **kwargs)
    else:
        if this_rank is None or this_rank == rank:
            print(*args, **kwargs)

def current_lr(optimizer: optim.Optimizer):
    lrs = [pg.get("lr", None) for pg in optimizer.param_groups]
    return lrs

def has_any_nan_or_inf(t: torch.Tensor):
    return not torch.isfinite(t).all().item()

def grad_total_norm(parameters, norm_type=2.0):
    if not DEBUG_GRAD_NORM:
        return None
    params = [p.grad.detach() for p in parameters if p.grad is not None]
    if len(params) == 0:
        return 0.0
    device = params[0].device
    norms = torch.stack([torch.norm(p, p=norm_type).to(device) for p in params])
    total = torch.norm(norms, p=norm_type).item()
    return total

def main(seed):
    device = torch.device(train_config.device)

    # Initialize distributed process group (Gloo backend for CPU)
    dist.init_process_group(backend="gloo")
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    dprint(f"[Init] rank={rank}/{world_size-1} backend=gloo device={device}", this_rank=rank)
    dprint(f"[Init] CUDA available={torch.cuda.is_available()} cudnn.benchmark=True", this_rank=rank)
    dprint(f"[Init] AMP enabled (torch.cuda.amp)={True}", this_rank=rank)

    # Fixed random number seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dprint(f"[Seed] seed={seed}", this_rank=rank)

    # cudnn
    cudnn.benchmark = True

    # Initialize the gradient scaler
    scaler = amp.GradScaler()
    dprint(f"[AMP] GradScaler enabled={True}", this_rank=rank)

    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training network evaluation indicators
    best_acc1 = 0.0

    train_prefetcher, valid_prefetcher = load_dataset(device=device, _rank=rank)
    vgg_model, ema_vgg_model = build_model(device=device, _rank=rank)
    criterion = define_loss(device=device, _rank=rank)
    optimizer = define_optimizer(vgg_model, _rank=rank)
    scheduler = define_scheduler(optimizer, _rank=rank)

    if train_config.pretrained_model_weights_path:
        vgg_model = load_pretrained_state_dict(vgg_model, train_config.pretrained_model_weights_path)
        dprint(f"[Weights] Loaded pretrained weights: {train_config.pretrained_model_weights_path}", this_rank=rank)
    else:
        dprint("[Weights] Pretrained model weights not found.", this_rank=rank)

    if train_config.resume_model_weights_path:
        vgg_model, ema_vgg_model, start_epoch, best_acc1, optimizer, scheduler = load_resume_state_dict(
            vgg_model,
            train_config.resume_model_weights_path,
            ema_vgg_model,
            optimizer,
            scheduler
        )
        dprint(f"[Resume] Loaded from {train_config.resume_model_weights_path} | start_epoch={start_epoch} best_acc1={best_acc1:.4f}", this_rank=rank)
    else:
        dprint("[Resume] Not found. Training from scratch.", this_rank=rank)

    # Create a experiment results
    samples_dir = os.path.join("samples", train_config.exp_name)
    results_dir = os.path.join("results", train_config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)
    if rank == 0:
        dprint(f"[Dirs] samples_dir={samples_dir}", this_rank=rank)
        dprint(f"[Dirs] results_dir={results_dir}", this_rank=rank)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", train_config.exp_name))
    if rank == 0:
        dprint(f"[TB] Logging to samples/logs/{train_config.exp_name}", this_rank=rank)

    for epoch in range(start_epoch, train_config.epochs):
        if rank == 0:
            dprint(f"\n[Epoch {epoch+1}/{train_config.epochs}] Starting | lr={current_lr(optimizer)}", this_rank=rank)

        train(vgg_model, ema_vgg_model, train_prefetcher, criterion, optimizer, epoch, scaler, writer, _rank=rank)

        # Validate on EMA
        acc1 = test(ema_vgg_model, valid_prefetcher, device)
        if rank == 0:
            dprint(f"[Epoch {epoch+1}] Validation Acc@1 (EMA) = {float(acc1):.4f}", this_rank=rank)
            print("\n")

        # Update LR
        scheduler.step()
        if rank == 0:
            dprint(f"[Scheduler] After step lr={current_lr(optimizer)}", this_rank=rank)

        # Automatically save the model with the highest index
        is_best = acc1 > best_acc1
        is_last = (epoch + 1) == train_config.epochs
        best_acc1 = max(acc1, best_acc1)
        if rank == 0:
            dprint(f"[Checkpoint] is_best={is_best} is_last={is_last} best_acc1={best_acc1:.4f}", this_rank=rank)
            #save_checkpoint({"epoch": epoch + 1,
             #                "best_acc1": best_acc1,
             #                "state_dict": vgg_model.state_dict(),
             #                "ema_state_dict": ema_vgg_model.state_dict(),
             #                "optimizer": optimizer.state_dict(),
             #                "scheduler": scheduler.state_dict()},
             #               f"epoch_{epoch + 1}.pth.tar",
             #               samples_dir,
             #               results_dir,
             #               "best.pth.tar",
             #               "last.pth.tar",
             #               is_best,
             #               is_last)
    
    dist.destroy_process_group()
    dprint("[Finish] Destroyed process group.", this_rank=rank)


def load_dataset(
        train_image_dir: str = train_config.train_image_dir,
        valid_image_dir: str = train_config.valid_image_dir,
        resized_image_size=train_config.resized_image_size,
        crop_image_size=train_config.crop_image_size,
        dataset_mean_normalize=train_config.dataset_mean_normalize,
        dataset_std_normalize=train_config.dataset_std_normalize,
        device: torch.device = torch.device("cpu"),
        _rank: int = 0,
) -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_dataset = ImageDataset(train_image_dir,
                                 resized_image_size,
                                 crop_image_size,
                                 dataset_mean_normalize,
                                 dataset_std_normalize,
                                 "Train")
    valid_dataset = ImageDataset(valid_image_dir,
                                 resized_image_size,
                                 crop_image_size,
                                 dataset_mean_normalize,
                                 dataset_std_normalize,
                                 "Valid")

    if _rank == 0:
        dprint(f"[Data] Train dir={train_image_dir} Valid dir={valid_image_dir}", this_rank=_rank)
        dprint(f"[Data] resized={resized_image_size} crop={crop_image_size}", this_rank=_rank)
        dprint(f"[Data] mean={dataset_mean_normalize} std={dataset_std_normalize}", this_rank=_rank)
        try:
            dprint(f"[Data] len(train)={len(train_dataset)} len(valid)={len(valid_dataset)}", this_rank=_rank)
        except Exception as e:
            dprint(f"[Data] Could not get dataset lengths: {e}", this_rank=_rank)

    # Generator all dataloader
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_config.batch_size,
                                  sampler=train_sampler,
                                  num_workers=train_config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=False)

    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=train_config.batch_size,
                                  sampler=valid_sampler,
                                  num_workers=train_config.num_workers,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=False)

    if _rank == 0:
        dprint(f"[Loader] batch_size={train_config.batch_size} num_workers={train_config.num_workers} pin_memory=True", this_rank=_rank)
        # Best-effort sampler sizes
        try:
            dprint(f"[Sampler] train sampler shards roughly len={len(train_sampler)} valid sampler len={len(valid_sampler)}", this_rank=_rank)
        except Exception:
            pass

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, device)
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, device)

    if _rank == 0:
        try:
            dprint(f"[Prefetcher] train batches={len(train_prefetcher)} valid batches={len(valid_prefetcher)}", this_rank=_rank)
        except Exception:
            pass

    return train_prefetcher, valid_prefetcher


def build_model(
        model_arch_name: str = train_config.model_arch_name,
        model_num_classes: int = train_config.model_num_classes,
        model_ema_decay: float = train_config.model_ema_decay,
        device: torch.device = torch.device("cpu"),
        _rank: int = 0,
) -> [nn.Module, nn.Module]:
    vgg_model = model.__dict__[model_arch_name](num_classes=model_num_classes)
    vgg_model = vgg_model.to(device)
    dprint(f"[Model] arch={model_arch_name} num_classes={model_num_classes} device={device}", this_rank=_rank)

    # DDP wrap
    vgg_model = DDP(vgg_model)
    dprint(f"[DDP] Wrapped model in DDP.", this_rank=_rank)

    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
        (1 - model_ema_decay) * averaged_model_parameter + model_ema_decay * model_parameter
    ema_vgg_model = AveragedModel(vgg_model, device=device, avg_fn=ema_avg)
    dprint(f"[EMA] decay={model_ema_decay}", this_rank=_rank)

    # Param count
    if _rank == 0:
        total_params = sum(p.numel() for p in vgg_model.parameters())
        trainable_params = sum(p.numel() for p in vgg_model.parameters() if p.requires_grad)
        dprint(f"[Model] params total={total_params} trainable={trainable_params}", this_rank=_rank)

    return vgg_model, ema_vgg_model


def define_loss(
        loss_label_smoothing: float = train_config.loss_label_smoothing,
        device: torch.device = torch.device("cpu"),
        _rank: int = 0,
) -> nn.CrossEntropyLoss:
    criterion = nn.CrossEntropyLoss(label_smoothing=loss_label_smoothing)
    criterion = criterion.to(device)
    dprint(f"[Loss] CrossEntropy label_smoothing={loss_label_smoothing}", this_rank=_rank)
    return criterion


def define_optimizer(
        model: nn.Module,
        lr: float = train_config.model_lr,
        momentum: float = train_config.model_momentum,
        weight_decay: float = train_config.model_weight_decay,
        _rank: int = 0,
) -> optim.SGD:
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          momentum=momentum,
                          weight_decay=weight_decay)
    dprint(f"[Optim] SGD lr={lr} momentum={momentum} weight_decay={weight_decay}", this_rank=_rank)
    return optimizer


def define_scheduler(
        optimizer: optim.SGD,
        t_0: int = train_config.lr_scheduler_T_0,
        t_mult=train_config.lr_scheduler_T_mult,
        eta_min=train_config.lr_scheduler_eta_min,
        _rank: int = 0,
) -> lr_scheduler.CosineAnnealingWarmRestarts:
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                         t_0,
                                                         t_mult,
                                                         eta_min)
    dprint(f"[Sched] CosineAnnealingWarmRestarts T0={t_0} Tmult={t_mult} eta_min={eta_min}", this_rank=_rank)
    return scheduler


def train(
        model: nn.Module,
        ema_model: nn.Module,
        train_prefetcher: CUDAPrefetcher,
        criterion: nn.CrossEntropyLoss,
        optimizer: optim.SGD,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        _rank: int = 0,
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":6.6f", Summary.NONE)
    acc1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    acc5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(batches,
                             [batch_time, data_time, losses, acc1, acc5],
                             prefix=f"Epoch: [{epoch + 1}]")

    if _rank == 0:
        dprint(f"[Train] epoch={epoch+1} batches={batches}", this_rank=_rank)

    # Put the generative network model in training mode
    model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Transfer in-memory data to CUDA devices to speed up training
        images = batch_data["image"].to(train_config.device, non_blocking=True)
        target = batch_data["target"].to(train_config.device, non_blocking=True)

        if batch_index == 0 and _rank == 0:
            dprint(f"[Batch0] images.shape={tuple(images.shape)} dtype={images.dtype} device={images.device}", this_rank=_rank)
            dprint(f"[Batch0] target.shape={tuple(target.shape)} dtype={target.dtype} device={target.device}", this_rank=_rank)

        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Get batch size
        batch_size = images.size(0)

        # Initialize generator gradients
        model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        if batch_index == 0 and _rank == 0:
            dprint(f"[Batch0] output.shape={tuple(output.shape)}", this_rank=_rank)

        # Quick sanity checks
        if has_any_nan_or_inf(loss.detach()):
            dprint(f"[WARN][Epoch {epoch+1} Batch {batch_index}] Loss has NaN/Inf: {loss.detach().float().item()}", rank="all", this_rank=_rank)
        if has_any_nan_or_inf(output.detach()):
            dprint(f"[WARN][Epoch {epoch+1} Batch {batch_index}] Output has NaN/Inf", rank="all", this_rank=_rank)

        # Backpropagation
        scaler.scale(loss).backward()

        # Optional: total grad norm (infrequent)
        if DEBUG_GRAD_NORM and (batch_index % (DEBUG_PRINT_FREQ * 2) == 0):
            try:
                total_norm = grad_total_norm(model.parameters())
                dprint(f"[Grad] total_norm≈{total_norm:.4f}", this_rank=_rank)
            except Exception as e:
                dprint(f"[Grad] could not compute total norm: {e}", this_rank=_rank)

        # update generator weights
        scaler.step(optimizer)
        scaler.update()

        # Update EMA
        try:
            # track first param delta magnitude (rough proxy for update health)
            with torch.no_grad():
                p_model = next(model.parameters())
                p_ema = next(ema_model.parameters())
                before = p_ema.detach().float().view(-1)[0].item()
            ema_model.update_parameters(model)
            with torch.no_grad():
                after = next(ema_model.parameters()).detach().float().view(-1)[0].item()
            if batch_index % DEBUG_PRINT_FREQ == 0 and _rank == 0:
                dprint(f"[EMA] sample_param change≈{abs(after - before):.3e}", this_rank=_rank)
        except Exception as e:
            dprint(f"[EMA] update error: {e}", this_rank=_rank)

        # measure accuracy and record loss
        top1, top5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), batch_size)
        # top1/top5 may be tensors; convert for safety
        acc1.update(float(top1[0]), batch_size)
        acc5.update(float(top5[0]), batch_size)

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Logging per frequency
        if batch_index % DEBUG_PRINT_FREQ == 0:
            # LR may have multiple param groups
            lrs = current_lr(optimizer)
            if _rank == 0:
                dprint(f"[Train] Epoch {epoch+1} [{batch_index}/{batches}] "
                       f"Loss {losses.val:.4f} (avg {losses.avg:.4f}) "
                       f"Acc@1 {acc1.val:.3f} (avg {acc1.avg:.3f}) "
                       f"Acc@5 {acc5.val:.3f} (avg {acc5.avg:.3f}) "
                       f"LR {lrs} "
                       f"Data {data_time.val:.3f}s Batch {batch_time.val:.3f}s",
                       this_rank=_rank)

        # Write the data during training to the training log file
        if batch_index % train_config.train_print_frequency == 0 and _rank == 0:
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches)
            print(f"Epoch: [{epoch + 1}][{batch_index}/{batches}]\t "
                  f"Loss: {losses.val:.4f} ({losses.avg:.4f})\t "
                  f"Acc@1: {acc1.val:.3f} ({acc1.avg:.3f})\t "
                  f"Acc@5: {acc5.val:.3f} ({acc5.avg:.3f})")
            progress.display(batch_index)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1

    if _rank == 0:
        dprint(f"[Epoch {epoch+1}] DONE | avg_loss={losses.avg:.4f} avg_acc1={acc1.avg:.3f} avg_acc5={acc5.avg:.3f}", this_rank=_rank)


if __name__ == "__main__":
    main(train_config.seed)

