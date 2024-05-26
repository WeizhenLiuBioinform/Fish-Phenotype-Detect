
import math
import sys
import time

import torch
import tempfile
import transforms
import train_utils.distributed_utils as utils
from .coco_eval import EvalCOCOMetric
from .loss import KpLoss
from .loss import KpLoss_One
from .loss import KpLoss_val

# Validation function for one epoch
def val_one_epoch(model, data_loader, device, print_freq=10, scaler=None):
    model.eval()  # Set model to evaluation mode
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    mse = KpLoss_val()

    # Initialize accumulative values for three types of loss
    total_original_loss = 0
    total_additional_loss = 0
    total_combined_loss = 0
    num_samples = 0

    with torch.no_grad():  # No gradient calculation during evaluation
        for images, targets, scale, angle, src_center in metric_logger.log_every(data_loader, print_freq, header):
            images = torch.stack([image.to(device) for image in images])
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                results = model(images)
                original_loss, additional_loss, combined_loss = mse(results, targets)
                # original_loss, additional_loss, combined_loss = mse(results, targets, scale, angle, src_center)

            # Accumulate three types of loss values
            total_original_loss += original_loss
            total_additional_loss += additional_loss
            total_combined_loss += combined_loss
            num_samples += 1

            # Update MetricLogger
            metric_logger.update(loss=combined_loss)

    # Calculate average loss
    avg_original_loss = total_original_loss / num_samples
    avg_additional_loss = total_additional_loss / num_samples
    avg_combined_loss = total_combined_loss / num_samples

    # Return average values for three types of loss
    return avg_original_loss, avg_additional_loss, avg_combined_loss

# Training function for one epoch
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50, warmup=False, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # Enable warmup training during the first epoch
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mse = KpLoss()
    mse = mse.to(device)
    print("Device of weight_main_loss:", mse.weight_main_loss.device)
    print("Device of weight_additional_loss:", mse.weight_additional_loss.device)
    print("Is weight_main_loss a leaf tensor:", mse.weight_main_loss.is_leaf)
    print("Is weight_additional_loss a leaf tensor:", mse.weight_additional_loss.is_leaf)

    mse_one = KpLoss_One()
    mloss = torch.zeros(1).to(device)  # Mean losses
    gradnorm_optimizer = torch.optim.Adam([mse.weight_main_loss, mse.weight_additional_loss], lr=1e-3)

    for i, [images, targets, scale, angle, src_center] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = torch.stack([image.to(device) for image in images])

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            results = model(images)
            if epoch < 20:
                losses = mse_one(results, targets)
            else:
                losses = mse(results, targets)
                # losses = mse(results, targets, scale, angle, src_center)

        loss_dict_reduced = utils.reduce_dict({"losses": losses})
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        mloss = (mloss * i + loss_value) / (i + 1)  # Update mean losses

        if not math.isfinite(loss_value):  # Stop training if loss is infinite
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        gradnorm_optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        gradnorm_optimizer.step()

        if lr_scheduler is not None:  # Use warmup training in the first epoch
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    return mloss, now_lr

# Evaluation function
@torch.no_grad()
def evaluate(model, data_loader, device, flip=False, flip_pairs=None):
    if flip:
        assert flip_pairs is not None, "enable flip must provide flip_pairs."

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tf:
        results_file_name = tf.name
    key_metric = EvalCOCOMetric(data_loader.dataset.coco, "keypoints", results_file_name)
    for image, targets, scale, angle, src_center in metric_logger.log_every(data_loader, 100, header):
        images = torch.stack([img.to(device) for img in image])

        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(images)
        if flip:
            flipped_images = transforms.flip_images(images)
            flipped_outputs = model(flipped_images)
            flipped_outputs = transforms.flip_back(flipped_outputs, flip_pairs)
            flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
            outputs = (outputs + flipped_outputs) * 0.5

        model_time = time.time() - model_time

        reverse_trans = [t["reverse_trans"] for t in targets]
        outputs = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)

        key_metric.update(targets, outputs)
        metric_logger.update(model_time=model_time)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    key_metric.synchronize_results()

    if utils.is_main_process():
        coco_info = key_metric.evaluate()
    else:
        coco_info = None

    return coco_info
