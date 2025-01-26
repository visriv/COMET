import os
import random
import time
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import wandb
from datetime import datetime
from utils import (
    get_network, get_test_dataloader, get_val_dataloader, init_training_dataloader, 
    WarmUpLR, save_model, load_model, cal_acc, show_image
)
from conf import settings
from models.Saliency_mapper import Posthoc_loss
from train_module import (
    train_env_ours, auto_split, refine_split, update_pre_optimizer, 
    update_pre_optimizer_vit, update_bias_optimizer, auto_cluster
)
from eval_module import (
    eval_training, eval_best, eval_mode, eval_explain, posthoc_eval, eval_CAM, 
    posthoc_explain, B_cos_attribution, RBF_eval, eval_explain_NICO, 
    posthoc_explain_NICO, eval_CAM_NICO, RBF_eval_nico, B_cos_attribution_nico
)
from timm.scheduler import create_scheduler
from CAM import CAM_mapgeneration

def entropy(model_output):
    probabilities = F.softmax(model_output, dim=1)
    return torch.mean(-torch.sum(probabilities * torch.log2(probabilities), dim=1))

def batch_total_variation_loss(masks):
    pixel_dif1 = masks[:, :, 1:, :] - masks[:, :, :-1, :]  # Vertical difference
    pixel_dif2 = masks[:, :, :, 1:] - masks[:, :, :, :-1]  # Horizontal difference
    tv_loss = torch.sum(torch.abs(pixel_dif1), dim=[1, 2, 3]) + torch.sum(torch.abs(pixel_dif2), dim=[1, 2, 3])
    return tv_loss.mean()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def initialize_model(config, variance_opt):
    if variance_opt['mode'] == 'ours':
        if config['net'] == 'vit':
            return get_custom_network_vit(config, variance_opt)
        return get_custom_network(config, variance_opt)
    return get_network(config)

def initialize_optimizers_and_schedulers(model, training_opt):
    optimizers, schedulers, warmup_schedulers = [], [], []
    if "COMET" in training_opt['exp_name']:
        if "3p" in training_opt['exp_name']:
            optimizer1 = optim.AdamW(model.predictor.parameters(), lr=training_opt['lr'], weight_decay=0.03)
            optimizer2 = optim.AdamW(model.completement_pred.parameters(), lr=training_opt['lr'], weight_decay=0.03)
            optimizer3 = optim.AdamW(model.generator.parameters(), lr=training_opt['lr'], weight_decay=0.03)
            optimizers = [optimizer1, optimizer2, optimizer3]
        elif "2p" in training_opt['exp_name']:
            optimizer1 = optim.AdamW(model.predictor.parameters(), lr=training_opt['lr'], weight_decay=0.03)
            optimizer2 = optim.AdamW(model.generator.parameters(), lr=training_opt['lr'], weight_decay=0.03)
            optimizers = [optimizer1, optimizer2]
        else:
            optimizers = [optim.AdamW(model.generator.parameters(), lr=training_opt['lr'], weight_decay=0.03)]
    else:
        optimizers = [optim.AdamW(model.parameters(), lr=training_opt['lr'], weight_decay=training_opt['weight_decay'])]

    for optimizer in optimizers:
        schedulers.append(optim.lr_scheduler.MultiStepLR(optimizer, milestones=training_opt['milestones'], gamma=training_opt['gamma']))
        warmup_schedulers.append(WarmUpLR(optimizer, len(train_loader) * training_opt['warm']))

    return optimizers, schedulers, warmup_schedulers

def train_epoch(epoch, model, train_loader, config, optimizers, schedulers, warmup_schedulers):
    model.train()
    train_correct = 0
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()

        outputs = model(images)

        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(outputs, labels)

        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        _, preds = outputs.max(1)
        train_correct += preds.eq(labels).sum().item()

    train_accuracy = train_correct / len(train_loader.dataset)
    print(f"Epoch {epoch}: Train Accuracy: {train_accuracy:.4f}, Time Taken: {time.time() - start_time:.2f}s")
    return train_accuracy

def evaluate_model(epoch, model, val_loader, config):
    model.eval()
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()

    val_accuracy = val_correct / len(val_loader.dataset)
    print(f"Epoch {epoch}: Validation Accuracy: {val_accuracy:.4f}")
    return val_accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()

    config = load_config(args.cfg)
    set_seed(config['training_opt']['seed'])

    wandb.init(project=config['wandb']['project'], name=config['wandb']['run_name'], config=config)

    model = initialize_model(config, config['variance_opt'])
    if torch.cuda.is_available():
        model = model.cuda()

    train_loader = init_training_dataloader(config, config['training_opt']['mean'], config['training_opt']['std']).get_dataloader(
        config['training_opt']['batch_size'], shuffle=True
    )
    val_loader = get_val_dataloader(config, config['training_opt']['mean'], config['training_opt']['std'], batch_size=config['training_opt']['batch_size'])

    optimizers, schedulers, warmup_schedulers = initialize_optimizers_and_schedulers(model, config['training_opt'])

    best_acc = 0
    for epoch in range(1, config['training_opt']['epochs'] + 1):
        train_accuracy = train_epoch(epoch, model, train_loader, config, optimizers, schedulers, warmup_schedulers)
        val_accuracy = evaluate_model(epoch, model, val_loader, config)

        wandb.log({
            "epoch": epoch,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy
        })

        if val_accuracy > best_acc:
            print("New Best Model Found. Saving...")
            save_model(model, os.path.join(config['training_opt']['checkpoint_path'], f"best_model_epoch_{epoch}.pth"))
            best_acc = val_accuracy

if __name__ == '__main__':
    main()
