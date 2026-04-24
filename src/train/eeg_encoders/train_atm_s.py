import os
import json
import torch
import argparse
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import AdamW
import re
import tqdm
from src.models.eeg_encoders.ATM_S.atm_s_encoder import ATMSEncoder
from src.data.eeg_dataset import EEGDataset
from src.models.losses import MultiObjectiveLoss, LowLevelLoss
from torch import nn
from torch.amp import GradScaler, autocast


def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None

def train(config_path):
    # 加载配置
    config = load_config(config_path)["training"]
    exp_cfg = config["experiment"]

    # 设备设置
    device = config["device"]

    # 设置随机种子
    seed = exp_cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 初始化受试者id
    subjects = config["subjects"]
    subject_ids = [extract_id_from_string(s) for s in subjects]
    subject_ids = torch.tensor(subject_ids, device=device)

    # 初始化数据集
    dataset = EEGDataset(
        data_path=config["data_path"],
        subjects=subjects,
        train=True
    )

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    # 初始化模型
    model = ATMSEncoder().to(device)
    criterion = LowLevelLoss(
        alpha=config["loss_weights"]["alpha"],
        beta=config["loss_weights"]["beta"],
        device=device
    )

    # 优化器设置
    optimizer = AdamW(
        model.parameters(),
        lr=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"]
    )

    # 初始化GradScaler
    scaler = GradScaler(device)

    # 训练准备
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    epochs = config["epochs"]
    save_dir = exp_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    log_dir = exp_cfg["log_dir"]
    log = ""

    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_smooth_l1_loss = 0.0
        train_ssim_loss = 0.0

        from PIL import Image
        from torchvision import transforms

        # Define a transform to convert the image to a tensor
        transform = transforms.ToTensor()

        for batch in tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config['epochs']}"):
            eeg_data, _, img = batch
            eeg_data = eeg_data.to(device, non_blocking=True)

            # Ensure img is a tensor
            if isinstance(img, list):
                img = [transform(Image.open(i).convert("RGB")) for i in img]
                img = torch.stack(img)

            target_img = img.to(device, non_blocking=True)

            # 获取当前批次的subject_ids
            batch_size = eeg_data.size(0)
            subject_ids_batch = torch.full((batch_size,), subject_ids[0], dtype=torch.long).to(device) # TODO:暂且选择第一个subject，后面改成每个subject都训练一次

            # Forward pass with autocast
            with autocast(device):
                reconstructed_img = model(eeg_data, subject_ids=subject_ids_batch)
                (total_loss, smooth_l1_loss,
                 ssim_loss) = criterion(reconstructed_img, target_img)

            # Backward pass with GradScaler
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()
            train_smooth_l1_loss += smooth_l1_loss.item()
            train_ssim_loss += ssim_loss.item()

        # 计算平均损失
        avg_train_loss = train_loss / len(dataloader)
        avg_train_smooth_l1_loss = train_smooth_l1_loss / len(dataloader)
        avg_train_ssim_loss = train_ssim_loss / len(dataloader)

        epoch_loss_info = f"Epoch {epoch + 1}/{config['epochs']} | " \
                f"Total Train Loss: {avg_train_loss:.6f}| " \
                f"Smooth L1 Loss: {avg_train_smooth_l1_loss:.6f} | " \
                f"SSIM Loss: {avg_train_ssim_loss:.6f} "
        print(epoch_loss_info)
        log += epoch_loss_info + "\n"

        if (epoch + 1) % exp_cfg["save_interval"] == 0:
            ckpt_name = exp_cfg["checkpoint_format"].format(subject=subjects[0], epochs=epoch+1)
            saving_path = os.path.join(save_dir, ckpt_name)
            torch.save(model.state_dict(), saving_path)
            print(f"Model saved at {saving_path}")
            log += f"Model saved at {saving_path}\n"

    # 保存日志
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, f"train_{timestamp}_{subjects[0]}_{epochs}_log.txt")
    with open(log_path, "w") as f:
        f.write(log)
    print(f"Log saved at {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=f"configs/eeg_encoders/atm_s_config.json", help="配置文件路径")
    args = parser.parse_args()
    train(args.config)