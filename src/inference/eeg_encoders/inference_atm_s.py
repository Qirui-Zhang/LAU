import os
import json
import argparse
from datetime import datetime
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import re

from src.models.eeg_encoders.ATM_S.atm_s_encoder import ATMSEncoder
from src.data.eeg_dataset import EEGDataset


def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)["inference"]

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None

def save_image(tensor, save_path):
    img = tensor.squeeze(0).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255).astype(np.uint8)
    # img = img.astype(np.uint8)
    Image.fromarray(img).save(save_path)


def infer(config_path):
    config = load_config(config_path)
    device = torch.device(config["device"])

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config["output"]["root_dir"], f"inference_{timestamp}_{config['model_checkpoint'].split('/')[-1].split('.')[0]}")
    os.makedirs(output_dir, exist_ok=True)

    # 初始化受试者id
    subjects = config["subjects"]
    subject_ids = [extract_id_from_string(s) for s in subjects]
    subject_ids = torch.tensor(subject_ids, device=device)

    # Load model
    model = ATMSEncoder().to(device)
    checkpoint_path = config["model_checkpoint"]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Load dataset
    test_dataset = EEGDataset(
        data_path=config["data_path"],
        subjects=subjects,
        train=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4
    )

    # Inference
    with torch.no_grad():
        for batch_idx, (eeg_data, label, img) in enumerate(test_loader):
            eeg_data = eeg_data.to(device)

            # print("label", label)

            # 获取当前批次的subject_ids
            batch_size = eeg_data.size(0)
            subject_ids_batch = torch.full((batch_size,), subject_ids[0], dtype=torch.long).to(
                device)  # TODO:暂且选择第一个subject，后面改成每个subject都训练一次

            reconstructed_img = model(eeg_data, subject_ids=subject_ids_batch)

            for i in range(reconstructed_img.size(0)):
                save_path = os.path.join(
                    output_dir,
                    config["output"]["naming_convention"].format(
                        subject=config["subjects"][0],
                        label=label[i].item(), # TODO: Fix this
                    )
                )
                save_image(reconstructed_img[i], save_path)
    print(f"LR results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default=f"configs/eeg_encoders/atm_s_config.json", help="配置文件路径")
    args = parser.parse_args()
    infer(args.config)