import os
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
from datetime import datetime


def extract_features(dataset_type, dataset_root, device, model, preprocess, batch_size):
    """提取指定数据集的特征"""
    features = []
    img_dir = os.path.join(dataset_root, f"{dataset_type}_images")

    # 获取排序后的类别目录
    classes = sorted([d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))])

    for class_dir in tqdm(classes, desc=f"Processing {dataset_type}"):
        class_path = os.path.join(img_dir, class_dir)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # 批量处理
        for i in range(0, len(images), batch_size):
            batch_images = []
            for img_name in images[i:i + batch_size]:
                try:
                    img_path = os.path.join(class_path, img_name)
                    image = Image.open(img_path).convert("RGB")
                    batch_images.append(preprocess(image))
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue

            if batch_images:
                # 转换为tensor并推理
                with torch.no_grad():
                    image_input = torch.stack(batch_images).to(device)
                    image_features = model.encode_image(image_input)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    features.append(image_features.cpu())

    # 合并所有特征
    all_features = torch.cat(features, dim=0) if features else torch.Tensor()
    return all_features

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_root", type=str, default="./data/processed/image_set_512_gaussian_blur_4/")
    args.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args.add_argument("--model_name", type=str, default="ViT-H-14")
    args.add_argument("--pretrained", type=str, default="laion2b_s32b_b79k")
    args.add_argument("--batch_size", type=int, default=32)
    args.add_argument("--output_dir", type=str, default="./data/processed/image_set_512_gaussian_blur_4/")
    args = args.parse_args()

    # 配置参数
    device = args.device
    model_name = args.model_name
    pretrained = args.pretrained
    batch_size = args.batch_size
    dataset_root = args.dataset_root
    output_dir = args.output_dir
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # 加载模型和预处理
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device
    )
    model.eval()

    output_dir_actual = os.path.join(output_dir, f"{model_name}_features_{timestamp}")
    os.makedirs(output_dir_actual, exist_ok=True)

    # 提取并保存训练集特征
    train_features = extract_features("training", dataset_root, device, model, preprocess, batch_size)
    output_path_train = os.path.join(output_dir_actual, f"{model_name}_features_train.pt")
    torch.save(train_features, output_path_train)
    print(f"Saved training features with shape: {train_features.shape}")

    # 提取并保存测试集特征
    test_features = extract_features("test", dataset_root, device, model, preprocess, batch_size)
    output_path_test = os.path.join(output_dir_actual, f"{model_name}_features_test.pt")
    torch.save(test_features, output_path_test)
    print(f"Saved test features with shape: {test_features.shape}")

if __name__ == "__main__":
    main()