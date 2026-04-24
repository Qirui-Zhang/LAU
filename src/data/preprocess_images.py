import os
import argparse
import numpy as np
from PIL import Image, ImageFilter


class ImagePreprocessor:
    def __init__(self, output_size=(128, 128), noise_config=None, blur_config=None):
        """
        初始化图像预处理器

        参数:
            output_size: 元组，输出图像尺寸（默认(128,128)）
            noise_config: 列表，噪声配置字典列表
            blur_config: 字典，模糊配置字典
        """
        self.output_size = output_size
        self.noise_config = noise_config or []
        self.blur_config = blur_config

    def _apply_blur(self, image):
        """应用模糊处理"""
        if self.blur_config['type'] == 'gaussian':
            return image.filter(
                ImageFilter.GaussianBlur(self.blur_config.get('radius', 2))
            )
        return image

    def _apply_noise(self, image):
        """应用噪声处理"""
        img_array = np.array(image).astype(np.float32)  # 转换为浮点型

        for noise in self.noise_config:
            if noise['type'] == 'gaussian':
                # 高斯噪声标准化处理
                noise_var = noise.get('var', 0.1)
                noise_arr = np.random.normal(0, noise_var ** 0.5, img_array.shape)
                img_array = np.clip(img_array + noise_arr * 255, 0, 255)

            elif noise['type'] == 'salt_pepper':
                # 椒盐噪声三维处理
                salt_prob = noise.get('salt_prob', 0.05)
                pepper_prob = noise.get('pepper_prob', 0.05)

                prob_matrix = np.random.random(img_array.shape)
                salt_mask = prob_matrix < salt_prob
                pepper_mask = (prob_matrix >= salt_prob) & \
                              (prob_matrix < (salt_prob + pepper_prob))

                img_array[salt_mask] = 255
                img_array[pepper_mask] = 0

        return Image.fromarray(img_array.astype(np.uint8))

    def process_image(self, image_path):
        """处理单个图像"""
        try:
            img = Image.open(image_path).convert('RGB')
        except IOError:
            print(f"Cannot open image: {image_path}")
            return None

        # 降采样
        img = img.resize(self.output_size, Image.LANCZOS)

        # 应用模糊
        if self.blur_config:
            img = self._apply_blur(img)

        # 应用噪声
        if self.noise_config:
            img = self._apply_noise(img)

        return img

    def process_directory(self, input_dir, output_dir):
        """处理目录中的所有图像"""
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    input_path = os.path.join(root, file)
                    output_relpath = os.path.relpath(root, input_dir)
                    output_path = os.path.join(output_dir, output_relpath, file)

                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    processed_img = self.process_image(input_path)
                    if processed_img:
                        processed_img.save(output_path)
                        print(f"Preprocessed image saved to: {output_path}")

'''
        Test example:
        python preprocess_images.py \
          --input_dir ./input_images \
          --output_dir ./processed \
          --size 128 \
          --noise gaussian salt_pepper \
          --gaussian_var 0.2 \
          --salt_prob 0.03 \
          --pepper_prob 0.03 \
          --blur_type gaussian \
          --blur_radius 3
'''
def main():
    parser = argparse.ArgumentParser(description="Image Preprocessor")
    parser.add_argument("--input_dir", default="./data/raw/THINGS-EEG/THINGS-EEG_images_set", help="Input directory path")
    parser.add_argument("--output_dir", default="./data/processed/image_set", help="Output directory path")
    parser.add_argument("--size", type=int, default=128, help="Output image size (default: 128)")

    # 噪声参数
    parser.add_argument("--noise", nargs="+", choices=["gaussian", "salt_pepper"],
                        help="Choose noise type")
    parser.add_argument("--gaussian_var", type=float, default=0.1,
                        help="Gaussian noise variance (default: 0.1)")
    parser.add_argument("--salt_prob", type=float, default=0.05,
                        help="Salt noise probability (default: 0.05)")
    parser.add_argument("--pepper_prob", type=float, default=0.05,
                        help="Pepper noise probability (default: 0.05)")

    # 模糊参数
    parser.add_argument("--blur_type", choices=["gaussian"],
                        help="Choose blur type")
    parser.add_argument("--blur_radius", type=int, default=2,
                        help="Blur radius (default: 2)")

    args = parser.parse_args()

    # 构建噪声配置
    noise_config = []
    if args.noise:
        for noise_type in args.noise:
            if noise_type == "gaussian":
                noise_config.append({
                    "type": "gaussian",
                    "var": args.gaussian_var
                })
            elif noise_type == "salt_pepper":
                noise_config.append({
                    "type": "salt_pepper",
                    "salt_prob": args.salt_prob,
                    "pepper_prob": args.pepper_prob
                })

    # 构建模糊配置
    blur_config = None
    if args.blur_type:
        blur_config = {
            "type": args.blur_type,
            "radius": args.blur_radius
        }

    processor = ImagePreprocessor(
        output_size=(args.size, args.size),
        noise_config=noise_config,
        blur_config=blur_config
    )

    processor.process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()