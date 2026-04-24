import clip
clip.available_models()
model, preprocess = clip.load('RN50', device='cuda:5')

import open_clip
vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k', precision='fp16', device = 'cuda:5')