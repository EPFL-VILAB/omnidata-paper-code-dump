import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms
from .class_dict import class_dict


def predict_imagenet_video(video_in_path, video_out_path, model, device='cuda', normalize=True, image_size=224, batch_size=16):

    mean = torch.Tensor([0.485, 0.456, 0.406]) if normalize else torch.Tensor([0,0,0])
    std = torch.Tensor([0.229, 0.224, 0.225]) if normalize else torch.Tensor([1,1,1])

    image_transforms = transforms.Compose([
        transforms.Resize(image_size, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    

    # 1. Loop over video to make batched predictions

    cap = cv2.VideoCapture(video_in_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    batch = []
    preds = []

    for frame_idx in range(num_frames):
        ret, frame = cap.read()
        img = Image.fromarray(frame)
        batch.append(image_transforms(img))
        
        if len(batch) >= batch_size or frame_idx >= num_frames-1:
            batch = torch.stack(batch)
            with torch.no_grad():
                model_pred = model(batch.to(device))
            preds.append(model_pred.detach().cpu())
            batch = []

    preds = torch.cat(preds, dim=0)
    preds = torch.softmax(preds, dim=1)
    top5_prob, top5_classes = torch.topk(preds, 5, dim=1, sorted=True)

    cap.release()


    # 2. Write predictions into new video

    cap = cv2.VideoCapture(video_in_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    dir_name = os.path.dirname(video_out_path)
    os.makedirs(dir_name, exist_ok=True)
    out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    for frame_idx in range(num_frames):
        ret, frame = cap.read()
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        
        for rank, (class_prob, class_idx) in enumerate(zip(top5_prob[frame_idx], top5_classes[frame_idx])):
            class_name = class_dict[class_idx.item()].split(',')[0]
            text = f'{rank+1}. ({100*class_prob:05.2f}%) {class_name}'
            draw.text([15, 15 + rank * 10], text, fill='black')
            
        img = np.array(img)
        out.write(img)
        
    cap.release()
    out.release()

    print(f'Saved annotated video under: {video_out_path}')
