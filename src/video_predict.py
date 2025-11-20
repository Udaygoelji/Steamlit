import os
import time
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
from src.predict import load_model, _grad_cam


def process_video(video_path, out_root='outputs', scale=1.0, frame_step=1, codec='mp4v'):
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    model, classes, img_size = load_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('Failed to open video: ' + video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w = int(width * scale)
    out_h = int(height * scale)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    timestr = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(out_root, timestr)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{base}_annotated.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    frame_idx = 0
    processed = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % frame_step != 0:
                # write scaled original frame to keep timing
                if scale != 1.0:
                    frame_out = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
                else:
                    frame_out = frame
                writer.write(frame_out)
                continue

            # convert to PIL RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)
            resized = pil.resize((img_size, img_size))
            x = tfm(resized).unsqueeze(0).to(device)
            x.requires_grad_(True)

            with torch.enable_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                top_idx = int(torch.argmax(probs).item())
                pred_class = classes[top_idx] if classes else str(top_idx)
                pred_prob = float(probs[top_idx].item())
                cam = _grad_cam(model, x, top_idx)

            # resize cam to frame size
            cam_img = Image.fromarray((cam * 255).astype(np.uint8))
            cam_img = cam_img.resize(pil.size, resample=Image.BILINEAR)
            cam_arr = np.array(cam_img).astype(np.float32) / 255.0
            thr = max(0.15, cam_arr.mean() + 0.25 * cam_arr.std())
            mask = cam_arr >= thr
            coords = np.argwhere(mask)

            out_pil = pil.copy()
            draw = ImageDraw.Draw(out_pil)
            if coords.size > 0:
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0)
                pad_x = int(0.05 * pil.size[0])
                pad_y = int(0.05 * pil.size[1])
                left = max(0, x0 - pad_x)
                top = max(0, y0 - pad_y)
                right = min(pil.size[0] - 1, x1 + pad_x)
                bottom = min(pil.size[1] - 1, y1 + pad_y)

                # scale canvas if requested
                if scale != 1.0:
                    out_pil = out_pil.resize((out_w, out_h), resample=Image.LANCZOS)
                    left = int(left * scale)
                    top = int(top * scale)
                    right = int(right * scale)
                    bottom = int(bottom * scale)

                draw = ImageDraw.Draw(out_pil)
                box_width = max(2, int(min(out_pil.size[0], out_pil.size[1]) / 100))
                draw.rectangle([left, top, right, bottom], outline='red', width=box_width)

                # label
                label = f"{pred_class} {pred_prob:.2%}"
                font_size = max(12, int(min(out_pil.size[0], out_pil.size[1]) * 0.05))
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                except Exception:
                    font = ImageFont.load_default()
                try:
                    text_w, text_h = font.getsize(label)
                except Exception:
                    bbox = font.getbbox(label)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]

                padding = max(4, int(min(out_pil.size[0], out_pil.size[1]) * 0.01))
                text_x0 = left
                text_y0 = max(0, top - text_h - padding)
                text_x1 = text_x0 + text_w + padding * 2
                text_y1 = text_y0 + text_h + padding
                try:
                    bg = Image.new('RGBA', out_pil.size, (0, 0, 0, 0))
                    bg_draw = ImageDraw.Draw(bg)
                    bg_draw.rectangle([text_x0, text_y0, text_x1, text_y1], fill=(0, 0, 0, 160))
                    out_pil = Image.alpha_composite(out_pil.convert('RGBA'), bg).convert('RGB')
                    draw = ImageDraw.Draw(out_pil)
                except Exception:
                    draw.rectangle([text_x0, text_y0, text_x1, text_y1], fill='black')
                draw.text((text_x0 + padding, text_y0 + (padding // 2)), label, fill='white', font=font)
            else:
                if scale != 1.0:
                    out_pil = out_pil.resize((out_w, out_h), resample=Image.LANCZOS)

            out_frame = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
            writer.write(out_frame)
            processed += 1
    finally:
        cap.release()
        writer.release()

    return out_path, processed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', required=True, help='Input video file')
    parser.add_argument('--out_root', default='outputs', help='Output root folder')
    parser.add_argument('--scale', type=float, default=1.0, help='Output scale factor')
    parser.add_argument('--frame_step', type=int, default=1, help='Process every N-th frame')
    args = parser.parse_args()

    out_path, processed = process_video(args.video_path, out_root=args.out_root, scale=args.scale, frame_step=args.frame_step)
    print(f'Processed frames: {processed}. Saved annotated video to: {out_path}')
