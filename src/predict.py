import argparse
import os
import time
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms, models
from torch import nn


def load_model(ckpt_path='models/best_model.pt', num_classes=6):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    img_size = ckpt.get('img_size', 128)
    classes = ckpt.get('classes', None)

    model = models.mobilenet_v2(weights=None)
    in_feat = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feat, num_classes)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    return model, classes, img_size


def _compute_saliency(model, x_tensor, target_idx):
    # NOTE: this function is no longer used; kept for backward-compatibility
    model.zero_grad()
    logits = model(x_tensor)
    score = logits[0, target_idx]
    score.backward()
    grad = x_tensor.grad.data.abs().mean(dim=1, keepdim=False).squeeze(0)  # HxW
    sal = grad.cpu().numpy()
    sal = (sal - sal.min())
    if sal.max() > 0:
        sal = sal / sal.max()
    return sal


def _find_last_conv_module(model):
    # return the last nn.Conv2d module in the model
    last_conv = None
    for name, m in model.named_modules():
        from torch import nn as _nn

        if isinstance(m, _nn.Conv2d):
            last_conv = m
    return last_conv


def _grad_cam(model, x_tensor, target_idx):
    """Compute Grad-CAM heatmap for `target_idx` class.
    Returns a 2D numpy float array (H, W) normalized to [0,1]."""
    activations = None
    gradients = None

    # find last conv layer
    last_conv = _find_last_conv_module(model)
    if last_conv is None:
        raise RuntimeError('No Conv2d layer found for Grad-CAM')

    # hooks
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()

    fh = last_conv.register_forward_hook(forward_hook)
    bh = last_conv.register_full_backward_hook(backward_hook)

    model.zero_grad()
    logits = model(x_tensor)
    score = logits[0, target_idx]
    score.backward(retain_graph=False)

    # remove hooks
    fh.remove()
    bh.remove()

    if activations is None or gradients is None:
        raise RuntimeError('Failed to collect activations/gradients for Grad-CAM')

    # pooled gradients
    pooled_grads = torch.mean(gradients, dim=(0, 2, 3), keepdim=False)  # C
    # weight activations
    activ = activations.squeeze(0)  # C x H x W
    for i in range(activ.shape[0]):
        activ[i, :, :] *= pooled_grads[i]

    cam = torch.sum(activ, dim=0).cpu().numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def predict_and_save_with_box(image_path, out_root='outputs', save=True, out_scale=1.25):
    """Predict class for `image_path` and save an annotated copy with a bounding box
    drawn around the most salient region. Returns (pred_class, pred_prob, out_path_or_None)."""
    model, classes, img_size = load_model()

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    orig = Image.open(image_path).convert('RGB')
    resized = orig.copy()
    resized = resized.resize((img_size, img_size))
    x = tfm(resized).unsqueeze(0)
    x.requires_grad_(True)

    with torch.enable_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        top_idx = int(torch.argmax(probs).item())
        pred_class = classes[top_idx] if classes else str(top_idx)
        pred_prob = float(probs[top_idx].item())

        # compute Grad-CAM heatmap for the predicted class
        sal = _grad_cam(model, x, top_idx)

    # Resize grad-cam heatmap to original image size
    sal_img = Image.fromarray((sal * 255).astype(np.uint8))
    sal_img = sal_img.resize(orig.size, resample=Image.BILINEAR)
    sal_arr = np.array(sal_img).astype(np.float32) / 255.0

    # threshold and compute bbox
    thr = max(0.15, sal_arr.mean() + 0.25 * sal_arr.std())
    mask = sal_arr >= thr
    coords = np.argwhere(mask)
    out_path = None
    if coords.size > 0 and save:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        # add padding
        h, w = orig.size[1], orig.size[0]
        pad_x = int(0.05 * w)
        pad_y = int(0.05 * h)
        left = max(0, x0 - pad_x)
        top = max(0, y0 - pad_y)
        right = min(w - 1, x1 + pad_x)
        bottom = min(h - 1, y1 + pad_y)

        # scale output image first if requested, then draw scaled box/text
        scale = float(out_scale) if out_scale and out_scale > 0 else 1.0
        if scale != 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            out = orig.resize((new_w, new_h), resample=Image.LANCZOS)
            # scale box coords
            left = int(left * scale)
            top = int(top * scale)
            right = int(right * scale)
            bottom = int(bottom * scale)
            w, h = new_w, new_h
        else:
            out = orig.copy()

        draw = ImageDraw.Draw(out)
        box_width = max(2, int(min(w, h) / 100))
        draw.rectangle([left, top, right, bottom], outline='red', width=box_width)

        # prepare label text
        label = f"{pred_class} {pred_prob:.2%}"
        from PIL import ImageFont
        # choose a font size proportional to image size for readability
        font_size = max(12, int(min(w, h) * 0.06))
        font = None
        # try common TTF fonts; fall back to default bitmap font if not available
        for fname in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
            try:
                font = ImageFont.truetype(fname, font_size)
                break
            except Exception:
                font = None
        if font is None:
            font = ImageFont.load_default()

        # compute text size robustly
        try:
            text_w, text_h = font.getsize(label)
        except Exception:
            try:
                bbox = font.getbbox(label)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except Exception:
                text_w, text_h = (len(label) * (font_size // 2), font_size)

        # place label above the box if space, otherwise inside
        padding = max(4, int(min(w, h) * 0.01))
        text_x0 = left
        text_y0 = max(0, top - text_h - padding)
        text_x1 = text_x0 + text_w + padding * 2
        text_y1 = text_y0 + text_h + padding

        # background rectangle for text (semi-opaque)
        try:
            # create semi-opaque background by composing
            bg = Image.new('RGBA', out.size, (0, 0, 0, 0))
            bg_draw = ImageDraw.Draw(bg)
            bg_draw.rectangle([text_x0, text_y0, text_x1, text_y1], fill=(0, 0, 0, 160))
            out = Image.alpha_composite(out.convert('RGBA'), bg).convert('RGB')
            draw = ImageDraw.Draw(out)
        except Exception:
            # fallback: draw solid rectangle
            draw.rectangle([text_x0, text_y0, text_x1, text_y1], fill='black')

        # draw text
        text_pos = (text_x0 + padding, text_y0 + (padding // 2))
        draw.text(text_pos, label, fill='white', font=font)

        # prepare output folder
        timestr = time.strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join(out_root, timestr)
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.basename(image_path)
        out_path = os.path.join(out_dir, base)
        out.save(out_path)

    return pred_class, pred_prob, out_path


def predict(image_path):
    pred_class, pred_prob, _ = predict_and_save_with_box(image_path, save=False)
    print(f'Prediction: {pred_class} ({pred_prob:.3f})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to an image file')
    parser.add_argument('--out_root', type=str, default='outputs', help='Root folder to save annotated predictions')
    parser.add_argument('--save', action='store_true', help='Save annotated image when predicting')
    parser.add_argument('--scale', type=float, default=1.25, help='Scale factor for saved annotated image (e.g., 1.2)')
    args = parser.parse_args()
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(args.image_path)
    pred_class, pred_prob, out_path = predict_and_save_with_box(args.image_path, out_root=args.out_root, save=args.save, out_scale=args.scale)
    print(f'Prediction: {pred_class} ({pred_prob:.3f})')
    if out_path:
        print(f'Annotated image saved to: {out_path}')
