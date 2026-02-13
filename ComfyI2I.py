import logging
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torchvision.ops import masks_to_boxes

LOGGER = logging.getLogger("ComfyI2I")

VERY_BIG_SIZE = 1024 * 1024
MAX_RESOLUTION = 8192
SAFE_MAX_PIXELS = MAX_RESOLUTION * MAX_RESOLUTION

_RESIZE_INTERP_TORCH = {
    "nearest": "nearest",
    "bilinear": "bilinear",
    "bicubic": "bicubic",
    "area": "area",
}

_RESIZE_INTERP_CV2 = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


def _ensure_safe_size(width: int, height: int, context: str) -> Tuple[int, int]:
    w = max(1, min(int(width), MAX_RESOLUTION))
    h = max(1, min(int(height), MAX_RESOLUTION))
    if w * h > SAFE_MAX_PIXELS:
        raise ValueError(f"{context}: requested size {w}x{h} exceeds safe pixel budget.")
    return w, h


def tensor2rgb(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 2:
        return t.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)

    if t.ndim == 3:
        if t.shape[-1] in (1, 3, 4):
            if t.shape[-1] == 1:
                return t.unsqueeze(0).repeat(1, 1, 1, 3)
            if t.shape[-1] == 4:
                return t.unsqueeze(0)[..., :3]
            return t.unsqueeze(0)
        return t.unsqueeze(-1).repeat(1, 1, 1, 3)

    if t.shape[-1] == 1:
        return t.repeat(1, 1, 1, 3)
    if t.shape[-1] >= 3:
        return t[..., :3]
    return t


def tensor2rgba(t: torch.Tensor) -> torch.Tensor:
    rgb = tensor2rgb(t)
    alpha = torch.ones((*rgb.shape[:3], 1), device=rgb.device, dtype=rgb.dtype)
    if t.ndim >= 4 and t.shape[-1] == 4:
        alpha = t[..., 3:4]
    return torch.cat((rgb, alpha), dim=-1)


def tensor2mask(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 2:
        return t.unsqueeze(0)

    if t.ndim == 3:
        if t.shape[-1] in (1, 3, 4):
            rgb = t[..., :3] if t.shape[-1] > 1 else t[..., :1].repeat(1, 1, 3)
            gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).unsqueeze(0)
            return gray
        return t

    if t.shape[-1] == 4 and torch.min(t[..., 3]).item() < 0.9999:
        return t[..., 3]

    rgb = tensor2rgb(t)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def mask_to_image(mask: torch.Tensor) -> torch.Tensor:
    m = tensor2mask(mask)
    return m.unsqueeze(-1).repeat(1, 1, 1, 3)


def _repeat_to_batch(t: torch.Tensor, batch: int) -> torch.Tensor:
    if t.shape[0] == batch:
        return t
    if t.shape[0] == 1:
        return t.repeat(batch, *([1] * (t.ndim - 1)))
    if batch % t.shape[0] == 0:
        return t.repeat(batch // t.shape[0], *([1] * (t.ndim - 1)))
    return t[:batch]


def _sanitize_mapping(mapping: Optional[torch.Tensor], out_batch: int, src_batch: int, device: torch.device) -> torch.Tensor:
    if mapping is None:
        return torch.arange(out_batch, device=device, dtype=torch.long) % max(src_batch, 1)

    mapping = mapping.to(device=device)
    if mapping.ndim > 1:
        mapping = mapping.reshape(-1)
    if mapping.numel() == 0:
        return torch.arange(out_batch, device=device, dtype=torch.long) % max(src_batch, 1)

    mapping = mapping.long()
    if mapping.shape[0] < out_batch:
        reps = math.ceil(out_batch / mapping.shape[0])
        mapping = mapping.repeat(reps)[:out_batch]
    else:
        mapping = mapping[:out_batch]

    return mapping.clamp(0, max(src_batch - 1, 0))


def _resize_image_cv2(image: torch.Tensor, width: int, height: int, method: str) -> torch.Tensor:
    interp = _RESIZE_INTERP_CV2.get(method, cv2.INTER_CUBIC)
    out = []
    for i in range(image.shape[0]):
        arr = image[i].detach().cpu().numpy()
        resized = cv2.resize(arr, (width, height), interpolation=interp)
        out.append(torch.from_numpy(resized))
    return torch.stack(out, dim=0).to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)


def _resize_mask_cv2(mask: torch.Tensor, width: int, height: int, method: str) -> torch.Tensor:
    interp = _RESIZE_INTERP_CV2.get(method, cv2.INTER_LINEAR)
    m = tensor2mask(mask)
    out = []
    for i in range(m.shape[0]):
        arr = m[i].detach().cpu().numpy()
        resized = cv2.resize(arr, (width, height), interpolation=interp)
        out.append(torch.from_numpy(resized))
    return torch.stack(out, dim=0).to(device=m.device, dtype=m.dtype).clamp(0.0, 1.0)


def _resize_image(image: torch.Tensor, width: int, height: int, method: str = "bicubic") -> torch.Tensor:
    width, height = _ensure_safe_size(width, height, "image resize")
    if method == "lanczos":
        return _resize_image_cv2(image, width, height, method)

    mode = _RESIZE_INTERP_TORCH.get(method, "bicubic")
    nchw = image.permute(0, 3, 1, 2)
    if mode in ("nearest", "area"):
        out = F.interpolate(nchw, size=(height, width), mode=mode)
    else:
        out = F.interpolate(nchw, size=(height, width), mode=mode, align_corners=False)
    return out.permute(0, 2, 3, 1).clamp(0.0, 1.0)


def _resize_mask(mask: torch.Tensor, width: int, height: int, method: str = "bilinear") -> torch.Tensor:
    width, height = _ensure_safe_size(width, height, "mask resize")
    if method == "lanczos":
        return _resize_mask_cv2(mask, width, height, method)

    mode = _RESIZE_INTERP_TORCH.get(method, "bilinear")
    m = tensor2mask(mask).unsqueeze(1)
    if mode in ("nearest", "area"):
        out = F.interpolate(m, size=(height, width), mode=mode)
    else:
        out = F.interpolate(m, size=(height, width), mode=mode, align_corners=False)
    return out[:, 0].clamp(0.0, 1.0)


def _mask_boxes(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    m = tensor2mask(mask).clone()
    if m.ndim != 3:
        raise ValueError(f"Expected mask batch [B,H,W], got {tuple(m.shape)}")

    b, h, w = m.shape
    if h <= 0 or w <= 0:
        boxes = torch.zeros((b, 4), device=m.device, dtype=torch.float32)
        empty = torch.ones((b,), device=m.device, dtype=torch.bool)
        return boxes, empty

    flat = m.reshape(b, -1)
    empty = flat.max(dim=1).values <= 0.0
    m[empty, 0, 0] = 1.0
    boxes = masks_to_boxes(m)
    m[empty, 0, 0] = 0.0
    return boxes, empty


def _dilate(mask: torch.Tensor, steps: int) -> torch.Tensor:
    out = mask.unsqueeze(1)
    for _ in range(max(steps, 0)):
        out = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)
    return out[:, 0]


def _erode(mask: torch.Tensor, steps: int) -> torch.Tensor:
    return 1.0 - _dilate(1.0 - mask, steps)


def _gaussian_blur_mask(mask: torch.Tensor, radius: float) -> torch.Tensor:
    if radius <= 0.0:
        return mask

    sigma = max(radius / 3.0, 1e-3)
    ksize = max(3, int(radius * 2) * 2 + 1)
    coords = torch.arange(ksize, device=mask.device, dtype=mask.dtype) - (ksize - 1) / 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, ksize, ksize)

    src = mask.unsqueeze(1)
    pad = ksize // 2
    if src.shape[-1] > 1 and src.shape[-2] > 1:
        src = F.pad(src, (pad, pad, pad, pad), mode="reflect")
    else:
        src = F.pad(src, (pad, pad, pad, pad), mode="replicate")
    out = F.conv2d(src, kernel_2d)
    return out[:, 0].clamp(0.0, 1.0)


def _apply_levels(mask: torch.Tensor, black_level: float, mid_level: float, white_level: float) -> torch.Tensor:
    black = max(0.0, min(1.0, black_level / 255.0))
    white = max(black + 1e-6, min(1.0, white_level / 255.0))
    mid = max(1e-6, min(1.0 - 1e-6, mid_level / 255.0))

    normalized = ((mask - black) / (white - black)).clamp(0.0, 1.0)
    gamma = math.log(0.5) / math.log(mid)
    return normalized.pow(gamma).clamp(0.0, 1.0)


def _split_connected_components(
    mask: torch.Tensor,
    min_area: int = 0,
    max_components: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    m = tensor2mask(mask)
    components: List[Tuple[int, int, torch.Tensor]] = []

    for b in range(m.shape[0]):
        sample_np = m[b].detach().cpu().numpy()
        labels, count = ndimage.label(sample_np > 0.0)
        if count == 0:
            continue

        for idx in range(1, count + 1):
            area = labels == idx
            px = int(np.count_nonzero(area))
            if px <= 0 or px < int(min_area):
                continue
            comp = np.zeros_like(sample_np, dtype=np.float32)
            comp[area] = sample_np[area]
            components.append((px, b, torch.from_numpy(comp)))

    if not components:
        return m, torch.arange(m.shape[0], device=m.device, dtype=torch.long)

    components.sort(key=lambda x: x[0], reverse=True)
    if max_components > 0:
        components = components[: max_components]

    stack = torch.stack([c[2] for c in components], dim=0).to(device=m.device, dtype=m.dtype)
    mapping = torch.tensor([c[1] for c in components], device=m.device, dtype=torch.long)
    return stack, mapping


def _simple_color_transfer_lab(source_bgr: np.ndarray, target_bgr: np.ndarray, mask: Optional[np.ndarray], strength: float) -> np.ndarray:
    src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    sel = None
    if mask is not None and np.count_nonzero(mask > 0.01) > 16:
        sel = mask > 0.01

    adjusted = src_lab.copy()
    for c in range(3):
        src_chan = src_lab[..., c]
        tgt_chan = tgt_lab[..., c]

        if sel is None:
            src_mean, src_std = float(src_chan.mean()), float(src_chan.std())
            tgt_mean, tgt_std = float(tgt_chan.mean()), float(tgt_chan.std())
        else:
            src_mean = float(src_chan[sel].mean()) if np.any(sel) else float(src_chan.mean())
            src_std = float(src_chan[sel].std()) if np.any(sel) else float(src_chan.std())
            tgt_mean = float(tgt_chan[sel].mean()) if np.any(sel) else float(tgt_chan.mean())
            tgt_std = float(tgt_chan[sel].std()) if np.any(sel) else float(tgt_chan.std())

        src_std = max(src_std, 1e-3)
        tgt_std = max(tgt_std, 1e-3)

        chan = (src_chan - src_mean) * (tgt_std / src_std) + tgt_mean
        adjusted[..., c] = src_chan * (1.0 - strength) + chan * strength

    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return cv2.cvtColor(adjusted, cv2.COLOR_LAB2BGR)


def _simple_color_transfer_rgb(source_bgr: np.ndarray, target_bgr: np.ndarray, strength: float) -> np.ndarray:
    src = source_bgr.astype(np.float32)
    tgt = target_bgr.astype(np.float32)

    src_mean = src.mean(axis=(0, 1))
    src_std = np.maximum(src.std(axis=(0, 1)), 1e-3)
    tgt_mean = tgt.mean(axis=(0, 1))
    tgt_std = np.maximum(tgt.std(axis=(0, 1)), 1e-3)

    normalized = (src - src_mean) / src_std
    matched = normalized * tgt_std + tgt_mean
    blended = src * (1.0 - strength) + matched * strength
    return np.clip(blended, 0, 255).astype(np.uint8)


def _apply_tone_controls(image_bgr: np.ndarray, gamma: float, contrast: float, brightness: float) -> np.ndarray:
    x = np.arange(256, dtype=np.float32)
    inv_gamma = 1.0 / max(gamma, 1e-6)
    lut = np.clip(((x / 255.0) ** inv_gamma) * 255.0, 0, 255).astype(np.uint8)

    corrected = cv2.LUT(image_bgr, lut).astype(np.float32)
    corrected = corrected * contrast + brightness
    return np.clip(corrected, 0, 255).astype(np.uint8)


def _kmeans_quantize_lab(image_bgr: np.ndarray, colors: int) -> np.ndarray:
    if colors <= 1:
        return image_bgr

    cv2.setRNGSeed(0)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    flat = lab.reshape(-1, 3).astype(np.float32)
    sample_size = min(flat.shape[0], 30000)
    step = max(1, flat.shape[0] // sample_size)
    sel = flat[::step]

    _, _, centers = cv2.kmeans(
        sel,
        colors,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2),
        2,
        cv2.KMEANS_PP_CENTERS,
    )
    centers = centers.astype(np.float32)

    dists = np.linalg.norm(flat[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    quant = centers[labels].reshape(lab.shape).astype(np.uint8)
    return cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)


def _combine_tensors(image1: torch.Tensor, image2: torch.Tensor, op: str, clamp_result: str, round_result: str) -> torch.Tensor:
    a = image1
    b = image2

    if b.ndim == 3 and a.ndim == 4:
        b = b.unsqueeze(-1)
    if a.ndim == 3 and b.ndim == 4:
        a = a.unsqueeze(-1)

    if a.shape[0] != b.shape[0]:
        batch = max(a.shape[0], b.shape[0])
        a = _repeat_to_batch(a, batch)
        b = _repeat_to_batch(b, batch)

    if op == "union (max)":
        result = torch.max(a, b)
    elif op == "intersection (min)":
        result = torch.min(a, b)
    elif op == "difference":
        result = a - b
    elif op == "multiply":
        result = a * b
    elif op == "multiply_alpha":
        rgba = tensor2rgba(a)
        m = tensor2mask(b)
        alpha = (rgba[..., 3] * m).unsqueeze(-1)
        result = torch.cat((rgba[..., :3], alpha), dim=-1)
    elif op == "add":
        result = a + b
    elif op == "greater_or_equal":
        result = torch.where(a >= b, torch.ones_like(a), torch.zeros_like(a))
    elif op == "greater":
        result = torch.where(a > b, torch.ones_like(a), torch.zeros_like(a))
    else:
        result = a

    if clamp_result == "yes":
        result = result.clamp(0.0, 1.0)
    if round_result == "yes":
        result = torch.round(result)

    return result


def _apply_color_correction_tensor(target_image: torch.Tensor, source_image: torch.Tensor, factor: float = 1.0) -> torch.Tensor:
    target = tensor2rgb(target_image)
    source = tensor2rgb(source_image)
    batch = max(target.shape[0], source.shape[0])
    target = _repeat_to_batch(target, batch)
    source = _repeat_to_batch(source, batch)

    result = []
    for i in range(batch):
        src = source[i].detach().cpu().numpy().astype(np.float32)
        tgt = target[i].detach().cpu().numpy().astype(np.float32)

        src_mean = src.mean(axis=(0, 1))
        src_std = np.maximum(src.std(axis=(0, 1)), 1e-6)
        tgt_mean = tgt.mean(axis=(0, 1))
        tgt_std = np.maximum(tgt.std(axis=(0, 1)), 1e-6)

        corrected = ((tgt - tgt_mean) * (src_std / tgt_std) + src_mean)
        blended = tgt * (1.0 - factor) + corrected * factor
        result.append(torch.from_numpy(np.clip(blended, 0.0, 1.0)).to(device=target.device, dtype=target.dtype))

    return torch.stack(result, dim=0)


def _unsharp_tensor(image: torch.Tensor, amount: float) -> torch.Tensor:
    if amount <= 0:
        return image

    out = []
    for i in range(image.shape[0]):
        arr = (image[i].detach().cpu().numpy() * 255.0).astype(np.float32)
        blur = cv2.GaussianBlur(arr, (0, 0), sigmaX=1.2)
        sharp = arr * (1.0 + amount) - blur * amount
        out.append(torch.from_numpy(np.clip(sharp / 255.0, 0.0, 1.0)))

    return torch.stack(out, dim=0).to(device=image.device, dtype=image.dtype)


def _detail_boost_from_reference(patch: torch.Tensor, reference: torch.Tensor, strength: float) -> torch.Tensor:
    if strength <= 0:
        return patch

    out = []
    patch_b = max(patch.shape[0], reference.shape[0])
    p = _repeat_to_batch(patch, patch_b)
    r = _repeat_to_batch(reference, patch_b)

    for i in range(patch_b):
        pp = (p[i].detach().cpu().numpy() * 255.0).astype(np.float32)
        rr = (r[i].detach().cpu().numpy() * 255.0).astype(np.float32)
        rr_blur = cv2.GaussianBlur(rr, (0, 0), sigmaX=1.2)
        high = rr - rr_blur
        merged = pp + high * strength
        out.append(torch.from_numpy(np.clip(merged / 255.0, 0.0, 1.0)))

    return torch.stack(out, dim=0).to(device=patch.device, dtype=patch.dtype)


def _anchor_offsets(extra_w: int, extra_h: int, anchor: str) -> Tuple[int, int]:
    if anchor == "top_left":
        return 0, 0
    if anchor == "top_right":
        return max(extra_w, 0), 0
    if anchor == "bottom_left":
        return 0, max(extra_h, 0)
    if anchor == "bottom_right":
        return max(extra_w, 0), max(extra_h, 0)
    return max(extra_w, 0) // 2, max(extra_h, 0) // 2


def _fit_to_size(
    data: torch.Tensor,
    target_w: int,
    target_h: int,
    fit_mode: str,
    anchor: str,
    resize_method: str,
    is_mask: bool,
    pad_rgb: Tuple[float, float, float],
) -> torch.Tensor:
    target_w, target_h = _ensure_safe_size(target_w, target_h, "fit")
    if is_mask:
        src = tensor2mask(data)
        src = src.unsqueeze(-1)
    else:
        src = tensor2rgb(data)

    b, h, w, c = src.shape

    if fit_mode == "stretch":
        if is_mask:
            resized = _resize_mask(src[..., 0], target_w, target_h, resize_method)
            return resized
        return _resize_image(src, target_w, target_h, resize_method)

    scale = min(target_w / max(w, 1), target_h / max(h, 1)) if fit_mode == "contain" else max(
        target_w / max(w, 1), target_h / max(h, 1)
    )
    new_w, new_h = _ensure_safe_size(int(round(w * scale)), int(round(h * scale)), "fit resize")

    if is_mask:
        resized = _resize_mask(src[..., 0], new_w, new_h, resize_method)
        resized = resized.unsqueeze(-1)
    else:
        resized = _resize_image(src, new_w, new_h, resize_method)

    if fit_mode == "contain":
        if is_mask:
            canvas = torch.zeros((b, target_h, target_w, 1), device=src.device, dtype=src.dtype)
        else:
            canvas = torch.zeros((b, target_h, target_w, 3), device=src.device, dtype=src.dtype)
            canvas[..., 0] = pad_rgb[0]
            canvas[..., 1] = pad_rgb[1]
            canvas[..., 2] = pad_rgb[2]

        extra_w = target_w - new_w
        extra_h = target_h - new_h
        ox, oy = _anchor_offsets(extra_w, extra_h, anchor)
        canvas[:, oy : oy + new_h, ox : ox + new_w, :] = resized
        if is_mask:
            return canvas[..., 0]
        return canvas

    # cover
    extra_w = new_w - target_w
    extra_h = new_h - target_h
    ox, oy = _anchor_offsets(extra_w, extra_h, anchor)
    cropped = resized[:, oy : oy + target_h, ox : ox + target_w, :]
    if is_mask:
        return cropped[..., 0]
    return cropped


class Mask_Ops:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "separate_mask": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
                "component_min_area": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION * MAX_RESOLUTION, "step": 1}),
                "max_components": ("INT", {"default": 128, "min": 1, "max": 2048, "step": 1}),
                "blend_percentage": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "black_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 255.0, "step": 0.1}),
                "mid_level": ("FLOAT", {"default": 127.5, "min": 0.0, "max": 255.0, "step": 0.1}),
                "white_level": ("FLOAT", {"default": 255.0, "min": 0.0, "max": 255.0, "step": 0.1}),
                "channel": (["red", "green", "blue"],),
                "shrink_grow": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                "invert": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
                "blur_radius": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 128.0, "step": 0.1}),
                "mask_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    CATEGORY = "I2I"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK_MAPPING")
    RETURN_NAMES = ("mask_image", "mask", "mask mapping")
    FUNCTION = "Mask_Ops"

    def Mask_Ops(
        self,
        image,
        separate_mask,
        component_min_area,
        max_components,
        blend_percentage,
        black_level,
        mid_level,
        white_level,
        channel,
        shrink_grow,
        invert=0,
        blur_radius=2.0,
        mask_threshold=0.0,
        mask=None,
    ):
        image_rgb = tensor2rgb(image).float().clamp(0.0, 1.0)
        b, h, w, _ = image_rgb.shape

        if mask is not None:
            base_mask = tensor2mask(mask).float().clamp(0.0, 1.0)
        else:
            base_mask = torch.ones((b, h, w), device=image_rgb.device, dtype=image_rgb.dtype)

        base_mask = _repeat_to_batch(base_mask, b)
        if base_mask.shape[1] != h or base_mask.shape[2] != w:
            base_mask = _resize_mask(base_mask, w, h, method="bilinear")

        if mask_threshold > 0:
            base_mask = torch.where(base_mask >= mask_threshold, base_mask, torch.zeros_like(base_mask))

        chan_idx = {"red": 0, "green": 1, "blue": 2}[channel]
        channel_mask = image_rgb[..., chan_idx]
        merged = (1.0 - blend_percentage) * base_mask + blend_percentage * channel_mask

        merged = _apply_levels(merged, black_level, mid_level, white_level)

        if shrink_grow < 0:
            merged = _erode(merged, abs(int(shrink_grow)))
        elif shrink_grow > 0:
            merged = _dilate(merged, int(shrink_grow))

        if blur_radius > 0:
            merged = _gaussian_blur_mask(merged, blur_radius)

        if int(round(invert)) == 1:
            merged = 1.0 - merged

        merged = merged.clamp(0.0, 1.0)

        if int(round(separate_mask)) == 1:
            separated, mapping = _split_connected_components(
                merged,
                min_area=int(component_min_area),
                max_components=int(max_components),
            )
        else:
            separated = merged
            mapping = torch.arange(separated.shape[0], device=separated.device, dtype=torch.long)

        return (mask_to_image(separated), separated, mapping)


class Color_Correction:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "match_mode": (["lab_reinhard", "rgb_stats"], {"default": "lab_reinhard"}),
                "no_of_colors": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                "blur_radius": ("INT", {"default": 2, "min": 0, "max": 100, "step": 1}),
                "blur_amount": ("INT", {"default": 2, "min": 0, "max": 100, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -255.0, "max": 255.0, "step": 1.0}),
                "preserve_luminance": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    CATEGORY = "I2I"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "ColorXfer2"

    def ColorXfer2(
        self,
        source_image,
        target_image,
        match_mode,
        no_of_colors,
        blur_radius,
        blur_amount,
        strength,
        gamma,
        contrast,
        brightness,
        preserve_luminance,
        mask=None,
    ):
        src = tensor2rgb(source_image).float().clamp(0.0, 1.0)
        tgt = tensor2rgb(target_image).float().clamp(0.0, 1.0)

        batch = max(src.shape[0], tgt.shape[0])
        src = _repeat_to_batch(src, batch)
        tgt = _repeat_to_batch(tgt, batch)

        mask_tensor = None
        if mask is not None:
            mask_tensor = tensor2mask(mask).float().clamp(0.0, 1.0)
            mask_tensor = _repeat_to_batch(mask_tensor, batch)
            if mask_tensor.shape[1] != src.shape[1] or mask_tensor.shape[2] != src.shape[2]:
                mask_tensor = _resize_mask(mask_tensor, src.shape[2], src.shape[1], method="bilinear")

        out = []
        for i in range(batch):
            src_bgr = cv2.cvtColor((src[i].detach().cpu().numpy() * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
            tgt_bgr = cv2.cvtColor((tgt[i].detach().cpu().numpy() * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

            m = None
            if mask_tensor is not None:
                m = mask_tensor[i].detach().cpu().numpy().astype(np.float32)
                if blur_radius > 0:
                    k = max(1, int(blur_radius) * 2 + 1)
                    m = cv2.GaussianBlur(m, (k, k), max(blur_amount / 3.0, 1e-3))
                    m = np.clip(m, 0.0, 1.0)

            if match_mode == "rgb_stats":
                transferred = _simple_color_transfer_rgb(src_bgr, tgt_bgr, min(max(strength, 0.0), 2.0))
            else:
                transferred = _simple_color_transfer_lab(src_bgr, tgt_bgr, m, min(max(strength, 0.0), 2.0))

            if no_of_colors > 1:
                transferred = _kmeans_quantize_lab(transferred, int(no_of_colors))

            if int(round(preserve_luminance)) == 1:
                src_yuv = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2YCrCb)
                out_yuv = cv2.cvtColor(transferred, cv2.COLOR_BGR2YCrCb)
                out_yuv[:, :, 0] = src_yuv[:, :, 0]
                transferred = cv2.cvtColor(out_yuv, cv2.COLOR_YCrCb2BGR)

            toned = _apply_tone_controls(transferred, gamma, contrast, brightness)
            rgb = cv2.cvtColor(toned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            if m is not None:
                m3 = np.repeat(m[:, :, None], 3, axis=2)
                src_rgb = src[i].detach().cpu().numpy()
                rgb = src_rgb * (1.0 - m3) + rgb * m3

            out.append(torch.from_numpy(np.clip(rgb, 0.0, 1.0)).to(device=src.device, dtype=src.dtype))

        return (torch.stack(out, dim=0),)


class MaskToRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
                "force_resize_width": ("INT", {"default": 1024, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "force_resize_height": ("INT", {"default": 1024, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "kind": (["mask", "RGB", "RGBA"],),
                "padding": ("INT", {"default": 8, "min": 0, "max": 2048, "step": 1}),
                "constraints": (["keep_ratio", "keep_ratio_divisible", "multiple_of", "ignore"],),
                "constraint_x": ("INT", {"default": 64, "min": 2, "max": 4096, "step": 1}),
                "constraint_y": ("INT", {"default": 64, "min": 2, "max": 4096, "step": 1}),
                "min_width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "min_height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "batch_behavior": (["match_ratio", "match_size"],),
                "resize_method": (["nearest", "bilinear", "bicubic", "area", "lanczos"], {"default": "lanczos"}),
                "empty_mask_behavior": (["full_image", "skip"], {"default": "full_image"}),
                "max_output_regions": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
            },
            "optional": {
                "mask_mapping_optional": ("MASK_MAPPING",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "MASK_MAPPING")
    RETURN_NAMES = ("cut image", "cut mask", "region", "cut mask (MASK)", "crop mapping")
    FUNCTION = "get_region"
    CATEGORY = "I2I"

    def get_region(
        self,
        image,
        mask,
        force_resize_width,
        force_resize_height,
        kind,
        padding,
        constraints,
        constraint_x,
        constraint_y,
        min_width,
        min_height,
        batch_behavior,
        resize_method="lanczos",
        empty_mask_behavior="full_image",
        max_output_regions=512,
        mask_mapping_optional=None,
    ):
        image_rgb = tensor2rgb(image).float().clamp(0.0, 1.0)
        mask_tensor = tensor2mask(mask).float().clamp(0.0, 1.0)

        b_img, h, w, _ = image_rgb.shape
        mask_tensor = _repeat_to_batch(mask_tensor, max(mask_tensor.shape[0], 1))
        if mask_tensor.shape[1] != h or mask_tensor.shape[2] != w:
            mask_tensor = _resize_mask(mask_tensor, w, h, method="bilinear")

        b_mask = mask_tensor.shape[0]
        mapping = _sanitize_mapping(mask_mapping_optional, b_mask, b_img, image_rgb.device)

        boxes, empty = _mask_boxes(mask_tensor)

        indices: List[int] = []
        for i in range(b_mask):
            if empty[i] and empty_mask_behavior == "skip":
                continue
            indices.append(i)

        if len(indices) == 0:
            zero_tile = torch.zeros((1, max(1, force_resize_height), max(1, force_resize_width), 3), device=image_rgb.device, dtype=image_rgb.dtype)
            zero_mask = torch.zeros((1, max(1, force_resize_height), max(1, force_resize_width)), device=image_rgb.device, dtype=image_rgb.dtype)
            zero_region = torch.zeros((1, h, w), device=image_rgb.device, dtype=image_rgb.dtype)
            return (
                zero_tile,
                mask_to_image(zero_mask),
                mask_to_image(zero_region),
                zero_mask,
                torch.zeros((1,), device=image_rgb.device, dtype=torch.long),
            )

        indices = indices[: int(max_output_regions)]

        widths = []
        heights = []
        crop_boxes = []
        selected_mapping = []
        selected_empty = []

        for idx in indices:
            selected_mapping.append(int(mapping[idx].item()))
            selected_empty.append(bool(empty[idx].item()))

            if empty[idx]:
                crop_boxes.append((0, 0, w - 1, h - 1))
                tw = force_resize_width if force_resize_width > 0 else w
                th = force_resize_height if force_resize_height > 0 else h
                tw, th = _ensure_safe_size(tw, th, "empty crop")
                widths.append(tw)
                heights.append(th)
                continue

            box = boxes[idx]
            x1 = max(0, int(math.floor(float(box[0]))) - int(padding))
            y1 = max(0, int(math.floor(float(box[1]))) - int(padding))
            x2 = min(w - 1, int(math.ceil(float(box[2]))) + int(padding))
            y2 = min(h - 1, int(math.ceil(float(box[3]))) + int(padding))

            bw = max(1, x2 - x1 + 1)
            bh = max(1, y2 - y1 + 1)

            tw = max(bw, int(min_width))
            th = max(bh, int(min_height))

            if force_resize_width > 0:
                tw = int(force_resize_width)
            if force_resize_height > 0:
                th = int(force_resize_height)

            if constraints == "multiple_of":
                if constraint_x > 1 and tw % constraint_x:
                    tw = ((tw + constraint_x - 1) // constraint_x) * constraint_x
                if constraint_y > 1 and th % constraint_y:
                    th = ((th + constraint_y - 1) // constraint_y) * constraint_y
            elif constraints == "keep_ratio":
                if constraint_x > 0 and constraint_y > 0:
                    ratio = constraint_x / constraint_y
                    if tw / max(th, 1) > ratio:
                        th = max(1, int(round(tw / ratio)))
                    else:
                        tw = max(1, int(round(th * ratio)))
            elif constraints == "keep_ratio_divisible":
                if constraint_x > 0 and constraint_y > 0:
                    ratio = constraint_x / constraint_y
                    if tw / max(th, 1) > ratio:
                        th = max(constraint_y, int(round(tw / ratio)))
                    else:
                        tw = max(constraint_x, int(round(th * ratio)))
                    if tw % constraint_x:
                        tw = ((tw + constraint_x - 1) // constraint_x) * constraint_x
                    if th % constraint_y:
                        th = ((th + constraint_y - 1) // constraint_y) * constraint_y

            tw, th = _ensure_safe_size(tw, th, "crop target")

            crop_boxes.append((x1, y1, x2, y2))
            widths.append(tw)
            heights.append(th)

        widths_t = torch.tensor(widths, device=image_rgb.device)
        heights_t = torch.tensor(heights, device=image_rgb.device)

        if batch_behavior == "match_size":
            widths_t[:] = torch.max(widths_t)
            heights_t[:] = torch.max(heights_t)
        elif batch_behavior == "match_ratio":
            ratios = torch.abs(widths_t.float() / torch.clamp(heights_t.float(), min=1.0) - 1.0)
            for i, is_empty in enumerate(selected_empty):
                if is_empty:
                    ratios[i] = 9999.0
            idx = torch.argmin(ratios).item() if torch.isfinite(ratios).any() else 0
            ref_ratio = widths_t[idx].float() / max(float(heights_t[idx]), 1.0)
            heights_t = torch.max(heights_t, torch.round(widths_t.float() / max(ref_ratio, 1e-6)).long())
            widths_t = torch.max(widths_t, torch.round(heights_t.float() * ref_ratio).long())

        crop_images = []
        crop_masks = []
        region = torch.zeros((len(indices), h, w), device=image_rgb.device, dtype=image_rgb.dtype)

        for i in range(len(indices)):
            x1, y1, x2, y2 = crop_boxes[i]
            src_idx = selected_mapping[i]

            if not selected_empty[i]:
                region[i, y1 : y2 + 1, x1 : x2 + 1] = 1.0

            img_crop = image_rgb[src_idx : src_idx + 1, y1 : y2 + 1, x1 : x2 + 1, :]
            mask_crop = mask_tensor[indices[i] : indices[i] + 1, y1 : y2 + 1, x1 : x2 + 1]

            tw = int(widths_t[i].item())
            th = int(heights_t[i].item())
            tw, th = _ensure_safe_size(tw, th, "crop final")

            img_crop = _resize_image(img_crop, tw, th, resize_method)
            mask_crop = _resize_mask(mask_crop, tw, th, method="bilinear")

            crop_images.append(img_crop[0])
            crop_masks.append(mask_crop[0])

        cut_image = torch.stack(crop_images, dim=0)
        cut_mask = torch.stack(crop_masks, dim=0).clamp(0.0, 1.0)

        if kind == "mask":
            cut_image = cut_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        elif kind == "RGBA":
            cut_image = tensor2rgba(cut_image)
        else:
            cut_image = tensor2rgb(cut_image)

        out_mapping = torch.tensor(selected_mapping, device=image_rgb.device, dtype=torch.long)

        return (cut_image, mask_to_image(cut_mask), mask_to_image(region), cut_mask, out_mapping)


class Combine_And_Paste_Op:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decoded_vae": ("IMAGE",),
                "Original_Image": ("IMAGE",),
                "Cut_Image": ("IMAGE",),
                "Cut_Mask": ("IMAGE",),
                "region": ("IMAGE",),
                "color_xfer_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "op": (["union (max)", "intersection (min)", "difference", "multiply", "multiply_alpha", "add", "greater_or_equal", "greater"],),
                "clamp_result": (["yes", "no"],),
                "round_result": (["no", "yes"],),
                "resize_behavior": (["resize", "keep_ratio_fill", "keep_ratio_fit", "source_size", "source_size_unmasked"],),
                "patch_resize_method": (["nearest", "bilinear", "bicubic", "area", "lanczos"], {"default": "lanczos"}),
                "mask_resize_method": (["nearest", "bilinear", "bicubic", "area", "lanczos"], {"default": "bilinear"}),
                "mask_feather": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 128.0, "step": 0.1}),
                "edge_fix_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detail_boost": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "post_sharpen": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_regions": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
            },
            "optional": {
                "mask_mapping_optional": ("MASK_MAPPING",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("FinalOut",)
    FUNCTION = "com_paste_op"
    CATEGORY = "I2I"

    def com_paste_op(
        self,
        decoded_vae,
        Original_Image,
        Cut_Image,
        Cut_Mask,
        region,
        color_xfer_factor,
        op,
        clamp_result,
        round_result,
        resize_behavior,
        patch_resize_method="lanczos",
        mask_resize_method="bilinear",
        mask_feather=2.0,
        edge_fix_strength=0.3,
        detail_boost=0.0,
        post_sharpen=0.0,
        max_regions=512,
        mask_mapping_optional=None,
    ):
        decoded = tensor2rgb(decoded_vae).float().clamp(0.0, 1.0)
        original = tensor2rgb(Original_Image).float().clamp(0.0, 1.0)
        cut = tensor2rgb(Cut_Image).float().clamp(0.0, 1.0)
        cut_mask = tensor2mask(Cut_Mask).float().clamp(0.0, 1.0)
        region_mask = tensor2mask(region).float().clamp(0.0, 1.0)

        b_region, h, w = region_mask.shape
        b_base = original.shape[0]

        mapping = _sanitize_mapping(mask_mapping_optional, b_region, b_base, original.device)
        boxes, empty = _mask_boxes(region_mask)

        result = original.clone()

        process_count = min(int(max_regions), b_region)

        for i in range(process_count):
            if empty[i]:
                continue

            base_idx = int(mapping[i].item())
            dec_idx = i % decoded.shape[0]
            cut_idx = i % cut.shape[0]
            mask_idx = i % cut_mask.shape[0]

            box = boxes[i]
            bx1, by1, bx2, by2 = (
                int(torch.floor(box[0]).item()),
                int(torch.floor(box[1]).item()),
                int(torch.ceil(box[2]).item()),
                int(torch.ceil(box[3]).item()),
            )
            bw = max(1, bx2 - bx1 + 1)
            bh = max(1, by2 - by1 + 1)

            patch = decoded[dec_idx : dec_idx + 1]
            patch_mask = cut_mask[mask_idx : mask_idx + 1]
            source_h, source_w = patch.shape[1], patch.shape[2]

            target_w, target_h = bw, bh
            if resize_behavior in ("source_size", "source_size_unmasked"):
                target_w, target_h = source_w, source_h
            elif resize_behavior in ("keep_ratio_fill", "keep_ratio_fit"):
                target_ratio = bw / max(float(bh), 1.0)
                src_ratio = source_w / max(float(source_h), 1.0)
                if resize_behavior == "keep_ratio_fill":
                    if src_ratio > target_ratio:
                        target_h = bh
                        target_w = max(1, int(round(target_h * src_ratio)))
                    else:
                        target_w = bw
                        target_h = max(1, int(round(target_w / src_ratio)))
                else:
                    if src_ratio > target_ratio:
                        target_w = bw
                        target_h = max(1, int(round(target_w / src_ratio)))
                    else:
                        target_h = bh
                        target_w = max(1, int(round(target_h * src_ratio)))

            target_w, target_h = _ensure_safe_size(target_w, target_h, "paste target")

            patch = _resize_image(patch, target_w, target_h, patch_resize_method)
            patch_mask = _resize_mask(patch_mask, target_w, target_h, method=mask_resize_method)

            combined = _combine_tensors(patch, patch_mask.unsqueeze(-1), op, clamp_result, round_result)
            if combined.ndim == 3:
                combined = combined.unsqueeze(0)
            combined = tensor2rgb(combined)

            ref_crop = _resize_image(cut[cut_idx : cut_idx + 1], target_w, target_h, patch_resize_method)
            combined = _apply_color_correction_tensor(combined, ref_crop, color_xfer_factor)
            combined = _detail_boost_from_reference(combined, ref_crop, detail_boost)
            combined = _unsharp_tensor(combined, post_sharpen)

            cx = bx1 + bw // 2
            cy = by1 + bh // 2
            x1 = cx - target_w // 2
            y1 = cy - target_h // 2
            x2 = x1 + target_w
            y2 = y1 + target_h

            src_x1, src_y1 = 0, 0
            src_x2, src_y2 = target_w, target_h

            if x1 < 0:
                src_x1 = -x1
                x1 = 0
            if y1 < 0:
                src_y1 = -y1
                y1 = 0
            if x2 > w:
                src_x2 -= x2 - w
                x2 = w
            if y2 > h:
                src_y2 -= y2 - h
                y2 = h

            if x1 >= x2 or y1 >= y2:
                continue

            patch_slice = combined[0, src_y1:src_y2, src_x1:src_x2, :]
            alpha = patch_mask[0, src_y1:src_y2, src_x1:src_x2]

            if resize_behavior != "source_size_unmasked":
                region_slice = region_mask[i, y1:y2, x1:x2]
                alpha = alpha * region_slice

            if mask_feather > 0:
                alpha = _gaussian_blur_mask(alpha.unsqueeze(0), mask_feather)[0]

            if edge_fix_strength > 0:
                edge_smooth = _gaussian_blur_mask(alpha.unsqueeze(0), max(0.5, mask_feather * 1.5 + 0.5))[0]
                alpha = alpha * (1.0 - edge_fix_strength) + edge_smooth * edge_fix_strength

            alpha3 = alpha.unsqueeze(-1).clamp(0.0, 1.0)
            base_area = result[base_idx, y1:y2, x1:x2, :]
            result[base_idx, y1:y2, x1:x2, :] = patch_slice * alpha3 + base_area * (1.0 - alpha3)

        return (result.clamp(0.0, 1.0),)


class I2IAutoAlignImageMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "target_mode": (["use_image_size", "use_mask_size", "custom"], {"default": "use_image_size"}),
                "custom_width": ("INT", {"default": 1024, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "custom_height": ("INT", {"default": 1024, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "fit_mode": (["stretch", "contain", "cover"], {"default": "contain"}),
                "anchor": (["center", "top_left", "top_right", "bottom_left", "bottom_right"], {"default": "center"}),
                "auto_multiple_of": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                "image_resize_method": (["nearest", "bilinear", "bicubic", "area", "lanczos"], {"default": "lanczos"}),
                "mask_resize_method": (["nearest", "bilinear", "bicubic", "area", "lanczos"], {"default": "bilinear"}),
                "pad_r": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_g": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("aligned_image", "aligned_mask", "aligned_region")
    FUNCTION = "align"
    CATEGORY = "I2I/Advanced"

    def align(
        self,
        image,
        mask,
        target_mode,
        custom_width,
        custom_height,
        fit_mode,
        anchor,
        auto_multiple_of,
        image_resize_method,
        mask_resize_method,
        pad_r,
        pad_g,
        pad_b,
    ):
        img = tensor2rgb(image).float().clamp(0.0, 1.0)
        m = tensor2mask(mask).float().clamp(0.0, 1.0)

        img_h, img_w = img.shape[1], img.shape[2]
        mask_h, mask_w = m.shape[1], m.shape[2]

        if target_mode == "use_mask_size":
            target_w, target_h = mask_w, mask_h
        elif target_mode == "custom":
            target_w, target_h = int(custom_width), int(custom_height)
        else:
            target_w, target_h = img_w, img_h

        mul = max(1, int(auto_multiple_of))
        if target_w % mul:
            target_w = ((target_w + mul - 1) // mul) * mul
        if target_h % mul:
            target_h = ((target_h + mul - 1) // mul) * mul

        target_w, target_h = _ensure_safe_size(target_w, target_h, "align target")

        batch = max(img.shape[0], m.shape[0])
        img = _repeat_to_batch(img, batch)
        m = _repeat_to_batch(m, batch)

        aligned_img = _fit_to_size(
            img,
            target_w,
            target_h,
            fit_mode,
            anchor,
            image_resize_method,
            is_mask=False,
            pad_rgb=(float(pad_r), float(pad_g), float(pad_b)),
        )

        aligned_mask = _fit_to_size(
            m,
            target_w,
            target_h,
            fit_mode,
            anchor,
            mask_resize_method,
            is_mask=True,
            pad_rgb=(0.0, 0.0, 0.0),
        )

        region = mask_to_image((aligned_mask > 0.0).float())
        return (aligned_img, aligned_mask, region)


class I2IMaskRefinerPro:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "grow": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                "blur_radius": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 128.0, "step": 0.1}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05}),
                "invert": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "preview")
    FUNCTION = "refine"
    CATEGORY = "I2I/Advanced"

    def refine(self, mask, threshold, grow, blur_radius, gamma, invert):
        m = tensor2mask(mask).float().clamp(0.0, 1.0)
        m = torch.where(m >= threshold, m, torch.zeros_like(m))

        if grow > 0:
            m = _dilate(m, int(grow))
        elif grow < 0:
            m = _erode(m, abs(int(grow)))

        if blur_radius > 0:
            m = _gaussian_blur_mask(m, blur_radius)

        m = m.clamp(0.0, 1.0).pow(1.0 / max(gamma, 1e-6))
        if int(round(invert)) == 1:
            m = 1.0 - m

        return (m, mask_to_image(m))


class I2IMaskedTileExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "tile_size": ("INT", {"default": 768, "min": 128, "max": MAX_RESOLUTION, "step": 8}),
                "stride": ("INT", {"default": 384, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "padding": ("INT", {"default": 32, "min": 0, "max": 2048, "step": 1}),
                "min_coverage": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tiles": ("INT", {"default": 256, "min": 1, "max": 4096, "step": 1}),
                "resize_method": (["nearest", "bilinear", "bicubic", "area", "lanczos"], {"default": "lanczos"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK_MAPPING", "IMAGE")
    RETURN_NAMES = ("tiles", "tile_masks", "tile_mapping", "tile_regions")
    FUNCTION = "extract"
    CATEGORY = "I2I/Advanced"

    def extract(self, image, mask, tile_size, stride, padding, min_coverage, max_tiles, resize_method="lanczos"):
        tile_size, _ = _ensure_safe_size(tile_size, tile_size, "tile size")

        img = tensor2rgb(image).float().clamp(0.0, 1.0)
        m = tensor2mask(mask).float().clamp(0.0, 1.0)

        if m.shape[1] != img.shape[1] or m.shape[2] != img.shape[2]:
            m = _resize_mask(m, img.shape[2], img.shape[1], method="bilinear")

        m = _repeat_to_batch(m, img.shape[0])

        boxes, empty = _mask_boxes(m)

        tiles = []
        tile_masks = []
        mapping = []
        regions = []

        h, w = img.shape[1], img.shape[2]

        for b in range(img.shape[0]):
            if empty[b]:
                continue

            box = boxes[b]
            x1 = max(0, int(math.floor(float(box[0]))) - int(padding))
            y1 = max(0, int(math.floor(float(box[1]))) - int(padding))
            x2 = min(w - 1, int(math.ceil(float(box[2]))) + int(padding))
            y2 = min(h - 1, int(math.ceil(float(box[3]))) + int(padding))

            ys = list(range(y1, y2 + 1, stride)) or [y1]
            xs = list(range(x1, x2 + 1, stride)) or [x1]

            for yy in ys:
                for xx in xs:
                    if len(tiles) >= int(max_tiles):
                        break

                    sx1 = max(0, min(xx - tile_size // 2, w - tile_size))
                    sy1 = max(0, min(yy - tile_size // 2, h - tile_size))
                    sx2 = min(w, sx1 + tile_size)
                    sy2 = min(h, sy1 + tile_size)

                    mask_crop = m[b, sy1:sy2, sx1:sx2]
                    coverage = float((mask_crop > 0.01).float().mean().item())
                    if coverage < min_coverage:
                        continue

                    img_crop = img[b : b + 1, sy1:sy2, sx1:sx2, :]
                    mask_crop_b = mask_crop.unsqueeze(0)

                    img_resized = _resize_image(img_crop, tile_size, tile_size, resize_method)[0]
                    mask_resized = _resize_mask(mask_crop_b, tile_size, tile_size, method="bilinear")[0]

                    region = torch.zeros((h, w), device=img.device, dtype=img.dtype)
                    region[sy1:sy2, sx1:sx2] = 1.0

                    tiles.append(img_resized)
                    tile_masks.append(mask_resized)
                    mapping.append(b)
                    regions.append(region)

                if len(tiles) >= int(max_tiles):
                    break

        if not tiles:
            zero_tile = torch.zeros((tile_size, tile_size, 3), device=img.device, dtype=img.dtype)
            zero_mask = torch.zeros((tile_size, tile_size), device=img.device, dtype=img.dtype)
            zero_region = torch.zeros((img.shape[1], img.shape[2]), device=img.device, dtype=img.dtype)
            return (
                zero_tile.unsqueeze(0),
                zero_mask.unsqueeze(0),
                torch.zeros((1,), device=img.device, dtype=torch.long),
                mask_to_image(zero_region.unsqueeze(0)),
            )

        return (
            torch.stack(tiles, dim=0),
            torch.stack(tile_masks, dim=0),
            torch.tensor(mapping, device=img.device, dtype=torch.long),
            mask_to_image(torch.stack(regions, dim=0)),
        )


class I2IRegionOverlayDebug:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "alpha": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "show_boxes": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1}),
                "line_thickness": ("INT", {"default": 2, "min": 1, "max": 32, "step": 1}),
                "color_r": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "color_g": ("INT", {"default": 128, "min": 0, "max": 255, "step": 1}),
                "color_b": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("overlay",)
    FUNCTION = "overlay"
    CATEGORY = "I2I/Advanced"

    def overlay(self, image, mask, alpha, show_boxes, line_thickness, color_r, color_g, color_b):
        img = tensor2rgb(image).float().clamp(0.0, 1.0)
        m = tensor2mask(mask).float().clamp(0.0, 1.0)
        m = _repeat_to_batch(m, img.shape[0])
        if m.shape[1] != img.shape[1] or m.shape[2] != img.shape[2]:
            m = _resize_mask(m, img.shape[2], img.shape[1], method="bilinear")

        boxes, empty = _mask_boxes(m)
        color = np.array([color_r, color_g, color_b], dtype=np.float32)

        out = []
        for i in range(img.shape[0]):
            arr = (img[i].detach().cpu().numpy() * 255.0).astype(np.float32)
            mm = m[i].detach().cpu().numpy().astype(np.float32)
            mm3 = np.repeat(mm[:, :, None], 3, axis=2)

            blended = arr * (1.0 - alpha * mm3) + color[None, None, :] * (alpha * mm3)

            if int(show_boxes) == 1 and not empty[i]:
                x1, y1, x2, y2 = boxes[i].detach().cpu().numpy().astype(np.int32)
                cv2.rectangle(
                    blended,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (int(color_r), int(color_g), int(color_b)),
                    thickness=int(line_thickness),
                )

            out.append(torch.from_numpy(np.clip(blended / 255.0, 0.0, 1.0)).to(device=img.device, dtype=img.dtype))

        return (torch.stack(out, dim=0),)


class I2ISeamlessPatchPaste:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "patch_image": ("IMAGE",),
                "patch_mask": ("MASK",),
                "x": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "y": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "resize_method": (["nearest", "bilinear", "bicubic", "area", "lanczos"], {"default": "lanczos"}),
                "feather": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 128.0, "step": 0.1}),
                "post_sharpen": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "paste"
    CATEGORY = "I2I/Advanced"

    def paste(self, base_image, patch_image, patch_mask, x, y, width, height, resize_method="lanczos", feather=4.0, post_sharpen=0.0):
        width, height = _ensure_safe_size(width, height, "seamless paste")

        base = tensor2rgb(base_image).float().clamp(0.0, 1.0)
        patch = tensor2rgb(patch_image).float().clamp(0.0, 1.0)
        mask = tensor2mask(patch_mask).float().clamp(0.0, 1.0)

        patch = _repeat_to_batch(patch, base.shape[0])
        mask = _repeat_to_batch(mask, base.shape[0])

        patch = _resize_image(patch, width, height, resize_method)
        patch = _unsharp_tensor(patch, post_sharpen)
        mask = _resize_mask(mask, width, height, method="bilinear")
        if feather > 0:
            mask = _gaussian_blur_mask(mask, feather)

        out = base.clone()

        for b in range(base.shape[0]):
            x1 = x
            y1 = y
            x2 = x1 + width
            y2 = y1 + height

            sx1, sy1, sx2, sy2 = 0, 0, width, height

            if x1 < 0:
                sx1 = -x1
                x1 = 0
            if y1 < 0:
                sy1 = -y1
                y1 = 0
            if x2 > base.shape[2]:
                sx2 -= x2 - base.shape[2]
                x2 = base.shape[2]
            if y2 > base.shape[1]:
                sy2 -= y2 - base.shape[1]
                y2 = base.shape[1]

            if x1 >= x2 or y1 >= y2:
                continue

            alpha = mask[b, sy1:sy2, sx1:sx2].unsqueeze(-1)
            p = patch[b, sy1:sy2, sx1:sx2, :]
            out[b, y1:y2, x1:x2, :] = p * alpha + out[b, y1:y2, x1:x2, :] * (1.0 - alpha)

        return (out.clamp(0.0, 1.0),)


class I2IDetailPreserveBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "inpainted_image": ("IMAGE",),
                "mask": ("MASK",),
                "detail_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01}),
                "color_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_blur": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 64.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blend"
    CATEGORY = "I2I/Advanced"

    def blend(self, original_image, inpainted_image, mask, detail_strength, color_strength, mask_blur):
        orig = tensor2rgb(original_image).float().clamp(0.0, 1.0)
        gen = tensor2rgb(inpainted_image).float().clamp(0.0, 1.0)
        m = tensor2mask(mask).float().clamp(0.0, 1.0)

        batch = max(orig.shape[0], gen.shape[0], m.shape[0])
        orig = _repeat_to_batch(orig, batch)
        gen = _repeat_to_batch(gen, batch)
        m = _repeat_to_batch(m, batch)

        if m.shape[1] != orig.shape[1] or m.shape[2] != orig.shape[2]:
            m = _resize_mask(m, orig.shape[2], orig.shape[1], method="bilinear")

        if mask_blur > 0:
            m = _gaussian_blur_mask(m, mask_blur)

        out = []
        for i in range(batch):
            o = (orig[i].detach().cpu().numpy() * 255.0).astype(np.uint8)
            g = (gen[i].detach().cpu().numpy() * 255.0).astype(np.uint8)
            mk = m[i].detach().cpu().numpy().astype(np.float32)

            blur_o = cv2.GaussianBlur(o, (0, 0), sigmaX=1.2)
            high_o = o.astype(np.float32) - blur_o.astype(np.float32)

            detail_mix = g.astype(np.float32) + high_o * detail_strength
            color_mix = g.astype(np.float32) * (1.0 - color_strength) + o.astype(np.float32) * color_strength
            combined = detail_mix * (1.0 - color_strength) + color_mix * color_strength

            mk3 = np.repeat(mk[:, :, None], 3, axis=2)
            final = g.astype(np.float32) * (1.0 - mk3) + combined * mk3
            final = np.clip(final, 0, 255).astype(np.uint8)
            out.append(torch.from_numpy(final.astype(np.float32) / 255.0).to(device=orig.device, dtype=orig.dtype))

        return (torch.stack(out, dim=0),)


NODE_CLASS_MAPPINGS = {
    "Color Transfer": Color_Correction,
    "Mask Ops": Mask_Ops,
    "Inpaint Segments": MaskToRegion,
    "Combine and Paste": Combine_And_Paste_Op,
    "I2I Auto Align Image+Mask": I2IAutoAlignImageMask,
    "I2I Mask Refiner Pro": I2IMaskRefinerPro,
    "I2I Masked Tile Extractor": I2IMaskedTileExtractor,
    "I2I Region Overlay Debug": I2IRegionOverlayDebug,
    "I2I Seamless Patch Paste": I2ISeamlessPatchPaste,
    "I2I Detail Preserve Blend": I2IDetailPreserveBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Color Transfer": "Color Transfer",
    "Mask Ops": "Mask Ops",
    "Inpaint Segments": "Inpaint Segments",
    "Combine and Paste": "Combine and Paste",
    "I2I Auto Align Image+Mask": "I2I Auto Align Image+Mask",
    "I2I Mask Refiner Pro": "I2I Mask Refiner Pro",
    "I2I Masked Tile Extractor": "I2I Masked Tile Extractor",
    "I2I Region Overlay Debug": "I2I Region Overlay Debug",
    "I2I Seamless Patch Paste": "I2I Seamless Patch Paste",
    "I2I Detail Preserve Blend": "I2I Detail Preserve Blend",
}
