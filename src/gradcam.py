"""
gradcam.py — Grad-CAM for the multi-modal fusion model.

Unlike a single-encoder model, here each modality (SAR, optical, ...) has
its own ResNet18 branch feeding into a shared fusion head. To explain a
prediction we hook ONE branch at a time (whichever modality you want to
visualize) and backprop from the final fused logit through just that
branch's last conv block (layer4). Calling this once per modality gives
two separate heatmaps for the same building — e.g. "the SAR branch is
keying off rubble texture here, the optical branch is keying off a
collapsed roofline here" — a more informative explanation for a
multi-modal model than a single merged heatmap would be.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def generate_gradcam(model, inputs, target_modality, device):
    """inputs: dict of modality_name -> tensor (1,3,H,W), already
    preprocessed the same way dataset.py produces them.
    target_modality: which branch's layer4 to explain, e.g. 'sar' or 'opt'
    (must be a key present in `inputs` / this model's mode)."""
    model.eval()
    target_layer = model.branches[target_modality].layer4[-1]

    activation, gradient = {}, {}

    def fwd_hook(module, inp, out):
        activation["value"] = out

    def bwd_hook(module, grad_in, grad_out):
        gradient["value"] = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    # .clone() before requires_grad_ so we never mutate the caller's tensors
    # (v.to(device) is a no-op when already on that device, so without the
    # clone, requires_grad_ would silently flip the flag on the original
    # tensor the caller still holds a reference to)
    batch = {k: v.to(device).clone().requires_grad_(k == target_modality) for k, v in inputs.items()}
    logit = model(**batch)
    prob = torch.sigmoid(logit)

    model.zero_grad()
    logit.backward()

    grads = gradient["value"][0]      # (C,H,W)
    acts = activation["value"][0]     # (C,H,W)
    weights = grads.mean(dim=(1, 2))  # (C,)

    cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = F.relu(cam)
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    h1.remove()
    h2.remove()
    return cam.detach().cpu().numpy(), float(prob.item())


def overlay_heatmap(base_img_chw, cam_np, alpha=0.45):
    """base_img_chw: numpy array (3,H,W), roughly in [0,1] (the SAR or
    optical patch as loaded by dataset.py — already normalized, good
    enough for a visual overlay)."""
    h, w = base_img_chw.shape[1:]
    base = (np.clip(np.transpose(base_img_chw, (1, 2, 0)), 0, 1) * 255).astype(np.uint8)
    base_img = Image.fromarray(base)

    cam_img = Image.fromarray((cam_np * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    cam_arr = np.array(cam_img).astype(np.float32) / 255.0

    heat = np.zeros((h, w, 3), dtype=np.uint8)
    heat[..., 0] = (cam_arr * 255)                       # red channel = activation strength
    heat[..., 1] = (np.clip(1 - cam_arr, 0, 1) * 60)      # faint green in low-activation areas

    heat_img = Image.fromarray(heat)
    return Image.blend(base_img, heat_img, alpha=alpha)
