import cv2
import numpy as np


def cam_to_mask_adaptive(
    cam: np.ndarray,
    out_h: int,
    out_w: int,
    q_start: float = 0.92,
    q_max: float = 0.995,
    max_area_frac: float = 0.20,
    min_area_frac: float = 0.002,
) -> np.ndarray:
    cam_rs = cv2.resize(cam, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    cam_rs = np.clip(cam_rs, 0.0, 1.0)
    cam_rs = cv2.GaussianBlur(cam_rs, (0, 0), sigmaX=1.2)

    q = q_start
    best_mask = None

    while True:
        thr = float(np.quantile(cam_rs, q))
        mask = (cam_rs >= thr).astype(np.uint8) * 255

        k = max(3, (min(out_h, out_w) // 180) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        area_frac = float(mask.mean() / 255.0)
        best_mask = mask

        if area_frac > max_area_frac and q < q_max:
            q = min(q + 0.01, q_max)
            continue
        break

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(best_mask, connectivity=8)
    min_area = int(min_area_frac * out_h * out_w)

    clean = np.zeros_like(best_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == i] = 255

    return clean
def cam_is_flat(cam: np.ndarray, min_range: float = 0.15) -> bool:
    return float(cam.max() - cam.min()) < min_range