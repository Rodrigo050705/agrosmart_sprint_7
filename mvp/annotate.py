import cv2
import numpy as np


def annotate_with_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    highlight_strength: float = 0.55,
    draw_contour: bool = True,
) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    m = (mask.astype(np.float32) / 255.0)[..., None]  # HxWx1

    out = out * (1.0 + m * highlight_strength)
    out = np.clip(out, 0, 255).astype(np.uint8)

    if draw_contour:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.drawContours(bgr, cnts, -1, (0, 255, 255), 2)
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return out


def draw_label(rgb: np.ndarray, text: str) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)