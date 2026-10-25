import argparse
import json
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from db import init_db, has_hash, insert_result
from hashing import sha256_file
from io_utils import list_images, read_rgb, safe_stem
from model import load_meta, build_preprocess, load_densenet121_from_state_dict, predict_softmax
from campp import GradCAMPlusPlus, densenet_target_layer
from masking import cam_to_mask_adaptive, cam_is_flat
from annotate import annotate_with_mask, draw_label
from export_results_to_csv import export_sqlite_to_csv, upload_to_google_sheets

def random_2026_date() -> str:
    start = datetime(2026, 1, 1)
    end = datetime(2026, 12, 31)
    offset_days = random.randint(0, (end - start).days)
    sampled = start + timedelta(days=offset_days)
    return sampled.strftime("%d/%m/%Y")

def random_plot() -> str:
    return f"PLT{random.randint(1, 20)}"

def random_farm_name() -> str:
    return random.choice(["Fazenda Recanto", "Fazenda Sol Nascente", "Fazenda Capao Rico"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="input_images")
    ap.add_argument("--output_dir", type=str, default="annotated_images")
    ap.add_argument("--db_path", type=str, default="results.sqlite")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--meta_path", type=str, required=True)
    ap.add_argument("--poll_seconds", type=float, default=5.0)
    ap.add_argument("--run_once", action="store_true")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mask_q_start", type=float, default=0.92)
    ap.add_argument("--mask_max_area", type=float, default=0.20)
    ap.add_argument("--mask_min_area", type=float, default=0.002)
    ap.add_argument("--highlight_strength", type=float, default=0.55)
    ap.add_argument("--no_contour", action="store_true")
    ap.add_argument("--skip_healthy_mask", action="store_true")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conn = init_db(Path(args.db_path))
    device = torch.device(args.device)

    meta = load_meta(Path(args.meta_path))
    classes = meta["classes"]
    tfm = build_preprocess(meta)
    model = load_densenet121_from_state_dict(Path(args.model_path), meta, device)
    campp = GradCAMPlusPlus(model, densenet_target_layer(model))

    print(f"🚀 Iniciando processamento em: {input_dir.resolve()}", flush=True)

    while True:
        files = list_images(input_dir)
        processou_algo = False  # 🚀 Flag para controlar o gatilho de atualização

        for p in files:
            sha = sha256_file(p)
            if has_hash(conn, sha): continue

            created_at = datetime.now(timezone.utc).isoformat()
            sampled_date = random_2026_date()
            farm_name = random_farm_name()
            plot = random_plot()
            
            try:
                rgb = read_rgb(p)
                H, W = rgb.shape[:2]
                x = tfm(Image.fromarray(rgb)).unsqueeze(0).to(device)
                probs_t = predict_softmax(model, x)
                probs = probs_t.numpy()
                pred_idx = int(np.argmax(probs))
                pred_label = classes[pred_idx]
                pred_prob = float(probs[pred_idx])

                mask = np.zeros((H, W), dtype=np.uint8)
                notes = None
                if not (args.skip_healthy_mask and pred_label.strip().lower() == "healthy"):
                    x2 = x.clone().requires_grad_(True)
                    cam = campp(x2, class_idx=pred_idx).numpy().astype(np.float32)
                    if cam_is_flat(cam): notes = "CAM flat; mask suppressed."
                    else: mask = cam_to_mask_adaptive(cam, H, W, q_start=args.mask_q_start)

                annot = annotate_with_mask(rgb, mask, highlight_strength=args.highlight_strength, draw_contour=(not args.no_contour))
                annot = draw_label(annot, f"{pred_label} ({pred_prob:.2f})")
                
                annot_file = output_dir / f"{safe_stem(p)}__{pred_label.replace(' ', '_')}__annot.png"
                Image.fromarray(annot).save(annot_file)
                
                insert_result(conn, created_at, sampled_date, farm_name, plot, str(p), sha, pred_label, pred_prob, 
                              json.dumps({classes[i]: float(probs[i]) for i in range(len(classes))}), str(annot_file), notes)
                print(f"✅ Processado: {p.name} -> {pred_label}")
                
                processou_algo = True # 🚀 Marcamos que o banco local mudou

            except Exception as e:
                print(f"❌ Erro em {p.name}: {e}")

        # 🚀 FORA DO LOOP FOR: Atualiza a nuvem uma única vez por lote!
        if processou_algo:
            print("📤 Sincronizando lote de novas inferências com o ecossistema...", flush=True)
            if export_sqlite_to_csv(Path(args.db_path), Path("results.csv"), flatten_probs=True):
                upload_to_google_sheets("results.csv", "Dashboard_Saude_Milho")

        if args.run_once: break
        time.sleep(args.poll_seconds)

if __name__ == "__main__":
    main()
