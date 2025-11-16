# Fusiona D1,D2,D3 -> dataset_merged/, entrena YOLOv8n y exporta a TFLite (FP16)
from pathlib import Path
import shutil, hashlib, yaml
from collections import defaultdict
from ultralytics import YOLO

# === CONFIGURACIÓN ===
# === CONFIGURACIÓN ===
ROOT = Path(r"E:\DATASETS")  # ← antes estaba en C:\...
DATASETS = [ROOT/"D1", ROOT/"D2", ROOT/"D3"]
OUT = ROOT/"dataset_merged"
NAMES = ['1000','10000','2000','20000','5000']
MODEL = "yolov8n.pt"   # o "yolov11n.pt"
IMGSZ = 640
EPOCHS = 100
BATCH = 32
DEVICE = 0


# === UTILIDADES ===
IMG_EXTS = {".jpg",".jpeg",".png",".bmp"}
def sha256(p: Path, buf: int = 1<<20) -> str:
    h = hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda: f.read(buf), b''):
            h.update(b)
    return h.hexdigest()

def ensure_dirs(base: Path):
    for s in ["train","val","test"]:
        (base/s/"images").mkdir(parents=True, exist_ok=True)
        (base/s/"labels").mkdir(parents=True, exist_ok=True)

def merge_datasets(roots, out: Path):
    ensure_dirs(out)
    seen = set()
    counts = defaultdict(int)

    for r in roots:
        if not r.exists():
            print(f"[WARN] No existe: {r}")
            continue
        for split in ["train","val","valid","test"]:
            imgd = r/split/"images"
            lbld = r/split/"labels"
            if not (imgd.is_dir() and lbld.is_dir()):
                continue
            dst_split = "val" if split=="valid" else split
            for img in sorted(imgd.iterdir()):
                if img.suffix.lower() not in IMG_EXTS:
                    continue
                lbl = lbld/(img.stem + ".txt")
                if not lbl.exists():
                    # crea etiqueta vacía si no hay objetos
                    lbl = out/"_tmp_empty.txt"
                    lbl.write_text("", encoding="utf-8")

                h = sha256(img)
                if h in seen:
                    counts[f"dup_{dst_split}"] += 1
                    continue
                seen.add(h)

                i = counts[f"{dst_split}_copied"]
                new_img = out/dst_split/"images"/f"{dst_split}_{i:06d}{img.suffix.lower()}"
                new_lbl = out/dst_split/"labels"/(new_img.stem + ".txt")
                shutil.copy2(img, new_img)
                shutil.copy2(lbl, new_lbl)
                counts[f"{dst_split}_copied"] += 1

        print(f"[OK] Merge desde: {r.name}")

    data = {
        "train": (out/"train"/"images").as_posix(),
        "val":   (out/"val"/"images").as_posix(),
        "test":  (out/"test"/"images").as_posix(),
        "nc": len(NAMES),
        "names": NAMES
    }
    (out/"data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print("[OK] data.yaml escrito en", out)
    for k in sorted(counts): print(k, counts[k])

def main():
    print("== FUSIONANDO D1+D2+D3 ==")
    merge_datasets(DATASETS, OUT)

    print("\n== ENTRENANDO YOLO ==")
    model = YOLO(MODEL)
    results = model.train(
        data=str(OUT/"data.yaml"),
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE
    )
    best = Path(results.save_dir)/"weights"/"best.pt"
    print("[OK] Entrenamiento listo:", best)

    print("\n== EXPORTANDO A TFLite (FP16) ==")
    export_path = model.export(format="tflite", half=True)  # devuelve ruta del .tflite
    print("[OK] Export:", export_path)

if _name_ == "_main_":
    main()
