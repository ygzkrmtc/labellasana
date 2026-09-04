"""
Model eğitimi altyapısı ve model kütüphanesi.

Tasarım kararı: eğitim `ultralytics` + `torch` ister, bu ikisi paketlenmiş
uygulamaya ~2 GB ekler. Bu yüzden eğitim, uygulamanın İÇİNDE değil, proje
klasörünün yanında kurulan AYRI bir Python ortamında alt süreç olarak çalışır.
Uygulama sadece komutu kurar, çıktıyı okur ve biten modeli ONNX olarak
kütüphaneye kaydeder. Böylece `.exe` küçük kalır, eğitim yeteneği isteğe bağlı olur.

Bu modül Qt bilmez; süreç yönetimi `train_dialog.py` tarafındadır.
"""

import json
import os
import re
import shutil
import sys
from typing import Dict, List, Optional, Sequence, Tuple

MODELS_DIRNAME = "models"
TRAIN_ENV_DIRNAME = ".venv-train"
WORK_DIRNAME = "_train"
DATASET_DIRNAME = "_train_dataset"
RUN_NAME = "run"

# Alt sürecin bize yazdığı makine-okunur işaretler (çıktı ayrıştırmayı
# kırılganlıktan kurtarır: ilerleme metni değişse de bunlar sabittir)
MARK_BEST = "LABELER_BEST::"
MARK_ONNX = "LABELER_ONNX::"
MARK_ERROR = "LABELER_ERROR::"
MARK_DEVICE = "LABELER_DEVICE::"

BASE_MODELS = ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt")

_EPOCH_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s")
_METRICS_RE = re.compile(
    r"^\s*all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")


# ------------------------------------------------------------------ yollar
def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def base_python() -> str:
    """Eğitim ortamını kurmak için kullanılabilecek Python. Exe'de boştur."""
    return "" if is_frozen() else sys.executable


def venv_python(env_dir: str) -> str:
    if os.name == "nt":
        return os.path.join(env_dir, "Scripts", "python.exe")
    return os.path.join(env_dir, "bin", "python")


def training_env_dir(project_dir: str) -> str:
    return os.path.join(project_dir, TRAIN_ENV_DIRNAME)


def find_training_python(project_dir: str) -> Optional[str]:
    """Proje yanında kurulmuş eğitim ortamı varsa onun python'unu döndürür."""
    if not project_dir:
        return None
    candidate = venv_python(training_env_dir(project_dir))
    return candidate if os.path.isfile(candidate) else None


def work_dir(project_dir: str) -> str:
    return os.path.join(project_dir, WORK_DIRNAME)


def dataset_dir(project_dir: str) -> str:
    return os.path.join(project_dir, DATASET_DIRNAME)


def models_dir(project_dir: str) -> str:
    return os.path.join(project_dir, MODELS_DIRNAME)


# ------------------------------------------------------------------ kurulum
def setup_commands(python_exe: str, env_dir: str) -> List[Tuple[str, List[str]]]:
    """
    Eğitim ortamını kuran komut dizisi: (açıklama, [program, *argümanlar]).

    Sırayla çalıştırılır; biri başarısız olursa zincir durur.
    """
    target_python = venv_python(env_dir)
    return [
        ("Sanal ortam oluşturuluyor", [python_exe, "-m", "venv", env_dir]),
        ("pip güncelleniyor",
         [target_python, "-m", "pip", "install", "--upgrade", "pip", "--quiet"]),
        ("ultralytics kuruluyor (~2 GB, bu adım uzun sürer)",
         [target_python, "-m", "pip", "install", "ultralytics", "onnx"]),
    ]


def python_version_warning() -> str:
    """
    torch/ultralytics, çok yeni Python sürümleri için genellikle geç paket
    yayınlar. Kurulum saatlerce sürüp başarısız olmasın diye önceden uyarıyoruz.
    """
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 13):
        return (f"Bu Python sürümü ({major}.{minor}) için PyTorch henüz hazır "
                f"paket yayınlamamış olabilir; kurulum başarısız olabilir. "
                f"Eğitim ortamı için Python 3.12 önerilir — kurup "
                f"'Var Olan Python'u Seç' ile gösterebilirsiniz. Bu, "
                f"uygulamanın kendi çalıştığı Python'u etkilemez.")
    return ""


def probe_command(python_exe: str) -> List[str]:
    """Verilen Python'da ultralytics var mı, GPU var mı?"""
    code = (
        "import json,sys\n"
        "info={'python':sys.version.split()[0]}\n"
        "try:\n"
        "    import ultralytics; info['ultralytics']=ultralytics.__version__\n"
        "except Exception as e:\n"
        "    info['ultralytics']=None; info['error']=str(e)\n"
        "try:\n"
        "    import torch\n"
        "    info['torch']=torch.__version__\n"
        "    info['cuda']=bool(torch.cuda.is_available())\n"
        "    info['gpu']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''\n"
        "except Exception:\n"
        "    info['torch']=None; info['cuda']=False; info['gpu']=''\n"
        "print('PROBE::'+json.dumps(info))\n"
    )
    return [python_exe, "-c", code]


def parse_probe(output: str) -> dict:
    for line in output.splitlines():
        if line.startswith("PROBE::"):
            try:
                return json.loads(line[len("PROBE::"):])
            except ValueError:
                return {}
    return {}


# ------------------------------------------------------------------ eğitim
def default_options(image_count: int, has_gpu: bool = False) -> dict:
    """
    Veri seti büyüklüğüne göre makul varsayılanlar.

    Küçük setlerde daha çok epoch ve RAM önbelleği; büyük setlerde tam tersi.
    Toplu iş boyutu GPU'da otomatik (-1), CPU'da sabit ve düşük tutulur.
    """
    if image_count <= 0:
        image_count = 1
    if image_count < 100:
        epochs = 150
    elif image_count < 500:
        epochs = 120
    elif image_count < 2000:
        epochs = 80
    else:
        epochs = 60
    return {
        "base_model": "yolov8n.pt",
        "epochs": epochs,
        "imgsz": 640,
        "batch": 0,                       # 0 = otomatik
        "patience": max(20, epochs // 4),
        "cache": image_count <= 3000,     # RAM'e sığar, her epoch'ta diskten okumaz
        "workers": 2 if os.name == "nt" else 4,
        "device": "",                     # "" = otomatik
        "seed": 0,
    }


def build_config(data_yaml: str, project_dir: str, options: dict) -> dict:
    merged = dict(options or {})
    merged.update({
        "data": os.path.abspath(data_yaml),
        "project": os.path.abspath(work_dir(project_dir)),
        "name": RUN_NAME,
    })
    return merged


TRAIN_SCRIPT = '''# Etiketleme Aracı tarafından üretildi - eğitim alt süreci
import json
import sys
from pathlib import Path

def main():
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    from ultralytics import YOLO

    batch = cfg.get("batch") or 0
    device = cfg.get("device") or None
    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False
    print("{device_mark}" + ("cuda" if has_cuda else "cpu"), flush=True)
    if not batch:
        batch = -1 if has_cuda else 8

    model = YOLO(cfg["base_model"])
    model.train(
        data=cfg["data"],
        epochs=int(cfg["epochs"]),
        imgsz=int(cfg["imgsz"]),
        batch=batch,
        workers=int(cfg.get("workers", 2)),
        device=device,
        project=cfg["project"],
        name=cfg["name"],
        exist_ok=True,
        patience=int(cfg.get("patience", 30)),
        cache=bool(cfg.get("cache", False)),
        seed=int(cfg.get("seed", 0)),
        verbose=True,
        plots=False,
    )

    best = Path(model.trainer.best)
    print("{best_mark}" + str(best), flush=True)

    exported = YOLO(str(best)).export(
        format="onnx", opset=12, dynamic=False, simplify=False,
        imgsz=int(cfg["imgsz"]))
    print("{onnx_mark}" + str(exported), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:            # hata alt süreçte kalmasın, arayüze taşınsın
        import traceback
        traceback.print_exc()
        print("{error_mark}" + str(exc), flush=True)
        sys.exit(1)
'''


def write_job(project_dir: str, config: dict) -> Tuple[str, str]:
    """Eğitim betiğini ve yapılandırmasını diske yazar; (betik, config) döndürür."""
    folder = work_dir(project_dir)
    os.makedirs(folder, exist_ok=True)
    script_path = os.path.join(folder, "train_job.py")
    config_path = os.path.join(folder, "train_config.json")

    source = TRAIN_SCRIPT.format(device_mark=MARK_DEVICE, best_mark=MARK_BEST,
                                 onnx_mark=MARK_ONNX, error_mark=MARK_ERROR)
    with open(script_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(source)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    return script_path, config_path


def parse_line(line: str) -> Optional[dict]:
    """
    Eğitim çıktısındaki bir satırı yorumlar.

    Dönüş: {"kind": "epoch"|"metrics"|"best"|"onnx"|"error"|"device", ...} veya None.
    """
    text = line.rstrip()
    if not text:
        return None

    if text.startswith(MARK_BEST):
        return {"kind": "best", "path": text[len(MARK_BEST):].strip()}
    if text.startswith(MARK_ONNX):
        return {"kind": "onnx", "path": text[len(MARK_ONNX):].strip()}
    if text.startswith(MARK_ERROR):
        return {"kind": "error", "message": text[len(MARK_ERROR):].strip()}
    if text.startswith(MARK_DEVICE):
        return {"kind": "device", "device": text[len(MARK_DEVICE):].strip()}

    metrics = _METRICS_RE.match(text)
    if metrics:
        precision, recall, map50, map95 = (float(v) for v in metrics.groups())
        return {"kind": "metrics", "precision": precision, "recall": recall,
                "map50": map50, "map50_95": map95}

    epoch = _EPOCH_RE.match(text)
    if epoch:
        current, total = int(epoch.group(1)), int(epoch.group(2))
        if 0 < current <= total:
            return {"kind": "epoch", "current": current, "total": total}
    return None


# ------------------------------------------------------------------ kütüphane
class ModelLibrary:
    """Proje klasöründeki `models/` dizininde duran ONNX modelleri."""

    def __init__(self, project_dir: str):
        self.project_dir = project_dir or ""
        self.directory = models_dir(self.project_dir) if self.project_dir else ""

    def ensure(self) -> str:
        if self.directory:
            os.makedirs(self.directory, exist_ok=True)
        return self.directory

    def list_models(self) -> List[str]:
        """Yeniden eskiye doğru sıralı ONNX dosyaları."""
        if not self.directory or not os.path.isdir(self.directory):
            return []
        entries = []
        for name in os.listdir(self.directory):
            if not name.lower().endswith(".onnx"):
                continue
            path = os.path.join(self.directory, name)
            if os.path.isfile(path):
                try:
                    entries.append((os.path.getmtime(path), path))
                except OSError:
                    entries.append((0.0, path))
        entries.sort(reverse=True)
        return [path for _, path in entries]

    def register(self, source_path: str, name: str) -> str:
        """Eğitilmiş ONNX'i kütüphaneye kopyalar; hedef yolu döndürür."""
        if not os.path.isfile(source_path):
            raise FileNotFoundError(source_path)
        self.ensure()
        target = os.path.join(self.directory, (safe_name(name) or "model") + ".onnx")
        if os.path.abspath(source_path) != os.path.abspath(target):
            shutil.copy2(source_path, target)
        return target

    def write_metadata(self, model_path: str, metadata: dict) -> None:
        """Modelin yanına eğitim bilgilerini yazar (ne zaman, hangi sınıflar, mAP)."""
        sidecar = os.path.splitext(model_path)[0] + ".json"
        try:
            with open(sidecar, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def read_metadata(self, model_path: str) -> dict:
        sidecar = os.path.splitext(model_path)[0] + ".json"
        if not os.path.isfile(sidecar):
            return {}
        try:
            with open(sidecar, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}


def safe_name(name: str) -> str:
    """Dosya adı olarak güvenli hale getirir."""
    cleaned = re.sub(r"[^\w\-.]+", "_", (name or "").strip(), flags=re.UNICODE)
    return cleaned.strip("._")


def suggest_model_name(class_names: Sequence[str], existing: Sequence[str]) -> str:
    """aku -> aku_v1, aku_v2 ... ; çok sınıflıysa 'model_v1'."""
    stem = safe_name(class_names[0]) if len(class_names) == 1 else "model"
    stem = stem or "model"
    taken = {os.path.splitext(os.path.basename(path))[0].lower()
             for path in existing}
    index = 1
    while f"{stem}_v{index}".lower() in taken:
        index += 1
    return f"{stem}_v{index}"


def describe_model(path: str, metadata: Dict) -> str:
    """Menüde gösterilecek kısa açıklama."""
    name = os.path.splitext(os.path.basename(path))[0]
    parts = [name]
    classes = metadata.get("classes")
    if classes:
        parts.append(f"{len(classes)} sınıf")
    map50 = metadata.get("map50")
    if isinstance(map50, (int, float)) and map50 > 0:
        parts.append(f"mAP50 {map50:.2f}")
    return "  ·  ".join(parts)
