"""
Model eğitimi penceresi.

Eğitim, ayrı bir Python ortamında alt süreç olarak çalışır (QProcess). Arayüz
donmaz, çıktı canlı akar, istenirse durdurulur. Biten model otomatik olarak
ONNX'e çevrilip proje klasöründeki `models/` kütüphanesine kaydedilir.
"""

import os
import time
from typing import List, Optional, Sequence, Tuple

from PyQt5.QtCore import QProcess, QProcessEnvironment, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                             QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
                             QProgressBar, QPushButton, QSpinBox, QVBoxLayout)

from . import training

MAX_LOG_BLOCKS = 4000


class TrainDialog(QDialog):
    """
    Eğitim akışı:
      1. Ortam kontrolü (ultralytics kurulu mu, GPU var mı)
      2. Veri seti dışa aktarımı (çağıran taraf yapar, data.yaml verilir)
      3. Alt süreçte eğitim + ONNX'e çevirme
      4. Modeli kütüphaneye kaydetme
    """

    modelReady = pyqtSignal(str)          # kütüphaneye kaydedilen .onnx yolu

    def __init__(self, parent, project_dir: str, class_names: Sequence[str],
                 image_count: int, dataset_builder):
        super().__init__(parent)
        self.setWindowTitle("Model Eğit")
        self.setMinimumSize(820, 640)

        self.project_dir = project_dir
        self.class_names = list(class_names)
        self.image_count = int(image_count)
        self._dataset_builder = dataset_builder      # () -> data.yaml yolu | None

        self.library = training.ModelLibrary(project_dir)
        self._process: Optional[QProcess] = None
        self._setup_queue: List[Tuple[str, List[str]]] = []
        self._phase = "idle"                         # idle | setup | probe | train
        self._probe_buffer = ""
        self._best_path = ""
        self._onnx_path = ""
        self._device = ""
        self._last_metrics = {}
        self._started_at = 0.0
        self._train_python = training.find_training_python(project_dir) or ""

        self._build_ui()
        self._refresh_environment()

    # ---------------------------------------------------------------- arayüz
    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- ortam
        env_box = QGroupBox("Eğitim ortamı", self)
        env_layout = QVBoxLayout(env_box)
        self.env_label = QLabel("Denetleniyor...")
        self.env_label.setWordWrap(True)
        env_layout.addWidget(self.env_label)

        env_buttons = QHBoxLayout()
        self.setup_button = QPushButton("Eğitim Ortamını Kur", self)
        self.setup_button.clicked.connect(self.setup_environment)
        self.browse_python_button = QPushButton("Var Olan Python'u Seç...", self)
        self.browse_python_button.clicked.connect(self.browse_python)
        env_buttons.addWidget(self.setup_button)
        env_buttons.addWidget(self.browse_python_button)
        env_buttons.addStretch(1)
        env_layout.addLayout(env_buttons)
        layout.addWidget(env_box)

        # --- ayarlar
        options_box = QGroupBox("Eğitim ayarları", self)
        form = QFormLayout(options_box)
        defaults = training.default_options(self.image_count)

        self.model_combo = QComboBox(self)
        for name, description in (("yolov8n.pt", "n — en hızlı, en küçük (başlangıç için)"),
                                  ("yolov8s.pt", "s — dengeli"),
                                  ("yolov8m.pt", "m — en isabetli, en yavaş")):
            self.model_combo.addItem(description, name)
        form.addRow("Temel model:", self.model_combo)

        self.epochs_spin = QSpinBox(self)
        self.epochs_spin.setRange(1, 2000)
        self.epochs_spin.setValue(defaults["epochs"])
        form.addRow("Epoch sayısı:", self.epochs_spin)

        self.imgsz_combo = QComboBox(self)
        for size in (416, 512, 640, 800, 960):
            self.imgsz_combo.addItem(f"{size} px", size)
        self.imgsz_combo.setCurrentIndex(2)
        self.imgsz_combo.setToolTip(
            "Eğitim çözünürlüğü. Küçük nesneler için büyütmek işe yarar ama "
            "eğitim yavaşlar. Tahmin de aynı boyutta yapılır.")
        form.addRow("Görüntü boyutu:", self.imgsz_combo)

        self.batch_spin = QSpinBox(self)
        self.batch_spin.setRange(0, 128)
        self.batch_spin.setValue(0)
        self.batch_spin.setSpecialValueText("Otomatik")
        self.batch_spin.setToolTip("0 = otomatik (GPU'da bellek doldurulur, CPU'da 8).")
        form.addRow("Toplu iş boyutu:", self.batch_spin)

        self.patience_spin = QSpinBox(self)
        self.patience_spin.setRange(0, 500)
        self.patience_spin.setValue(defaults["patience"])
        self.patience_spin.setToolTip(
            "Bu kadar epoch boyunca iyileşme olmazsa eğitim erken durur.")
        form.addRow("Sabır (erken durdurma):", self.patience_spin)

        self.cache_check = QCheckBox("Resimleri RAM'de önbelleğe al (küçük setlerde hızlandırır)")
        self.cache_check.setChecked(defaults["cache"])
        form.addRow("", self.cache_check)

        self.device_combo = QComboBox(self)
        self.device_combo.addItem("Otomatik", "")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("GPU (cuda:0)", "0")
        form.addRow("Donanım:", self.device_combo)

        self.name_edit = QLineEdit(
            training.suggest_model_name(self.class_names, self.library.list_models()), self)
        form.addRow("Model adı:", self.name_edit)
        layout.addWidget(options_box)

        # --- uyarı
        self.advice = QLabel()
        self.advice.setWordWrap(True)
        self.advice.setStyleSheet("color: #ef6c00;")
        layout.addWidget(self.advice)
        self._update_advice()

        # --- ilerleme + günlük
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Hazır")
        layout.addWidget(self.progress)

        self.metrics_label = QLabel("")
        self.metrics_label.setStyleSheet("color: #888;")
        layout.addWidget(self.metrics_label)

        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(MAX_LOG_BLOCKS)
        monospace = QFont("Consolas" if os.name == "nt" else "Monospace")
        monospace.setStyleHint(QFont.TypeWriter)
        monospace.setPointSize(9)
        self.log.setFont(monospace)
        layout.addWidget(self.log, 1)

        # --- düğmeler
        buttons = QDialogButtonBox(self)
        self.start_button = QPushButton("Eğitimi Başlat", self)
        self.start_button.setDefault(True)
        self.start_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("Durdur", self)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_process)
        buttons.addButton(self.start_button, QDialogButtonBox.AcceptRole)
        buttons.addButton(self.stop_button, QDialogButtonBox.DestructiveRole)
        close_button = buttons.addButton(QDialogButtonBox.Close)
        close_button.clicked.connect(self.close)
        layout.addWidget(buttons)

    def _update_advice(self):
        if self.image_count < 50:
            self.advice.setText(
                f"⚠ Şu an {self.image_count} etiketli resim var. Anlamlı bir model "
                f"için genelde en az 100-200 etiketli resim gerekir; daha azıyla "
                f"eğitilen model işe yaramayan öneriler üretir.")
        elif self.image_count < 150:
            self.advice.setText(
                f"{self.image_count} etiketli resim başlangıç için yeterli olabilir; "
                f"model ilk turda kabaca doğru kutular önerir, siz düzelttikçe "
                f"bir sonraki eğitim belirgin biçimde iyileşir.")
        else:
            self.advice.setText("")

    # ---------------------------------------------------------------- ortam
    def _refresh_environment(self):
        found = self._train_python or training.find_training_python(self.project_dir)
        if not found:
            self.env_label.setText(
                "Eğitim ortamı kurulu değil. <b>Eğitim Ortamını Kur</b> ile proje "
                "klasörünün yanına kurulur (~2 GB indirme, bir kez). Zaten "
                "<code>ultralytics</code> kurulu bir Python'unuz varsa onu seçebilirsiniz."
                + ("<br><b>Not:</b> uygulama .exe olarak çalıştığı için kurulumu "
                   "başlatmak üzere sistemde kurulu bir Python seçmeniz gerekir."
                   if training.is_frozen() else ""))
            self.setup_button.setEnabled(not training.is_frozen()
                                         or bool(training.base_python()))
            self.start_button.setEnabled(False)
            return

        self._train_python = found
        self.env_label.setText(f"Denetleniyor: <code>{found}</code>")
        self._start_probe(found)

    def _start_probe(self, python_exe: str):
        self._phase = "probe"
        self._probe_buffer = ""
        command = training.probe_command(python_exe)
        self._start_process(command[0], command[1:], log_command=False)

    def _apply_probe(self, info: dict):
        version = info.get("ultralytics")
        if not version:
            self.env_label.setText(
                f"<code>{self._train_python}</code><br>"
                f"<span style='color:#c62828;'>ultralytics kurulu değil.</span> "
                f"'Eğitim Ortamını Kur' ile kurabilirsiniz.")
            self.start_button.setEnabled(False)
            return

        cuda = info.get("cuda")
        gpu = info.get("gpu") or ""
        hardware = (f"<span style='color:#2e7d32;'>GPU: {gpu}</span>" if cuda
                    else "<span style='color:#ef6c00;'>GPU bulunamadı — CPU ile "
                         "eğitim çok yavaş olur (küçük setlerde yine de yapılabilir)."
                         "</span>")
        self.env_label.setText(
            f"<code>{self._train_python}</code><br>"
            f"Python {info.get('python', '?')} · ultralytics {version} · "
            f"torch {info.get('torch') or '?'}<br>{hardware}")
        self.start_button.setEnabled(True)
        if not cuda:
            self.device_combo.setCurrentIndex(1)     # CPU

    def browse_python(self):
        start_dir = os.path.dirname(self._train_python) if self._train_python else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "ultralytics kurulu Python'u seçin", start_dir,
            "Python (python.exe python3 python)" if os.name == "nt" else "Python (*)")
        if not path:
            return
        self._train_python = path
        self.env_label.setText(f"Denetleniyor: <code>{path}</code>")
        self._start_probe(path)

    def setup_environment(self):
        if self._process is not None:
            QMessageBox.information(self, "Meşgul", "Önce çalışan işlemi durdurun.")
            return
        python_exe = training.base_python()
        if not python_exe:
            QMessageBox.information(
                self, "Python gerekli",
                "Uygulama paketlenmiş olarak çalıştığı için ortamı kuracak bir "
                "Python bulunamadı. 'Var Olan Python'u Seç' ile sisteminizdeki "
                "python.exe dosyasını gösterin.")
            return
        if not self.project_dir:
            QMessageBox.information(self, "Klasör yok", "Önce bir resim klasörü açın.")
            return

        warning = training.python_version_warning()
        reply = QMessageBox.question(
            self, "Eğitim ortamını kur",
            "Proje klasörünün yanına ayrı bir Python ortamı kurulacak ve "
            "<b>ultralytics</b> indirilecek (~2 GB, internet gerekir).<br><br>"
            "Bu ortam yalnızca eğitim için kullanılır; uygulamanın kendisi ve "
            "üretilecek .exe bundan etkilenmez.<br><br>"
            f"Kurulacağı Python:<br><code>{python_exe}</code><br>"
            + (f"<br><span style='color:#ef6c00;'>{warning}</span><br>" if warning else "")
            + "<br>Devam edilsin mi?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No if warning else QMessageBox.Yes)
        if reply != QMessageBox.Yes:
            return

        env_dir = training.training_env_dir(self.project_dir)
        self._setup_queue = list(training.setup_commands(python_exe, env_dir))
        self._phase = "setup"
        self._set_running(True)
        self.progress.setRange(0, 0)                 # belirsiz ilerleme
        self.progress.setFormat("Ortam kuruluyor...")
        self._run_next_setup_step()

    def _run_next_setup_step(self):
        if not self._setup_queue:
            self._phase = "idle"
            self._set_running(False)
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress.setFormat("Ortam hazır")
            self._append("\n=== Eğitim ortamı kuruldu ===\n")
            self._train_python = training.venv_python(
                training.training_env_dir(self.project_dir))
            self._refresh_environment()
            return
        description, command = self._setup_queue.pop(0)
        self._append(f"\n=== {description} ===\n")
        self._start_process(command[0], command[1:])

    # ---------------------------------------------------------------- eğitim
    def start_training(self):
        if self._process is not None:
            return
        if not self._train_python:
            QMessageBox.information(self, "Ortam yok", "Önce eğitim ortamını kurun.")
            return

        self.log.clear()
        self._append("Veri seti hazırlanıyor...\n")
        data_yaml = self._dataset_builder()
        if not data_yaml:
            self._append("Veri seti hazırlanamadı; eğitim başlatılmadı.\n")
            return
        self._append(f"data.yaml: {data_yaml}\n")

        options = {
            "base_model": self.model_combo.currentData(),
            "epochs": self.epochs_spin.value(),
            "imgsz": self.imgsz_combo.currentData(),
            "batch": self.batch_spin.value(),
            "patience": self.patience_spin.value(),
            "cache": self.cache_check.isChecked(),
            "workers": 2 if os.name == "nt" else 4,
            "device": self.device_combo.currentData(),
            "seed": 0,
        }
        config = training.build_config(data_yaml, self.project_dir, options)
        script_path, config_path = training.write_job(self.project_dir, config)

        self._best_path = ""
        self._onnx_path = ""
        self._last_metrics = {}
        self._started_at = time.time()
        self._phase = "train"
        self._set_running(True)
        self.progress.setRange(0, self.epochs_spin.value())
        self.progress.setValue(0)
        self.progress.setFormat("Başlatılıyor... %v/%m epoch")
        self._append(f"\n=== Eğitim başlıyor ({options['epochs']} epoch) ===\n")
        self._start_process(self._train_python, [script_path, config_path])

    # ---------------------------------------------------------------- süreç
    def _start_process(self, program: str, arguments: List[str],
                       log_command: bool = True):
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.setWorkingDirectory(self.project_dir or os.getcwd())
        # systemEnvironment() şart: boş bir ortamla başlarsak alt süreç PATH'i
        # kaybeder ve hiçbir şey çalışmaz.
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")   # çıktı anında aksın
        environment.insert("PYTHONIOENCODING", "utf-8")
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._on_output)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_process_error)
        self._process = process
        if log_command:
            self._append(f"$ {program} {' '.join(arguments)}\n")
        process.start(program, arguments)

    def _on_output(self):
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput())
        text = data.decode("utf-8", errors="replace")
        if self._phase == "probe":
            self._probe_buffer += text
            return
        for line in text.splitlines():
            self._handle_line(line)

    def _handle_line(self, line: str):
        parsed = training.parse_line(line)
        if parsed is None:
            self._append(line + "\n")
            return

        kind = parsed["kind"]
        if kind == "epoch":
            self.progress.setMaximum(parsed["total"])
            self.progress.setValue(parsed["current"])
            elapsed = time.time() - self._started_at
            per_epoch = elapsed / max(1, parsed["current"])
            remaining = per_epoch * (parsed["total"] - parsed["current"])
            self.progress.setFormat(
                f"%v/%m epoch · kalan ~{self._format_duration(remaining)}")
            self._append(line + "\n")
        elif kind == "metrics":
            self._last_metrics = parsed
            self.metrics_label.setText(
                f"mAP50: <b>{parsed['map50']:.3f}</b> · "
                f"mAP50-95: {parsed['map50_95']:.3f} · "
                f"kesinlik {parsed['precision']:.3f} · duyarlılık {parsed['recall']:.3f}")
            self._append(line + "\n")
        elif kind == "device":
            self._device = parsed["device"]
            self._append(f"[donanım: {self._device}]\n")
        elif kind == "best":
            self._best_path = parsed["path"]
            self._append(f"\nEn iyi ağırlık: {self._best_path}\n")
        elif kind == "onnx":
            self._onnx_path = parsed["path"]
            self._append(f"ONNX: {self._onnx_path}\n")
        elif kind == "error":
            self._append(f"\nHATA: {parsed['message']}\n")

    def _on_process_error(self, error):
        names = {
            QProcess.FailedToStart: "Program başlatılamadı (yol yanlış olabilir).",
            QProcess.Crashed: "İşlem beklenmedik şekilde sonlandı.",
            QProcess.Timedout: "İşlem zaman aşımına uğradı.",
        }
        self._append("\n" + names.get(error, "İşlem hatası.") + "\n")

    def _on_finished(self, exit_code: int, _status):
        phase = self._phase
        self._process = None

        if phase == "probe":
            info = training.parse_probe(self._probe_buffer)
            self._phase = "idle"
            self._apply_probe(info)
            return

        if phase == "setup":
            if exit_code != 0:
                self._phase = "idle"
                self._set_running(False)
                self.progress.setRange(0, 100)
                self.progress.setFormat("Kurulum başarısız")
                self._append("\nKurulum adımı başarısız oldu. Yukarıdaki çıktıya bakın.\n")
                return
            self._run_next_setup_step()
            return

        # ---- eğitim bitti
        self._phase = "idle"
        self._set_running(False)
        if exit_code != 0 and not self._onnx_path:
            self.progress.setFormat("Eğitim durdu")
            self._append(f"\nEğitim {exit_code} koduyla sonlandı.\n")
            return

        if not self._onnx_path or not os.path.isfile(self._onnx_path):
            self.progress.setFormat("ONNX üretilemedi")
            QMessageBox.warning(
                self, "Model dışa aktarılamadı",
                "Eğitim tamamlandı ama ONNX dosyası bulunamadı. Günlükteki "
                "hatalara bakın; ağırlıklar şurada olabilir:\n"
                f"{self._best_path or training.work_dir(self.project_dir)}")
            return

        try:
            target = self.library.register(self._onnx_path, self.name_edit.text())
        except OSError as exc:
            QMessageBox.critical(self, "Kaydedilemedi", str(exc))
            return

        self.library.write_metadata(target, {
            "created": time.strftime("%Y-%m-%d %H:%M"),
            "classes": self.class_names,
            "images": self.image_count,
            "epochs": self.epochs_spin.value(),
            "imgsz": self.imgsz_combo.currentData(),
            "base_model": self.model_combo.currentData(),
            "device": self._device,
            "map50": self._last_metrics.get("map50"),
            "map50_95": self._last_metrics.get("map50_95"),
            "duration_sec": int(time.time() - self._started_at),
        })

        self.progress.setValue(self.progress.maximum())
        self.progress.setFormat("Tamamlandı")
        self._append(f"\n=== Model kütüphaneye eklendi: {target} ===\n")
        self.modelReady.emit(target)

        map50 = self._last_metrics.get("map50")
        quality = ""
        if isinstance(map50, float):
            if map50 >= 0.7:
                quality = "Bu değer iyi: model çoğu kutuyu doğru önerecektir."
            elif map50 >= 0.4:
                quality = ("Bu değer orta: model işe yarar öneriler üretir ama "
                           "düzeltme gerekir. Daha çok etiketle tekrar eğitin.")
            else:
                quality = ("Bu değer düşük: daha çok etiketli resim gerekiyor "
                           "ya da epoch sayısını artırın.")
        QMessageBox.information(
            self, "Eğitim tamamlandı",
            f"<b>{os.path.basename(target)}</b> kütüphaneye eklendi ve yüklendi."
            + (f"<br><br>mAP50: <b>{map50:.3f}</b><br>{quality}" if isinstance(map50, float)
               else "")
            + "<br><br>Artık <b>Ctrl+R</b> ile tahmin alabilirsiniz.")

    def stop_process(self):
        if self._process is None:
            return
        self._append("\nDurduruluyor...\n")
        self._process.kill()

    def _set_running(self, running: bool):
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.setup_button.setEnabled(not running)
        self.browse_python_button.setEnabled(not running)

    def _append(self, text: str):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        if seconds < 60:
            return f"{seconds} sn"
        minutes, seconds = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes} dk"
        hours, minutes = divmod(minutes, 60)
        return f"{hours} sa {minutes} dk"

    def closeEvent(self, event):
        if self._process is not None:
            reply = QMessageBox.question(
                self, "Çalışan işlem var",
                "Eğitim sürüyor. Durdurulup pencere kapatılsın mı?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            self._process.kill()
            self._process.waitForFinished(3000)
        event.accept()
