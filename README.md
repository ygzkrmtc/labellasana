# YOLOv8 Etiketleme İstasyonu

Bir klasördeki resimleri sınırlayıcı kutularla etiketleyip YOLOv8 formatında
`.txt` dosyaları üreten; veri setini eğitime hazırlayan, kalitesini denetleyen ve
isteğe bağlı olarak model destekli ön-etiketleme yapan masaüstü uygulaması.

```
yolo_labeler/
├── main.py                    # giriş noktası
├── app/
│   ├── models.py              # sınıf yönetimi, renk paleti            (Qt'siz)
│   ├── yolo_io.py             # YOLO normalizasyon/G-Ç, yedekleme      (Qt'siz)
│   ├── dataset.py             # bölme, istatistik, dışa aktarma        (Qt'siz)
│   ├── audit.py               # kalite denetimi kuralları              (Qt'siz)
│   ├── converters.py          # COCO / Pascal VOC dönüştürme           (Qt'siz)
│   ├── imaging.py             # CLAHE, kenar, manyetik kutu            (Qt'siz)
│   ├── inference.py           # ONNX YOLOv8 çıkarımı, NMS              (Qt'siz)
│   ├── project.py             # .labelproj proje dosyası               (Qt'siz)
│   ├── review.py              # model etiketleri onay kuyruğu          (Qt'siz)
│   ├── theme.py               # karanlık / açık tema
│   ├── canvas.py              # tuval, kutu öğesi, zoom/pan, büyüteç
│   ├── dialogs.py             # sınıf/kısayol/uyarı bileşenleri
│   ├── dataset_dialogs.py     # dışa aktarma sihirbazı, istatistik paneli
│   ├── audit_dialog.py        # kalite denetimi paneli
│   ├── ai_dialogs.py          # tahmin ayarları
│   └── main_window.py         # menüler, paneller, iş akışları
└── tests/                     # 157 test (107 mantık + 50 arayüz)
```

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
python main.py
```

Zorunlu olan tek paket **PyQt5**'tir. `numpy` + `opencv-python-headless` görüntü
işleme özelliklerini, `onnxruntime` model destekli ön-etiketlemeyi açar. Kurulu
değillerse ilgili menüler pasif görünür, program normal çalışır.

## Uyumluluk

| | Desteklenen | Not |
|---|---|---|
| Python | **3.10 – 3.12** (test edilen) · 3.8+ çalışır | 3.13/3.14'te bazı isteğe bağlı paketlerin hazır sürümü gecikebilir |
| İşletim sistemi | Windows, Linux (CI'da test edilir) · macOS beklenir | |
| Zorunlu paket | Yalnızca **PyQt5** | Diğerleri yoksa ilgili menü açıklama verir, program çalışır |

Alınan önlemler:

- `requirements.txt` sürüm aralıkları **üst sınırlıdır** — bir üst ana sürüm
  (numpy 3, PyQt 6 gibi) ikili uyumluluğu bozarsa kurulum onu çekmez.
- İsteğe bağlı paketler **yumuşak iner**: yoksa program açılır, ilgili menü
  neyin eksik olduğunu ve tam kurulum komutunu söyler.
- `onnxruntime`, Windows'ta Qt'den sonra yüklenince DLL hatası verebiliyor;
  `main.py` yükleme sırasını tersine çevirerek bunu önler.
- GitHub Actions her push'ta **Windows + Linux × Python 3.10/3.11/3.12** ve
  ayrıca "sadece PyQt5" kombinasyonunda tüm testleri çalıştırır.
- Kurulum sonrası tek komutla ortam denetimi:

```bash
python check_env.py
```

Bu betik yalnızca standart kütüphaneyi kullanır (hiçbir şey kurulmamışken de
çalışır), her eksik için çözümü yazar ve `.exe` dağıtımında karşılaşılan
üç gerçek sorunu (DLL yükleme sırası, eksik VC++ çalışma zamanı, `opencv-python`
tam sürümünün Qt çakışması) ayrıca denetler.

> **Uyarı:** `opencv-python` yerine **`opencv-python-headless`** kurun. Tam sürüm
> kendi Qt eklentilerini taşır ve PyQt5 ile çakışıp arayüzü bozabilir.
> `check_env.py` bunu tespit eder.

Dağıtımda en güvenli yol: kullanıcıya `.exe` vermek. Exe'nin içinde Python ve
tüm paketler gömülüdür; hedef makinede hiçbir kurulum gerekmez.

## Testler

```bash
python -m tests.run_all
```

Arayüz testleri Qt'nin `offscreen` sürücüsüyle çalışır; ekran gerekmez, pencere
açılmaz. Beklenen çıktı: **157 test, 0 hata**.

Testler özellikle şunları kilitler: sınıf silindiğinde sahipsiz kutu kalmaması,
silinen sınıfın geri gelmemesi, kaydet/oku turunda koordinatların korunması,
kilitli kutuların silinmemesi, görüntü iyileştirmenin etiketleri kaydırmaması.

## Çalışma akışı

1. `Ctrl+O` ile resim klasörünü açın.
2. `W` + fareyle sürükleyerek ilk kutuyu çizin — sınıf listesi boşsa **ad sorulur**.
3. Çizim modu açık kalır; arka arkaya kutu çizersiniz. Yeni kutulara aktif sınıf atanır.
4. `D` ile kaydedip sonraki resme geçin.
5. Bitince `Ctrl+E` ile veri setini dışa aktarın, `F5` ile kalitesini denetleyin.

Etiketler varsayılan olarak resim klasörünün altındaki `labels/` dizinine yazılır:

```
resimler/
├── 001.jpg
└── labels/
    ├── 001.txt
    ├── classes.txt
    ├── .backup/          # her kayıttan önceki 5 sürüm
    └── .review.json      # model etiketlerinin onay kuyruğu
```

## Kısayollar

| Kısayol | İşlev |
|---|---|
| `W` | Çizim modu aç/kapat |
| `E` | Yeni sınıf ekle |
| `C` | Seçili kutunun sınıfını değiştir |
| `1…9` | Sınıf seç (kutu seçiliyken ona atar) |
| Çift tık / sağ tık | Kutunun sınıfını değiştir |
| `Delete` / `Backspace` | Seçili kutuyu sil |
| `Ctrl+Shift+Delete` | Bu resimdeki tüm kutuları sil |
| `Ctrl+Z` | Geri al |
| `Ctrl+A` / `Ctrl`+tık / `Shift`+sürükle | Çoklu seçim |
| Ok tuşları / `Shift`+ok / `Alt`+ok | 1 px kaydır / 10 px / boyutlandır |
| `Ctrl+L` | Kutuyu kilitle |
| `S` / `M` | Kenarlara oturt / manyetik mod |
| `Ctrl+C` / `Ctrl+V` | Kutuları kopyala / yapıştır |
| `D` / `A` / `Ctrl+→` | Sonraki / önceki / sonraki etiketlenmemiş |
| `Ctrl+S` / `Ctrl+O` | Kaydet / klasör aç |
| Tekerlek · orta tuş veya `Alt`+sürükle | Zoom · pan |
| `Ctrl+0` / `Ctrl+1` | Ekrana sığdır / %100 |
| `H` / `G` / `I` / `L` | Kutuları gizle / kenar katmanı / iyileştirme şeridi / büyüteç |
| `Ctrl+I` / `Ctrl+E` / `F5` | İstatistikler / dışa aktar / kalite denetimi |
| `Ctrl+R` / `Enter` / `Esc` | Tahmin et / önerileri kabul et / iptal |
| `F1` | Kısayol rehberi |

## YOLO formatı

```
<class_id> <x_center> <y_center> <width> <height>
```

Tüm değerler resim boyutuna bölünerek 0-1 aralığına normalize edilir. Resmi
tekrar açtığınızda `.txt` okunur, piksel boyutlarıyla çarpılıp kutular kendi
sınıf renkleriyle aynı yerlerine çizilir. Kutu kalmayınca boş `.txt` yazılır —
YOLO'da bu "arka plan örneği olarak gözden geçirildi" demektir.

**Sınıf kimliği = `classes.txt` satır numarasıdır.** Bir sınıf silindiğinde
program, diskteki tüm etiket dosyalarını otomatik olarak yeniden numaralandırır;
veri seti hiçbir adımda tutarsız kalmaz.

## Veri setini eğitime hazırlama

`Araçlar → Veri Setini Dışa Aktar` (`Ctrl+E`):

```
resimler_dataset/
├── data.yaml
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

Bölme **katmanlıdır** (sınıf bileşimleri korunur) ve **tohumu sabittir** (aynı
tohum aynı bölmeyi verir). Sonra:

```bash
yolo detect train data="...\data.yaml" model=yolov8n.pt epochs=100 imgsz=640
```

`Ctrl+I` istatistikleri (sınıf dağılımı, nesne büyüklüğü, konum ısı haritası),
`F5` kalite denetimini (çok küçük kutu, yinelenen kutu, kenara yapışık kutu,
aşırı en-boy oranı, yetim `.txt`, sınıf dengesizliği) gösterir. Denetim
sonucundaki bir satıra çift tıklamak ilgili resmi açıp o kutuyu seçer.

COCO JSON ve Pascal VOC XML içe/dışa aktarma da `Araçlar` menüsündedir.

## Model eğitimi ve model kütüphanesi

Uygulama modeli **kendisi eğitebilir**. Eğitim, `ultralytics` + `torch` gerektirdiği
için uygulamanın içinde değil, proje klasörünün yanına kurulan **ayrı bir Python
ortamında alt süreç olarak** çalışır. Böylece `.exe` küçük kalır.

1. `Yapay Zekâ → Model Eğit` → ilk kez **Eğitim Ortamını Kur** (~2 GB, bir kez).
2. Temel model (n/s/m), epoch, görüntü boyutu, donanım seçin — varsayılanlar veri
   seti büyüklüğüne göre otomatik ayarlanır.
3. **Eğitimi Başlat**: mevcut etiketlerinizden `data.yaml` üretilir (%80/%20
   katmanlı bölme), eğitim arka planda çalışır, ilerleme ve mAP canlı görünür,
   istediğiniz an durdurabilirsiniz.
4. Biten model otomatik ONNX'e çevrilip `models/aku_v1.onnx` olarak kaydedilir ve
   yüklenir. Yanına `aku_v1.json` yazılır: tarih, sınıflar, mAP, süre.

Sonraki seferlerde **Yapay Zekâ → Modeller** menüsünden istediğiniz sürümü seçip
`Ctrl+R` ile tahmin alırsınız. Yeni etiketler biriktikçe tekrar eğitirsiniz
(`aku_v2`, `aku_v3`…) ve model her turda iyileşir.

> **Gerçekçi beklenti:** 12 resimle işe yarar model çıkmaz. Elle ~100-200 resim
> etiketleyin, ilk modeli eğitin, sonrasında model kalanların çoğunu sizin yerinize
> çizsin — siz sadece düzeltin. Bu döngü etiketleme süresini tipik olarak
> birkaç kat kısaltır.

## Hazır bir modeli elle yükleme

1. `yolo export model=best.pt format=onnx`
2. `Yapay Zekâ → ONNX Modeli Yükle`
3. `Ctrl+R` → kapsam ve güven eşiği seçin.

**Açık resimde** sonuçlar kesik çizgili **öneri** olarak görünür; `Enter` ile
kabul, `Esc` ile silinir — onaylanmadan diske yazılmaz. **Toplu tahminde**
sonuçlar diske yazılır ama o resimler listede `?` ile işaretlenir; açıp
onaylayana kadar gözden geçirilmemiş sayılırlar.

`ultralytics`/`torch` gerekmez ve paketlenmez: bunlar `.exe`'ye ~2 GB eklerdi,
`onnxruntime` ~50 MB'dır.

## Görüntü işleme yardımcıları

- **`I` — iyileştirme şeridi:** parlaklık, kontrast, gama ve CLAHE. Yalnızca
  ekranı etkiler; dosya da etiket koordinatları da değişmez.
- **`G` — kenar katmanı:** düşük kontrastta nesne sınırını görünür kılar.
- **`M` / `S` — manyetik kutu:** kaba çizilen kutuyu en güçlü gradyan çizgisine oturtur.
- **`L` — büyüteç:** imlecin altındaki bölgeyi 4× gösterir.

---

# .EXE olarak paketleme (PyInstaller)

### 1. Kurulum

```bash
pip install pyinstaller
```

Uygulamanın çalıştığı **aynı sanal ortamda** olmalı. `.exe` yalnızca üretildiği
işletim sistemi için çalışır.

### 2. Derleme

En kolayı, sanal ortam aktifken depodaki betiği çalıştırmak:

```bat
build_exe.bat
```

Elle yapmak isterseniz:

```bash
pyinstaller --onefile --windowed --name "YOLO_Etiketleyici" --clean --noconfirm ^
    --exclude-module torch --exclude-module ultralytics main.py
```

`--exclude-module torch --exclude-module ultralytics` önemli: eğitim ortamı ayrı
kurulduğu için bunlar exe'ye girmemeli, girerse dosya ~2 GB olur.

| Parametre | Ne yapar |
|---|---|
| `--onefile` | Her şeyi tek `.exe` içine gömer |
| `--windowed` | Arka planda siyah konsol açılmaz (`--noconsole` ile aynı) |
| `--name` | Üretilecek exe'nin adı |
| `--icon=icon.ico` | Uygulama ikonu (yoksa bu parametreyi atın) |
| `--clean` / `--noconfirm` | Önbelleği temizler / sormadan üzerine yazar |

Çıktı: **`dist/YOLO_Etiketleyici.exe`** — Python kurulu olmayan Windows'a
kopyalayıp çift tıklayabilirsiniz.

### 3. Boyut ve isteğe bağlı paketler

| Kurulum | Yaklaşık `.exe` boyutu |
|---|---|
| Sadece PyQt5 | ~45-60 MB |
| + numpy + OpenCV | ~150 MB |
| + onnxruntime | ~200 MB |

Yapay zekâ özelliklerini dağıtmak istemiyorsanız `onnxruntime`'ı kurmadan
derleyin: menü pasif görünür, program çalışır.

### 4. Sık karşılaşılan sorunlar

**Pencere açılmıyor:** hatayı görmek için geçici olarak konsollu derleyin ve
komut isteminden çalıştırın:

```bash
pyinstaller --onefile --name "YOLO_debug" main.py
dist\YOLO_debug.exe
```

**"no Qt platform plugin could be initialized":** PyInstaller'ı güncelleyin
(`pip install -U pyinstaller`), gerekirse `--collect-all PyQt5` ekleyin.

**OpenCV/onnxruntime bulunamadı hatası:** `--collect-all cv2` veya
`--collect-all onnxruntime` ekleyin.

**Antivirüs uyarısı / yavaş açılış:** `--onefile` yerine `--onedir` ile derleyin
(bu durumda `dist/YOLO_Etiketleyici/` klasörünün tamamını dağıtın).
