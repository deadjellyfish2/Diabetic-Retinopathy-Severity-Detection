# Diabetic Retinopathy Severity Classification

Notebook ini berisi pipeline eksperimen deep learning untuk mendeteksi dan mengklasifikasikan tingkat keparahan **Diabetic Retinopathy (DR)** dari citra fundus retina. Dataset yang digunakan adalah dataset Kaggle gabungan **EyePACS, APTOS, APTOS Gaussian Filtered, dan Messidor**.

Pipeline ini melakukan:
- instalasi dependency dan mounting Google Drive;
- download dataset dari Kaggle;
- scanning dataset dan pembuatan metadata;
- analisis distribusi kelas;
- split ulang dataset berbasis **Stratified Group Split 70/15/15**;
- preprocessing dan augmentasi citra;
- training beberapa model deep learning;
- evaluasi model menggunakan metrik klasifikasi dan efisiensi inferensi;
- penyimpanan artifact, grafik, confusion matrix, prediction log, dan summary ke Google Drive.

## Model yang Digunakan

Notebook ini mendukung beberapa model:

1. **MobileNetV3-Large**
2. **EfficientNet-B0**
3. **Swin Transformer Tiny (Swin-T)**
4. **Ensemble EfficientNet-B0 + Swin-T**

Model ensemble menggunakan pendekatan **feature-level fusion**, yaitu fitur dari EfficientNet-B0 dan Swin-T diekstraksi secara paralel, lalu digabung menggunakan concatenation sebelum masuk ke classifier akhir.

## Struktur Pipeline

Secara umum, notebook berjalan dengan urutan berikut:

1. Install dependency:
   - `kaggle`
   - `torch`
   - `torchvision`
   - `timm`
   - `tensorboard`
   - `wandb`
   - `onnx`
   - `onnxruntime`

2. Mount Google Drive untuk menyimpan output eksperimen.

3. Upload file `kaggle.json` untuk autentikasi Kaggle API.

4. Download dataset:
   ```bash
   kaggle datasets download -d ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy
   ```

5. Ekstrak dataset ke local runtime Colab.

6. Scan dataset dan membuat metadata:
   - filepath
   - filename
   - label
   - group_id
   - original split folder

7. Membuat split baru:
   - train: 70%
   - validation: 15%
   - test: 15%

   Split dilakukan berbasis group untuk mengurangi risiko data leakage dari gambar hasil augmentasi.

8. Melakukan transformasi citra:
   - resize
   - horizontal flip
   - vertical flip
   - random rotation
   - color jitter
   - normalization
   - random erasing

9. Melatih model dengan skema dua tahap:
   - **Stage 1: Head Training**
     - backbone dibekukan;
     - hanya classifier/head yang dilatih.
   - **Stage 2: Fine-Tuning**
     - sebagian atau seluruh backbone dibuka kembali;
     - model disesuaikan lebih lanjut terhadap dataset retina.

10. Evaluasi model pada test set:
    - test loss
    - accuracy
    - weighted precision
    - weighted recall
    - weighted F1-score
    - Quadratic Weighted Kappa (QWK)
    - balanced accuracy
    - latency per image
    - FPS

11. Simpan hasil evaluasi dan visualisasi ke Google Drive.

## Struktur Output

Output disimpan di:

```text
/content/drive/MyDrive/DR_Severity-Ensemble/outputs_dr/
```

Setiap model memiliki folder masing-masing, misalnya:

```text
outputs_dr/
├── efficientnet_b0/
├── swin_t/
├── effb0_swin_ensemble/
└── model-comparison-ensemble/
```

Di dalam folder model, output utama meliputi:

```text
artifacts/
├── best_model_stage1.pth
├── best_model.pth
├── best_model_final.pth
└── best_model.onnx

plots/
├── training_history.png
├── loss_curve.png
├── qwk_curve.png
├── learning_rate_curve.png
├── confusion_matrix.png
├── correct_predictions.png
├── wrong_predictions.png
└── confidence_bar_*.png

predictions/
└── test_predictions.csv

config.json
train_summary.json
history.csv
classification_report.csv
confusion_matrix.csv
evaluation_summary.json
```

## Cara Menjalankan Notebook

### 1. Siapkan Kaggle API Token

Unduh file `kaggle.json` dari akun Kaggle:

1. Buka Kaggle.
2. Masuk ke **Account Settings**.
3. Pilih **Create New API Token**.
4. File `kaggle.json` akan terunduh.

Saat notebook dijalankan, upload file tersebut pada cell:

```python
from google.colab import files
files.upload()
```

### 2. Jalankan Notebook di Google Colab

Disarankan menggunakan runtime GPU:

```text
Runtime > Change runtime type > Hardware accelerator > GPU
```

Kemudian jalankan cell dari atas ke bawah.

### 3. Pastikan Dataset Root Benar

Notebook menggunakan dataset root berikut:

```python
DATA_ROOT = "/content/kaggle_data_unzipped/augmented_resized_V2"
```

Jika struktur hasil ekstraksi berbeda, ubah `DATA_ROOT` sesuai folder dataset yang benar.

### 4. Jalankan Training Model

Notebook sudah menyediakan cell training untuk beberapa model.

Contoh training EfficientNet-B0:

```python
run_dir_eff = train_one_model(
    model_name="efficientnet_b0",
    img_size=300,
    batch_size=16,
    num_workers=4,
    epochs_head=5,
    epochs_ft=12,
    lr_head=1e-3,
    lr_ft=2e-4,
    imbalance_mode="weighted_loss",
    use_wandb=False
)
```

Contoh training Swin-T:

```python
run_dir_swin = train_one_model(
    model_name="swin_t",
    img_size=300,
    batch_size=16,
    num_workers=4,
    epochs_head=5,
    epochs_ft=12,
    lr_head=1e-3,
    lr_ft=1e-4,
    imbalance_mode="weighted_loss",
    use_wandb=False
)
```

Contoh training ensemble:

```python
run_dir_ens = train_one_model(
    model_name="effb0_swin_ensemble",
    img_size=300,
    batch_size=16,
    num_workers=4,
    epochs_head=5,
    epochs_ft=12,
    lr_head=1e-3,
    lr_ft=1e-4,
    imbalance_mode="weighted_loss",
    unfreeze_mode="all",
    use_wandb=False
)
```

### 5. Evaluasi Model

Setelah training, model dievaluasi menggunakan:

```python
evaluate_model("efficientnet_b0", run_dir_eff, img_size=256, batch_size=64)
show_qualitative_samples(run_dir_eff, n=8)
plot_prediction_confidence(run_dir_eff, num_samples=5)
```

Untuk model ensemble:

```python
evaluate_model("effb0_swin_ensemble", run_dir_ens, img_size=300, batch_size=32)
show_qualitative_samples(run_dir_ens, n=8)
plot_prediction_confidence(run_dir_ens, num_samples=5)
```

> Catatan: sebaiknya `img_size` saat evaluasi disamakan dengan `img_size` saat training agar konfigurasi eksperimen konsisten.

### 6. Bandingkan Model

Setelah semua model selesai dilatih dan dievaluasi, jalankan:

```python
cmp_df_with_cm = compare_models_with_confusion_matrices()
cmp_df_with_cm
```

Fungsi ini akan menghasilkan:
- tabel perbandingan metrik;
- grafik perbandingan accuracy, F1, QWK, balanced accuracy, dan FPS;
- confusion matrix setiap model;
- deployment ranking.

## Konfigurasi Penting

Beberapa konfigurasi utama yang digunakan:

```python
SEED = 42
NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
```

Training default:

```python
epochs_head = 5
epochs_ft = 10 atau 12
lr_head = 1e-3
lr_ft = 1e-4 atau 2e-4
weight_decay = 1e-4
dropout = 0.35
label_smoothing = 0.05
imbalance_mode = "weighted_loss"
```

## Penanganan Class Imbalance

Notebook menggunakan **weighted loss** untuk mengurangi bias terhadap kelas mayoritas:

```python
criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(DEVICE),
    label_smoothing=label_smoothing
)
```

Bobot kelas dihitung dari distribusi data training. Kelas dengan jumlah sampel lebih sedikit akan mendapatkan bobot loss lebih besar.

## Augmentasi Data

Augmentasi diterapkan hanya pada data training:

```python
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomVerticalFlip(p=0.2)
transforms.RandomRotation(12)
transforms.ColorJitter(...)
transforms.RandomErasing(...)
```

Validation dan test hanya menggunakan resize, tensor conversion, dan normalization.

## Catatan Penting

- Notebook ini ditujukan untuk dijalankan di Google Colab dengan GPU.
- Pastikan file `kaggle.json` sudah diupload sebelum download dataset.
- Output akan disimpan ke Google Drive.
- Training model ensemble membutuhkan memori GPU lebih besar daripada model tunggal.
- Jika mengalami CUDA out-of-memory, kurangi `batch_size` atau `img_size`.
- Untuk eksperimen yang fair, gunakan konfigurasi evaluasi yang konsisten dengan konfigurasi training.

## Ringkasan

Notebook ini membangun pipeline end-to-end untuk klasifikasi tingkat keparahan diabetic retinopathy menggunakan model CNN, Transformer, dan ensemble. Pipeline mencakup data preparation, preprocessing, training dua tahap, evaluasi multi-metrik, visualisasi kesalahan, dan perbandingan performa antar model.
