from google.colab import drive
drive.mount('/content/drive')


import os

# Drive üzerindeki proje yolumu belirttim
DRIVE_PROJECT_PATH = "/content/drive/MyDrive/face-aging-cyclegan"

os.makedirs(f"{DRIVE_PROJECT_PATH}/output", exist_ok=True)
os.makedirs(f"{DRIVE_PROJECT_PATH}/saved_models", exist_ok=True)

print("--> Klasör yapısı hazır!")



from google.colab import drive
import os
import shutil

drive.mount('/content/drive')

# Drive yolum
DRIVE_DATASET = "/content/drive/MyDrive/face-aging-cyclegan/datasets"
LOCAL_DATASET = "/tmp/datasets"

# Veriyi Colab'ın ssd yerel diskine taşıdım burda
if not os.path.exists(LOCAL_DATASET):
    print("--> Veri seti Colab yerel diskine kopyalanıyor (Dosya okuma hızlanacak)...")
    shutil.copytree(DRIVE_DATASET, LOCAL_DATASET)
    print("--> Kopyalama tamamlandı!")
else:
    print("--> Veri seti zaten yerel diskte mevcut.")