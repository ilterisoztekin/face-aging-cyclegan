\# CycleGAN v2 ile Yüz Yaşlandırma ve Gençleştirme (Saç ve Kırışıklık Odaklı)



Bu proje, \*\*CycleGAN\*\* mimarisi temel alınarak geliştirilmiş; saç ve sakal rengi değişimleri (ağarma/koyulaşma), detaylı kırışıklık sentezi ve biyometrik kimlik korunumu odaklı bir \*\*Yüz Yaşlandırma ve Gençleştirme (Face Aging \& De-Aging)\*\* sistemini içermektedir.



\---



\## 📌 Projeye Genel Bakış

Görüntüden görüntüye çevirim (Image-to-Image Translation) modellerinde karşılaşılan en büyük problemlerden biri kimlik kaybı (identity loss) ve cilt yüzeyinde oluşan aşırı pürüzsüzleşme (over-smoothing) etkileridir. 



Bu projede \*\*Version 2 CycleGAN\*\* mimarisine entegre edilen özel kayıp fonksiyonları ile bu sorunlar aşılmıştır:

\* \*\*VGG-19 Algısal ve Kimlik Kaybı (Perceptual \& Identity Loss):\*\* Dönüşüm sırasında kişinin temel yüz mimarisini, biyometrik ölçülerini ve karakteristik hatlarını korur.

\* \*\*Kırışıklık ve Doku Odaklı Kayıp (Wrinkle \& Texture Focus Loss):\*\* Yaşlandırma sürecinde alın, göz çevresi ve nazolabial bölgede ince çizgi ile doku detaylarını belirginleştirir.

\* \*\*Saç ve Sakal Özellik Kaybı (Hair \& Beard Feature Loss):\*\* Yaşlandırmada saç/sakal ağarmasını, gençleştirmede ise koyu pigmentasyon dönüşümünü gerçekçi biçimde simüle eder.



\---



\## 🏗️ Model Mimarisi ve Öne Çıkanlar



| Bileşen | Açıklama |

| :--- | :--- |

| \*\*Jeneratör (Generator)\*\* | Yüksek kaliteli dönüşüm için ResNet tabanlı 9 bloklu derin mimari |

| \*\*Ayrıştırıcı (Discriminator)\*\* | İnce taneli yerel gerçekçilik sağlayan PatchGAN mimarisi |

| \*\*Kayıp Fonksiyonları\*\* | Adversarial Loss + Cycle Consistency Loss + VGG Perceptual Loss + Wrinkle Focus Loss |

| \*\*Çözünürlük\*\* | 256x256 (Yerel) / 1024x1024 (Filtreli Çıktı) |



\---



\## 📊 Karşılaştırmalı Model Analizi (Ablation Study)



\* \*\*CycleGAN v1 (Temel Model):\*\* Genel yaş dönüşümünü başardı ancak saç beyazlaması ve ince cilt kırışıklığı detaylarında yetersiz kaldı.

\* \*\*CycleGAN v2 (Önerilen Yöntem - Odaklanmış Mimari):\*\* Saç ve sakal rengi geçişlerinde (ağarma/koyulaşma) belirgin bir performans artışı sağladı; cilt dokusunu pürüzsüzleştirmeden korudu.

\* \*\*Latent-Space Tabanlı Modeller (StyleGAN2 / SAM):\*\* Karşılaştırma amacıyla değerlendirildi; ancak CycleGAN v2 mimarisi, kararlılığı, doğrudan eşleme yeteneği ve eğitilebilir yapısı sebebiyle projenin ana modeli olarak seçildi.



\---



\## 📁 Proje Dizin Yapısı



```text

├── models/

│   ├── generator.py        # ResNet-9 Jeneratör mimarisi

│   ├── discriminator.py    # PatchGAN Ayrıştırıcı

│   └── losses.py           # VGG Kimlik ve Kırışıklık kayıp tanımları

├── scripts/

│   ├── train.py            # CycleGAN v2 eğitim döngüsü

│   └── inference.py        # Yeni görseller üzerinde test betiği

├── pretrained\_models/      # Eğitilmiş model ağırlıkları (Version 2)

├── samples/                # Örnek girdi ve 3'lü karşılaştırma çıktıları

├── README.md               # Proje dokümantasyonu

└── requirements.txt        # Kütüphane bağımlılıkları


##  Proje Dosyaları ve İndirme Bağlantıları

GitHub dosya boyutu kısıtlamaları nedeniyle eğitilmiş model ağırlıkları (`.pth`), eğitim veri kümeleri (`datasets/`) ve tüm kaynak kodlar Google Drive üzerinde paylaşılmıştır.

*  **Google Drive İndirme Bağlantısı:** [Face Aging CycleGAN - Tüm Dosyalar (https://drive.google.com/drive/folders/1gtJ_VHD7tSXKgffHm-hOHaTQmNJbuelE?usp=sharing)]

### 🚀 Kullanım Adımları:

1. Yukarıdaki bağlantıdan proje klasörünü bilgisayarınıza veya Google Colab dizininize indirin.
2. `saved_models/` klasöründe yer alan **`G_Y2O_epoch_99.pth`ve G_O2Y_epoch_99.pth** ağırlık dosyasının yolunu doğrulayın.
3. Preprocess.py -> Train.py -> Inference.py sıralamasında soldan sağa olacak şekilde programı çalıştırın.




Kişisel Yorumlarım



Proje görüntü işlemeye başlangıç için çok daha zor, üst seviye bir tasarım. Deep learning + image processing tabanlı bir proje olmasından kaynaklı, karmaşık bir problem olmasında da dolayı zor bir proje. Ayrıca veri setinin küçük olmasından dolayı 100 epoch verip, google colabdaki 80 gb'lık gpu yu kullandım. Buna rağmen tamamen kusursuz bir model geliştiridiğimi söylemem. Gerçekten yoğun measiler harcadım, özellikle train kısmı gerçekten çok uzun sürdü(2.5-3 saat civarı). Ve de iyileştirmeler yaptığımda mecburen bu süre sürekli tekrar etti. Gerçekten zaman alan bir proje oldu ve de google colab, google drive, git, github vb. arayüzlere hakim olmamı sağladı. 



