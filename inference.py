import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# 1. MODEL MİMARİLERİ AşağıDA
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3), nn.InstanceNorm2d(channels), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3), nn.InstanceNorm2d(channels)
        )
    def forward(self, x): return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_residual_blocks=6):
        super(Generator, self).__init__()
        model = [nn.ReflectionPad2d(3), nn.Conv2d(in_channels, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(inplace=True)]
        in_features = 64
        for _ in range(2):
            out_features = in_features * 2
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
            in_features = out_features
        for _ in range(num_residual_blocks): model += [ResidualBlock(in_features)]
        for _ in range(2):
            out_features = in_features // 2
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
            in_features = out_features
        model += [nn.ReflectionPad2d(3), nn.Conv2d(in_features, out_channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, x): return self.model(x)

# 2. TEST FONKSİYONU
def run_colab_inference(image_path, y2o_path, o2y_path, output_dir="/content/drive/MyDrive/face-aging-cyclegan/test_results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Aktif Donanım: {device}")
    
    G_Y2O = Generator().to(device)
    G_O2Y = Generator().to(device)
    
    G_Y2O.load_state_dict(torch.load(y2o_path, map_location=device))
    G_O2Y.load_state_dict(torch.load(o2y_path, map_location=device))
    G_Y2O.eval()
    G_O2Y.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        fake_old = G_Y2O(input_tensor)    # Gençten Yaşlıya -> (Aging)
        fake_young = G_O2Y(input_tensor)  # Yaşlıdan Gence -> (De-aging)

        input_tensor_save = torch.clamp((input_tensor + 1) / 2.0, 0, 1)
        fake_old_save = torch.clamp((fake_old + 1) / 2.0, 0, 1)
        fake_young_save = torch.clamp((fake_young + 1) / 2.0, 0, 1)

    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Görselleri Tek Tek KaydetTİM
    save_image(input_tensor_save, os.path.join(output_dir, "original.png"))
    save_image(fake_old_save, os.path.join(output_dir, "aged_result.png"))
    save_image(fake_young_save, os.path.join(output_dir, "de_aged_result.png"))
    
    # 2. Yan Yana Karşılaştırma Görseli OluşturDum
    comparison = torch.cat((input_tensor_save, fake_old_save, fake_young_save), dim=3)
    save_image(comparison, os.path.join(output_dir, "comparison_all.png"))

    print("\n--- İŞLEM BAŞARIYLA TAMAMLANDI ---")
    print(f"Orijinal Görsel : {os.path.join(output_dir, 'original.png')}")
    print(f"Yaşlandırılmış  : {os.path.join(output_dir, 'aged_result.png')}")
    print(f"Gençleştirilmiş : {os.path.join(output_dir, 'de_aged_result.png')}")
    print(f"Yan Yana Özet   : {os.path.join(output_dir, 'comparison_all.png')}")

# 3. RUN TEST (Old Klasöründeki Resim İle)
DRIVE_DIR = "/content/drive/MyDrive/face-aging-cyclegan"

# Resmin 'old' klasöründeki tam yolu:
test_img = "/tmp/datasets/old/8718_1945-03-01_2011.jpg"

run_colab_inference(
    image_path=test_img,
    y2o_path=f"{DRIVE_DIR}/saved_models/G_Y2O_epoch_99.pth",
    o2y_path=f"{DRIVE_DIR}/saved_models/G_O2Y_epoch_99.pth"
)
