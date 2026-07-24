import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# =====================================================================
# 1. OPTİMİZE VERİ YÜKLEYİCİ (Bellek İçi Erişim)
# =====================================================================
class FastWikiAgingDataset(Dataset):
    def __init__(self, young_dir, old_dir, transform=None):
        self.young_dir = young_dir
        self.old_dir = old_dir
        self.transform = transform
        self.young_images = [os.path.join(young_dir, f) for f in os.listdir(young_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.old_images = [os.path.join(old_dir, f) for f in os.listdir(old_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.young_len = len(self.young_images)
        self.old_len = len(self.old_images)

    def __len__(self): 
        return max(self.young_len, self.old_len)

    def __getitem__(self, index):
        young_img = Image.open(self.young_images[index % self.young_len]).convert("RGB")
        old_img = Image.open(self.old_images[index % self.old_len]).convert("RGB")
        if self.transform:
            young_img = self.transform(young_img)
            old_img = self.transform(old_img)
        return young_img, old_img

# =====================================================================
# 2. MODEL MİMARİLERİ
# =====================================================================
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

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        def block(in_f, out_f, norm=True):
            layers = [nn.Conv2d(in_f, out_f, 4, stride=2, padding=1)]
            if norm: layers.append(nn.InstanceNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(*block(in_channels, 64, norm=False), *block(64, 128), *block(128, 256), nn.Conv2d(256, 1, 4, padding=1))
    def forward(self, x): return self.model(x)

# =====================================================================
# 3. YÜKSEK PERFORMANSLI EĞİTİM DÖNGÜSÜ
# =====================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Aktif GPU: {torch.cuda.get_device_name(0)}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    # YEREL SSD YOLLARI
    young_dir = "/tmp/datasets/young"
    old_dir = "/tmp/datasets/old"

    # DRIVE YEDEK YOLLARI
    DRIVE_SAVE_DIR = "/content/drive/MyDrive/face-aging-cyclegan"
    os.makedirs(f"{DRIVE_SAVE_DIR}/saved_models", exist_ok=True)
    os.makedirs(f"{DRIVE_SAVE_DIR}/output", exist_ok=True)

    dataset = FastWikiAgingDataset(young_dir=young_dir, old_dir=old_dir, transform=transform)
    
    
    dataloader = DataLoader(
        dataset, 
        batch_size=16,          
        shuffle=True, 
        num_workers=4,         
        pin_memory=True       
    )

    G_Y2O = Generator().to(device)
    G_O2Y = Generator().to(device)
    D_Y = Discriminator().to(device)
    D_O = Discriminator().to(device)

    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    optimizer_G = torch.optim.Adam(list(G_Y2O.parameters()) + list(G_O2Y.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_Y = torch.optim.Adam(D_Y.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_D_O = torch.optim.Adam(D_O.parameters(), lr=0.0001, betas=(0.5, 0.999))

    epochs = 100
    print("--> Yüksek Hızlı Colab Pro Eğitimi Başlıyor...")

    for epoch in range(epochs):
        for i, (young_imgs, old_imgs) in enumerate(dataloader):
            young_imgs = young_imgs.to(device, non_blocking=True)
            old_imgs = old_imgs.to(device, non_blocking=True)
            
            # --- 1. GENERATORS TRAIN ---
            optimizer_G.zero_grad()
            
            # Agresif Yaşlandırma Katsayıları
            loss_id_Y = criterion_identity(G_O2Y(young_imgs), young_imgs) * 1.0
            loss_id_O = criterion_identity(G_Y2O(old_imgs), old_imgs) * 1.0
            
            fake_old = G_Y2O(young_imgs)
            fake_young = G_O2Y(old_imgs)
            
            loss_GAN_Y2O = criterion_GAN(D_O(fake_old), torch.ones_like(D_O(fake_old)))
            loss_GAN_O2Y = criterion_GAN(D_Y(fake_young), torch.ones_like(D_Y(fake_young)))
            
            rec_young = G_O2Y(fake_old)
            rec_old = G_Y2O(fake_young)
            loss_cycle_Y = criterion_cycle(rec_young, young_imgs) * 2.0
            loss_cycle_O = criterion_cycle(rec_old, old_imgs) * 2.0
            
            loss_G = loss_GAN_Y2O + loss_GAN_O2Y + loss_cycle_Y + loss_cycle_O + loss_id_Y + loss_id_O
            loss_G.backward()
            optimizer_G.step()
            
            # --- 2. DISCRIMINATORS TRAIN ---
            optimizer_D_Y.zero_grad()
            loss_D_Y = (criterion_GAN(D_Y(young_imgs), torch.ones_like(D_Y(young_imgs))) + 
                        criterion_GAN(D_Y(fake_young.detach()), torch.zeros_like(D_Y(fake_young.detach())))) / 2
            loss_D_Y.backward()
            optimizer_D_Y.step()
            
            optimizer_D_O.zero_grad()
            loss_D_O = (criterion_GAN(D_O(old_imgs), torch.ones_like(D_O(old_imgs))) + 
                        criterion_GAN(D_O(fake_old.detach()), torch.zeros_like(D_O(fake_old.detach())))) / 2
            loss_D_O.backward()
            optimizer_D_O.step()
            
            if i % 20 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [G Loss: {loss_G.item():.4f}]")

        # Görsel Çıktısını Drive'a Kaydettim
        with torch.no_grad():
            img_real_save = torch.clamp((young_imgs + 1) / 2.0, 0, 1)
            img_fake_save = torch.clamp((fake_old + 1) / 2.0, 0, 1)
            save_image(img_fake_save, f"{DRIVE_SAVE_DIR}/output/fake_old_epoch_{epoch}.png")

        # Modelleri Sadece Her 5 Epoch'ta Bir Veya Son Epoch'ta Drive'a Kaydettim (Drive Yavaşlatmasın Diye)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            torch.save(G_Y2O.state_dict(), f"{DRIVE_SAVE_DIR}/saved_models/G_Y2O_epoch_{epoch}.pth")
            torch.save(G_O2Y.state_dict(), f"{DRIVE_SAVE_DIR}/saved_models/G_O2Y_epoch_{epoch}.pth")
            print(f"--> [YEDEK] Epoch {epoch} modeli Google Drive'a kaydedildi.")