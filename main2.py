# 用晴天图像生成雨天图像
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from PIL import Image
import tqdm
import glob

sunny_path = glob.glob('data/train/Sunny/*.jpg')  # 获取数据集中的.jpg图片
rain_path = glob.glob('data/train/Rain/*.jpg')  # 获取数据集中的.jpg图片
# print(sunny_path[:3])
# print(rain_path[:3])
sunny_path_test = glob.glob('data/val/Sunny/*.jpg')  # 获取数据集中的.jpg图片
rain_path_test = glob.glob('data/val/Rain/*.jpg')  # 获取数据集中的.jpg图片

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256, 256)),
                                transforms.Normalize(mean=0.5, std=0.5)])  # Normalize为转化到-1~1之间


# 定义数据读取
class SGANDataset(Dataset):
    def __init__(self, imgs_path):  # 初始化
        super(SGANDataset, self).__init__()
        self.imgs_path = imgs_path  # 定义属性

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):  # 对数据切片
        img_path = self.imgs_path[index]

        # 从文件中读取图像
        pil_img = Image.open(img_path)
        pil_img = transform(pil_img)
        return pil_img


# 初始化训练集
sunny_dataset = SGANDataset(sunny_path)  # 创建dataset
rain_dataset = SGANDataset(rain_path)  # 创建dataset

# 初始化测试集
sunny_dataset_test = SGANDataset(sunny_path_test)  # 创建dataset
rain_dataset_test = SGANDataset(rain_path_test)  # 创建dataset

sunny_dataloader = torch.utils.data.DataLoader(sunny_dataset, batch_size=4, shuffle=True)
rain_dataloader = torch.utils.data.DataLoader(rain_dataset, batch_size=4, shuffle=True)

sunny_dataloader_test = torch.utils.data.DataLoader(sunny_dataset_test, batch_size=4)
rain_dataloader_test = torch.utils.data.DataLoader(rain_dataset_test, batch_size=4)


sunny_bath = next(iter(sunny_dataloader)) #查看
rain_bath = next(iter(rain_dataloader)) #查看
print(sunny_bath.shape) #torch.Size([4, 3, 256, 256])
print(rain_bath.shape) #torch.Size([4, 3, 256, 256])

# 查看数据集
plt.figure(figsize=(8, 12))
for i, (sunny, rain) in enumerate(zip(sunny_bath[:3], rain_bath[:3])): #zip代表元组
    # 因为dataset返回的数据是tensor，需要转为numpy格式，因为Normalize为转化到-1~1之间，所以加1再除以2将其转化到0~1之间
    sunny = (sunny.permute(1, 2, 0).numpy() + 1) / 2
    rain = (rain.permute(1, 2, 0).numpy() + 1) / 2
    plt.subplot(3, 2, 2*i+1)
    plt.title('sunny')
    plt.imshow(sunny)
    plt.subplot(3, 2, 2*i+2)
    plt.title('rain')
    plt.imshow(rain)
plt.show()


# 定义下采样模块
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.conv_leak = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            # nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(inplace=True)
        )
        self.bn = nn.InstanceNorm2d(out_channels)

    def forward(self, x, is_bn=True):  # is_bn用于确定是否使用bn层，默认为True
        x = self.conv_relu(x)
        if is_bn:
            x = self.bn(x)
        return x


# 定义上采样模块
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upconv_relu = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(inplace=True)
        )
        self.bn = nn.InstanceNorm2d(out_channels)

    def forward(self, x, is_drop=False):  # is_drop用于确定是否使用drop层，默认为False
        x = self.upconv_relu(x)
        x = self.bn(x)
        if is_drop:
            x = F.dropout2d(x)
        return x


# 定义生成器，包含6个下采样层，6个上采样层
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down1 = Downsample(3, 64)  # 3,256,256 -- 64,128,128
        self.down2 = Downsample(64, 128)  # 64,128,128 -- 128,64,64
        self.down3 = Downsample(128, 256)  # 128,64,64 -- 256,32,32
        self.down4 = Downsample(256, 512)  # 256,32,32 -- 512,16,16
        self.down5 = Downsample(512, 1024)  # 512,16,16 -- 1024,8,8
        self.down6 = Downsample(1024, 1024)  # 1024,8,8 -- 1024,4,4

        self.up1 = Upsample(1024, 1024)  # 1024,4,4 -- 1024,8,8
        self.up2 = Upsample(2048, 512)  # 2024,8,8 -- 512,16,16
        self.up3 = Upsample(1024, 256)  # 1024,16,16 -- 512,32,32
        self.up4 = Upsample(512, 128)  # 512,32,32 -- 128,64,64
        self.up5 = Upsample(256, 64)  # 512,64,64 -- 64,128,128
        # 128,128,128 -- 3,256,256
        self.last = nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)

        x6 = self.up1(x6, is_drop=True)#1024--1024
        x6 = torch.cat([x6, x5], dim=1)#1024+1024=2048

        x6 = self.up2(x6, is_drop=True)#2048--512
        x6 = torch.cat([x6, x4], dim=1)#512+512=1024

        x6 = self.up3(x6, is_drop=True)#1024--256
        x6 = torch.cat([x6, x3], dim=1)#256+256=512

        x6 = self.up4(x6)#512--128
        x6 = torch.cat([x6, x2], dim=1)#128+128=256

        x6 = self.up5(x6)#256--64
        x6 = torch.cat([x6, x1], dim=1)#64+64=128

        x6 = torch.tanh(self.last(x6))

        return x6


# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.down1 = Downsample(3, 64)
        self.down2 = Downsample(64, 128)
        self.last = nn.Conv2d(128, 1, 3)

    def forward(self, img):
        x = self.down1(img)
        x = self.down2(x)
        x = torch.sigmoid(self.last(x))
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化两个生成器
gen_AB = Generator().to(device)
gen_BA = Generator().to(device)

# 初始化两个判别器
dis_A = Discriminator().to(device)
dis_B = Discriminator().to(device)

# 损失函数  1.gan loss  2.cycle consistance loss  3.identity loss
bce_loss = torch.nn.BCELoss()
l1_loss = torch.nn.L1Loss()

# 初始化优化器
# 对两个生成器同时进行优化, 使用itertools.chain对二者同时进行迭代
gen_optimizer = torch.optim.Adam(itertools.chain(gen_AB.parameters(), gen_BA.parameters()), lr=2e-4, betas=(0.5, 0.999))

# 对两个判别器分别进行优化
dis_A_optimizer = torch.optim.Adam(dis_A.parameters(), lr=2e-4, betas=(0.5, 0.999))
dis_B_optimizer = torch.optim.Adam(dis_B.parameters(), lr=2e-4, betas=(0.5, 0.999))


# 绘图函数，将每一个epoch中生成器生成的图片绘制
def gen_img_plot(model, epoch, test_input):  # model为gen_AB/gen_BA，test_input
    generate = model(test_input).permute(0, 2, 3, 1).cpu().numpy()  # 将通道维度放在最后
    test_input = test_input.permute(0, 2, 3, 1).cpu().numpy()  # 1,3,256,256 -- 1,256,256,3
    plt.figure(figsize=(10, 6))
    display_list = [test_input[0], generate[0]]
    title = ['Input image', 'Generate image']
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        plt.imshow((display_list[i] + 1) / 2)  # 从-1~1 --> 0~1
        plt.axis('off')
    plt.savefig('./image/image_at_{}.png'.format(epoch))


test_batch = next(iter(sunny_dataloader_test))  # batch_size,3,256,256
# 测试输入:选取test_batch中的第一张图片，并添加一个batch_size维度  3,256,256--1,3,256,256
test_input = torch.unsqueeze(test_batch[0], 0).to(device)

# cycleGAN训练
D_loss = []
G_loss = []
epochs = 50
for epoch in range(epochs):
    d_epoch_loss = 0
    g_epoch_loss = 0
    for step, (real_A, real_B) in enumerate(zip(sunny_dataloader, rain_dataloader)):  # 取出真实的雨天，晴天图片
        real_A = real_A.to(device)
        real_B = real_B.to(device)
        # print(real_A.shape)
        # print(real_B.shape)
        # --------------------begin--------------------#
        # 生成器训练
        gen_optimizer.zero_grad()  # 训练之前梯度清0
        # identity loss
        same_B = gen_AB(real_B)  # 真实的B经过生成器gen_AB还是要得到真实的B
        identity_B_loss = l1_loss(same_B, real_B)
        same_A = gen_AB(real_A)  # 真实的A经过生成器gen_BA还是要得到真实的A
        identity_A_loss = l1_loss(same_A, real_A)
        # 对抗损失 gan loss
        fake_B = gen_AB(real_A)  # 真实A通过生成器生成了B，此时生成器希望判别器将其判别为真
        D_pred_fake_B = dis_B(fake_B)
        gen_loss_AB = bce_loss(D_pred_fake_B, torch.ones_like(D_pred_fake_B, device=device))
        fake_A = gen_BA(real_B)  # 真实B通过生成器生成了A，此时生成器希望判别器将其判别为真
        D_pred_fake_A = dis_A(fake_A)
        gen_loss_BA = bce_loss(D_pred_fake_A, torch.ones_like(D_pred_fake_A, device=device))
        # 循环一致损失
        recovered_A = gen_BA(fake_B)
        cycle_loss_ABA = l1_loss(recovered_A, real_A)

        recovered_B = gen_AB(fake_A)
        cycle_loss_BAB = l1_loss(recovered_B, real_B)

        # 生成器总的损失
        g_loss = identity_A_loss + identity_B_loss + gen_loss_AB + gen_loss_BA + cycle_loss_ABA + cycle_loss_BAB

        g_loss.backward()
        gen_optimizer.step()
        # --------------------end--------------------#

        # --------------------begin--------------------#
        # 判别器训练
        # dis_A训练
        dis_A_optimizer.zero_grad()
        dis_A_real_output = dis_A(real_A)  # 输入为真，期望判定为真
        dis_A_real_loss = bce_loss(dis_A_real_output, torch.ones_like(dis_A_real_output, device=device))

        dis_A_fake_output = dis_A(fake_A.detach())  # 输入为假，期望判定为假，梯度截断
        dis_A_fake_loss = bce_loss(dis_A_fake_output, torch.zeros_like(dis_A_fake_output, device=device))

        dis_A_loss = dis_A_real_loss + dis_A_fake_loss  # 生成器A的总损失
        dis_A_loss.backward()
        dis_A_optimizer.step()

        # dis_B训练
        dis_B_optimizer.zero_grad()
        dis_B_real_output = dis_B(real_B)  # 输入为真，期望判定为真
        dis_B_real_loss = bce_loss(dis_B_real_output, torch.ones_like(dis_B_real_output, device=device))

        dis_B_fake_output = dis_B(fake_B.detach())  # 输入为假，期望判定为假，梯度截断
        dis_B_fake_loss = bce_loss(dis_B_fake_output, torch.zeros_like(dis_B_fake_output, device=device))

        dis_B_loss = dis_B_real_loss + dis_B_fake_loss  # 生成器B的总损失
        dis_B_loss.backward()
        dis_B_optimizer.step()
        # --------------------end--------------------#

        with torch.no_grad():
            g_epoch_loss += g_loss.item()  # 将每一个批次的loss累加
            d_epoch_loss += (dis_A_loss + dis_B_loss).item()  # 将每一个批次的loss累加

    with torch.no_grad():
        g_epoch_loss /= (step + 1)  # 求得每一轮的平均loss
        d_epoch_loss /= (step + 1)  # 求得每一轮的平均loss
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        print('epoch:', epoch, 'g_epoch_loss:', g_epoch_loss, 'd_epoch_loss:', d_epoch_loss)
        gen_img_plot(gen_AB, epoch, test_input)
