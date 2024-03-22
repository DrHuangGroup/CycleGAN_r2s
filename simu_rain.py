from PIL import Image, ImageFilter
import numpy as np


# 加载图片
def load_image(path):
    image = Image.open(path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


# 添加模仿下雨的噪波效果
def add_rain_noise(image, intensity=0.02, drop_size=1):
    # 获取图片的尺寸
    width, height = image.size

    # 创建一个与图片相同大小的全0数组
    rain_array = np.zeros((height, width, 3), dtype=np.uint8)

    # 生成雨滴
    for _ in range(int(intensity * height * width)):
        y = np.random.randint(0, height)
        x = np.random.randint(0, width)
        # 随机选择一个颜色通道为255，其他为0
        channel = np.random.randint(3)
        rain_array[y][x][channel] = 255

    # 应用模糊滤镜模拟雨水落在水面上的扩散效果
    blurred = Image.fromarray(rain_array).filter(ImageFilter.GaussianBlur(drop_size))

    # 将原图和雨滴图层合并
    noise_image = Image.blend(image, blurred, alpha=0.5)

    return noise_image


# 保存图片
def save_image(image, path):
    image.save(path, 'PNG')


# 主函数
def main():
    image_path = 'D:\CycleGAN\image\image_at_47.png'  # 替换为你的图片路径
    output_path = r'D:\CycleGAN\generate_img\rain1.png'  # 输出图片路径
    image = load_image(image_path)
    rainy_image = add_rain_noise(image)
    save_image(rainy_image, output_path)
    print(f"图片已保存至： {output_path}")


if __name__ == '__main__':
    main()
