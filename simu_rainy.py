import cv2
import numpy as np

def get_noise(img, value=10):
    # 生成噪声图像
    noise = np.random.uniform(0, 256, img.shape[0:2])
    # 控制噪声水平，取浮点数，只保留最大的一部分作为噪声
    v = value * 0.5
    noise[np.where(noise < (256 - v))] = 0
    return noise

def rain_blur(noise, length=10, angle=0, w=1):
    # 将噪声加上运动模糊，模仿雨滴
    trans = cv2.getRotationMatrix2D((length / 10, length / 10), angle - 15, 2 - length / 30.0)
    dig = np.diag(np.ones(length))  # 生成对焦矩阵
    k = cv2.warpAffine(dig, trans, (length, length))  # 生成模糊核
    k = cv2.GaussianBlur(k, (w, w), 0)  # 高斯模糊这个旋转后的对角核，使得雨有宽度
    blurred = cv2.filter2D(noise, -1, k)  # 用刚刚得到的旋转后的核，进行滤波

    #转换到0-255区间
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)

    return blurred

def alpha_rain(rain, img, beta=0.8):
    # 输入雨滴噪声和图像
    # beta = 0.8   #results weight
    rain = np.expand_dims(rain, 2)
    rain_effect = np.concatenate((img, rain), axis=2)  # add alpha channel
    rain_result = img.copy()  # 拷贝一个掩膜
    rain = np.array(rain, dtype=np.float32)  # 数据类型变为浮点数，后面要叠加，防止数组越界要用32位
    rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
    rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    # 对每个通道先保留雨滴噪声图对应的黑色（透明）部分，再叠加白色的雨滴噪声部分（有比例因子）
    return rain_result

# 读取图像
image_path = 'D:\CycleGAN\image\image_at_47.png'
original_image = cv2.imread(image_path)

# 确保图像的模式是RGB
if len(original_image.shape) == 2:
    original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

# 添加噪声
noise = get_noise(original_image)

# 应用雨滴模糊
blurred_noise = rain_blur(noise)

# 应用Alpha通道混合雨滴效果
rain_result = alpha_rain(blurred_noise, original_image)

# 保存处理后的图像
output_path = r'D:\CycleGAN\generate_img\rain1.png'
cv2.imwrite(output_path, rain_result)

# 显示处理后的图像
cv2.imshow('Rainy Image', rain_result)
cv2.waitKey(0)
cv2.destroyAllWindows()