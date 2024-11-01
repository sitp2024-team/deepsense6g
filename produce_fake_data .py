
#################################################
# 此代码用于对图片进行处理，生成对应假图片

import os
import shutil
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

# 定义数据增强的变换
data_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1, hue=0.05),  # 颜色变换
    transforms.RandomHorizontalFlip(p=0.2),  # 随机水平翻转
    transforms.ToTensor()
])

# 加载CSV文件
data_csv = './scenario5_dev.csv'
df = pd.read_csv(data_csv)

# 创建新的文件夹（如果有旧文件夹就会删除）
new_dir = './unit1/fake_camera_data'
if os.path.exists(new_dir):
    shutil.rmtree(new_dir)
    print(f"已删除旧的文件夹及其中的所有内容: {new_dir}")
os.makedirs(new_dir, exist_ok=True)

# 打印新建文件夹的完整路径
print(f"新建文件夹的完整路径是: {os.path.abspath(new_dir)}")

# 存储新图像路径和对应关系
new_data = []


for index, row in df.iterrows():
    img_path = row['unit1_rgb_1']  # 获取原图像路径

    # 确保图像存在
    if not os.path.exists(img_path):
        print(f"警告: 图像文件不存在 - {img_path}")
        continue

    # 打开图像
    image = Image.open(img_path).convert('RGB')

    # 应用数据增强
    transformed_image = data_transforms(image)

    # 生成新的图像文件名
    new_img_name = os.path.basename(img_path)  # 获取文件名
    new_img_path = os.path.join(new_dir, new_img_name)  # 新路径
    print(new_img_name)

    # 保存新图像
    transformed_image = transforms.ToPILImage()(transformed_image)  # 将Tensor转换回图像
    transformed_image.save(new_img_path)

    # 记录新图像路径和其他信息
    new_data.append({
        'index': len(df) + len(new_data)+1,
        'unit2_loc_1': row['unit2_loc_1'],
        'unit1_rgb_1': new_img_path,  # 更新为新图像路径
        'unit1_pwr_1': row['unit1_pwr_1'],
        'beam_index_1': row['beam_index_1']
    })

# 创建新的 DataFrame
new_df = pd.DataFrame(new_data)

# 将原始数据和新数据合并
combined_df = pd.concat([df, new_df], ignore_index=True)

# 保存新的CSV文件，to_csv 方法默认会覆盖原有文件
new_csv_path = './scenario5_dev_include_fake.csv'  # 修改为包含新文件名的路径
combined_df.to_csv(new_csv_path, index=False)

print(f"处理完成，新的CSV文件已保存为：{new_csv_path}")
