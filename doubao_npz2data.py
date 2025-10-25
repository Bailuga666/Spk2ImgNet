import os
import numpy as np
import cv2
from functools import partial

# 确保输出目录存在
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# 将Spike矩阵（0/1）转换为.dat文件（1像素1位存储）
def spike_to_dat(spike_matrix, dat_path):
    """
    spike_matrix: 形状为 (T, H, W)，值为0或1的uint8数组
    dat_path: 输出.dat文件路径
    """
    T, H, W = spike_matrix.shape
    # 每帧图像需要的字节数：(H*W) / 8（1像素1位）
    bytes_per_frame = (H * W) // 8
    # 初始化字节数组（总长度：帧数 × 每帧字节数）
    dat_data = bytearray(T * bytes_per_frame)
    
    for t in range(T):
        frame = spike_matrix[t]  # 当前帧（H, W）
        # 按行遍历像素，每8个像素打包成1字节
        for i in range(H):
            for j in range(W):
                pixel_idx = i * W + j  # 像素在帧内的平坦索引
                byte_pos = t * bytes_per_frame + (pixel_idx // 8)  # 字节位置
                bit_pos = pixel_idx % 8  # 字节内的位位置（0-7）
                # 若像素值为1，则设置对应位为1
                if frame[i, j] == 1:
                    dat_data[byte_pos] |= (1 << bit_pos)
    
    # 写入.dat文件
    with open(dat_path, 'wb') as f:
        f.write(dat_data)
    print(f"已保存Spike数据到: {dat_path}")

# 主函数：提取并保存数据
def extract_and_save(npz_path, output_root):
    # 1. 定义输入输出路径
    input_dir = os.path.join(output_root, "input")  # .dat文件目录
    gt_dir = os.path.join(output_root, "gt")        # .png文件目录
    ensure_dir(input_dir)
    ensure_dir(gt_dir)
    
    # 2. 读取npz文件
    print(f"读取npz文件: {npz_path}")
    data = np.load(npz_path)
    left_spike = data["left_spk"]  # 形状：(800, 256, 256)，uint8（0或1）
    vid_left = data["vid_left"]      # 形状：(8, 256, 256)，float32（假设已在0~1范围）
    data.close()
    
    # 3. 保存Spike数据为.dat文件
    # 文件名：group_0011.dat（与原代码的input文件命名一致）
    dat_basename = os.path.splitext(os.path.basename(npz_path))[0] + ".dat"
    dat_path = os.path.join(input_dir, dat_basename)
    spike_to_dat(left_spike, dat_path)
    
    # 4. 保存GT图像为.png文件
    # GT时间点：80, 160, 240, 320, 400, 480, 560, 640（共8个，对应vid_left的8帧）
    gt_timestamps = [80, 160, 240, 320, 400, 480, 560, 640]
    assert len(gt_timestamps) == vid_left.shape[0], "GT时间点数量与vid_left帧数不匹配"
    
    # 原代码中GT文件名格式：{dat文件名前缀[:-6]} + {num} + .png
    # 这里dat文件名为group_0011.dat，前缀[:-6]是"group_00"（因为".dat"是4个字符，[:-6]即去掉"11.dat"）
    # 注：若文件名长度不同，可能需要微调[:-6]的索引，确保截取正确前缀
    dat_prefix = os.path.splitext(dat_basename)[0]  # "group_0011"
    gt_filename_prefix = dat_prefix[:-6]  # 截取前缀（根据原代码逻辑）
    
    for idx in range(vid_left.shape[0]):
        gt_img = vid_left[idx]  # 第idx帧GT（256,256）
        # 若GT是float32（0~1），转换为uint8（0~255）
        if gt_img.dtype == np.float32:
            gt_img = (gt_img * 255).astype(np.uint8)
        # 确保是单通道灰度图
        if gt_img.ndim == 3 and gt_img.shape[2] == 3:
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2GRAY)
        
        # 生成GT文件名：{前缀}{时间点}.png
        timestamp = gt_timestamps[idx]
        gt_filename = f"{gt_filename_prefix}{timestamp}.png"
        gt_path = os.path.join(gt_dir, gt_filename)
        
        # 保存图像
        cv2.imwrite(gt_path, gt_img)
        print(f"已保存GT图像: {gt_path}（对应时间点：{timestamp}）")

if __name__ == "__main__":
    # 配置路径
    npz_file_path = "spike_by/group_0011.npz"  # 你的npz文件路径
    output_root = "./Spk2ImgNet_train/train2/"  # 训练数据根目录（与原代码一致）
    
    # 执行提取和保存
    extract_and_save(npz_file_path, output_root)