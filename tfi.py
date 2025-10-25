
default="./Spk2ImgNet_test2/test2/input/group_0011.dat",
import argparse
import numpy as np
import cv2
import os


def parse_args():
    parser = argparse.ArgumentParser(description="通过 TFI 方法将 DAT 脉冲文件转换为灰度图（适配1位/像素存储）")
    parser.add_argument("--dat_path", type=str, default="./spike/input/group_0000.dat", 
                        help="输入 DAT 文件路径（如 ./input/group_0011.dat）")
    parser.add_argument("--img_h", type=int, default=256, 
                        help="输出灰度图高度（需与脉冲序列的图像高度一致，默认256）")
    parser.add_argument("--img_w", type=int, default=256, 
                        help="输出灰度图宽度（需与脉冲序列的图像宽度一致，默认256）")
    parser.add_argument("--tfi_window", type=int, default=800, 
                        help="TFI 积分窗口大小（默认800帧，建议与脉冲总帧数一致）")
    parser.add_argument("--save_dir", type=str, default="./tfi_gray_imgs", 
                        help="灰度图保存目录（默认自动创建）")
    return parser.parse_args()


def read_dat_1bit_per_pixel(dat_path, img_h, img_w):
    """
    读取 1位/像素 存储的 DAT 脉冲文件（适配你的spike_to_dat函数）
    核心：将每个字节的8位解包为8个像素，还原真实脉冲序列
    """
    with open(dat_path, "rb") as f:
        raw_data = np.frombuffer(f.read(), dtype="B")  # 读取所有字节（每个字节含8个像素的脉冲信息）
    
    # 计算单帧需要的字节数（1位/像素 → 总像素数/8）
    pixels_per_frame = img_h * img_w
    bytes_per_frame = pixels_per_frame // 8
    if pixels_per_frame % 8 != 0:
        raise ValueError(f"图像像素数 {pixels_per_frame} 需为8的倍数（1位/像素存储要求）")
    
    # 计算真实总帧数（总字节数 ÷ 单帧字节数）
    total_frames = len(raw_data) // bytes_per_frame
    if len(raw_data) % bytes_per_frame != 0:
        raise ValueError(f"DAT 文件长度不匹配！总字节数 {len(raw_data)} 无法被单帧字节数 {bytes_per_frame} 整除")
    
    # 关键：将1位/像素的字节数据解包为1字节/像素的脉冲序列（0/1）
    spike_array = np.zeros((total_frames, img_h, img_w), dtype=np.uint8)
    for t in range(total_frames):
        # 读取当前帧的所有字节（共 bytes_per_frame 个字节）
        frame_bytes = raw_data[t*bytes_per_frame : (t+1)*bytes_per_frame]
        # 遍历每个字节，解包8个像素的脉冲值
        for byte_idx in range(len(frame_bytes)):
            byte = frame_bytes[byte_idx]
            # 遍历字节的8个bit（对应8个像素）
            for bit_idx in range(8):
                # 计算当前像素在帧内的索引（平坦化）
                pixel_flat_idx = byte_idx * 8 + bit_idx
                if pixel_flat_idx >= pixels_per_frame:
                    break  # 避免超出单帧像素数（极端情况）
                # 转换为2D坐标（i=行，j=列）
                i = pixel_flat_idx // img_w
                j = pixel_flat_idx % img_w
                # 提取当前bit的值（1=有脉冲，0=无脉冲）
                spike_value = (byte >> bit_idx) & 1
                spike_array[t, i, j] = spike_value
    
    print(f"成功读取 DAT 文件：{dat_path}")
    print(f"脉冲序列信息：总帧数={total_frames}, 尺寸={img_h}×{img_w}（1位/像素存储适配）")
    return spike_array


def tfi_spike_to_gray(spike_array, tfi_window):
    total_frames = spike_array.shape[0]
    actual_window = min(tfi_window, total_frames)
    print(f"TFI 积分配置：设定窗口={tfi_window}，实际窗口={actual_window}（总帧数={total_frames}）")
    
    # 时间域积分（统计每个像素的脉冲总数）
    tfi_integral = np.sum(spike_array[:actual_window, :, :], axis=0)
    
    # 归一化到0~255
    integral_max = tfi_integral.max()
    integral_min = tfi_integral.min()
    if integral_max > integral_min:
        gray_img = (tfi_integral - integral_min) / (integral_max - integral_min) * 255
    else:
        gray_img = np.zeros_like(tfi_integral)
    gray_img = gray_img.astype(np.uint8)
    
    print(f"TFI 积分完成：积分最大值={integral_max}, 灰度图范围=0~255")
    return gray_img


def save_gray_img(gray_img, dat_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    dat_filename = os.path.basename(dat_path)
    gray_filename = os.path.splitext(dat_filename)[0] + "_tfi_gray.png"
    save_path = os.path.join(save_dir, gray_filename)
    cv2.imwrite(save_path, gray_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"灰度图已保存：{save_path}")
    return save_path


def main():
    args = parse_args()
    # 替换为适配1位/像素的读取函数
    spike_array = read_dat_1bit_per_pixel(args.dat_path, args.img_h, args.img_w)
    gray_img = tfi_spike_to_gray(spike_array, args.tfi_window)
    save_gray_img(gray_img, args.dat_path, args.save_dir)
    print("\n=== DAT 转灰度图（TFI 方法）完成 ===")


if __name__ == "__main__":
    main()