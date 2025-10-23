"""
convert_npy_to_dat.py

将目录下的 .npy 文件（每个文件包含 shape [T,H,W] 的 0/1 spike 矩阵）
转换为项目中使用的 .dat 位打包文件，并保存到 ./spike/input/ 下。

用法示例：
    python convert_npy_to_dat.py --npy_dir .\spike --out_dir .\spike\input --target_h 256 --target_w 256

说明：生成的 .dat 与项目中 dataset.raw_to_spike() 的读取方式兼容。
如果你的测试脚本（如 test_gen_imgseq.py）仍然硬编码了 h=250,w=400，需要把对应调用改为你的目标尺寸（例如 256,256），或在此脚本中使用 --target_h/--target_w 填充/裁剪到 250x400。
"""
import os
import argparse
import glob
import numpy as np


def pad_or_crop_frame(frame, target_h, target_w):
    """中心裁剪或对称填充单帧到 target_h x target_w"""
    h, w = frame.shape
    # crop if larger
    if target_h is not None and target_w is not None:
        # vertical
        if h > target_h:
            top = (h - target_h) // 2
            frame = frame[top: top + target_h, :]
            h = target_h
        if w > target_w:
            left = (w - target_w) // 2
            frame = frame[:, left: left + target_w]
            w = target_w
        # pad if smaller
        pad_top = pad_bottom = pad_left = pad_right = 0
        if h < target_h:
            total = target_h - h
            pad_top = total // 2
            pad_bottom = total - pad_top
        if w < target_w:
            total = target_w - w
            pad_left = total // 2
            pad_right = total - pad_left
        if any((pad_top, pad_bottom, pad_left, pad_right)):
            frame = np.pad(frame, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    return frame


def convert_file(npy_path, out_path, target_h=None, target_w=None):
    arr = np.load(npy_path)
    if arr.ndim != 3:
        raise ValueError(f"Expected .npy with shape [T,H,W], got {arr.shape} for {npy_path}")
    T, H, W = arr.shape
    # If target dims not provided, use original
    if target_h is None:
        target_h = H
    if target_w is None:
        target_w = W

    img_size = target_h * target_w
    if img_size % 8 != 0:
        raise ValueError(f"Target H*W must be divisible by 8, got {target_h}x{target_w}")

    num_bytes = img_size // 8
    out_bytes = bytearray()

    for t in range(T):
        frame = arr[t]
        # ensure binary 0/1 and uint8
        frame = (frame > 0).astype(np.uint8)
        # crop/pad to target
        if (H != target_h) or (W != target_w):
            frame = pad_or_crop_frame(frame, target_h, target_w)
        # flip vertically because raw_to_spike applies flipud after unpacking
        frame = np.flipud(frame)
        flat = frame.flatten(order='C')
        # pack bits little-endian so bit0 corresponds to first pixel in group
        packed = np.packbits(flat, bitorder='little')
        if packed.size != num_bytes:
            # safety check
            raise RuntimeError(f'Unexpected packed size {packed.size} vs {num_bytes}')
        out_bytes.extend(packed.tobytes())

    # write
    with open(out_path, 'wb') as f:
        f.write(out_bytes)


def convert_array(arr, out_path, target_h=None, target_w=None):
    """Convert a numpy array with shape [T,H,W] (0/1 spikes) to .dat and write to out_path."""
    if arr.ndim != 3:
        raise ValueError(f"Expected array with shape [T,H,W], got {arr.shape}")
    T, H, W = arr.shape
    if target_h is None:
        target_h = H
    if target_w is None:
        target_w = W

    img_size = target_h * target_w
    if img_size % 8 != 0:
        raise ValueError(f"Target H*W must be divisible by 8, got {target_h}x{target_w}")

    num_bytes = img_size // 8
    out_bytes = bytearray()

    for t in range(T):
        frame = arr[t]
        frame = (frame > 0).astype(np.uint8)
        if (H != target_h) or (W != target_w):
            frame = pad_or_crop_frame(frame, target_h, target_w)
        frame = np.flipud(frame)
        flat = frame.flatten(order='C')
        packed = np.packbits(flat, bitorder='little')
        if packed.size != num_bytes:
            raise RuntimeError(f'Unexpected packed size {packed.size} vs {num_bytes}')
        out_bytes.extend(packed.tobytes())

    with open(out_path, 'wb') as f:
        f.write(out_bytes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_dir', type=str, default='./spike', help='Directory containing .npy files')
    parser.add_argument('--out_dir', type=str, default='./spike/input', help='Output directory for .dat files')
    parser.add_argument('--target_h', type=int, default=None, help='Optional target height to crop/pad to')
    parser.add_argument('--target_w', type=int, default=None, help='Optional target width to crop/pad to')
    parser.add_argument('--ext', type=str, default='npz', help='Extension of input files (npy or npz)')
    parser.add_argument('--npz_key', type=str, default='left_spk', help='Key to extract from .npz files (default left_spk)')
    args = parser.parse_args()

    npy_dir = args.npy_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    files = glob.glob(os.path.join(npy_dir, f'*.{args.ext}'))
    files.sort()
    if not files:
        print(f'No .{args.ext} files found in {npy_dir}')
        return

    print(f'Found {len(files)} files, converting to {out_dir} with target size {args.target_h}x{args.target_w}')
    for p in files:
        base = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(out_dir, base + '.dat')
        print('Converting', p, '->', out_path)
        if args.ext.lower() == 'npz':
            data = np.load(p)
            if args.npz_key in data:
                arr = data[args.npz_key]
            else:
                # try to take the first array-like item
                keys = [k for k in data.files]
                if not keys:
                    raise RuntimeError(f'No arrays found inside {p}')
                print(f"Warning: key '{args.npz_key}' not found in {p}, using first key '{keys[0]}'")
                arr = data[keys[0]]
            convert_array(arr, out_path, target_h=args.target_h, target_w=args.target_w)
        else:
            convert_file(p, out_path, target_h=args.target_h, target_w=args.target_w)

    print('Done.')


if __name__ == '__main__':
    main()
