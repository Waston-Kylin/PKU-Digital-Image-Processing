#-*-coding:utf-8 -*-
import numpy as np
import cv2
import gradio as gr


def function_hw1(input_image, mode, h_slider, s_slider, l_slider, brightness, contrast):
    if input_image is None:
        raise gr.Error('输入错误：在处理之前请先输入一张图像', duration=5)    

    def rgb_to_hsl(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        cmax = np.max(rgb, axis=2)
        cmin = np.min(rgb, axis=2)
        delta = cmax - cmin
        l = (cmax + cmin) / 2.0
        
        s = np.zeros_like(l)
        mask = delta != 0
        s[mask] = delta[mask] / (1.0 - np.abs(2.0 * l[mask] - 1.0))
        
        h = np.zeros_like(l)
        mask_r = (cmax == r) & mask
        h[mask_r] = 60.0 * np.mod((g[mask_r] - b[mask_r]) / delta[mask_r], 6)
        mask_g = (cmax == g) & mask
        h[mask_g] = 60.0 * ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2)
        mask_b = (cmax == b) & mask
        h[mask_b] = 60.0 * ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4)
        return np.stack([h, s, l], axis=2)

    def hsl_to_rgb(hsl):
        h, s, l = hsl[:,:,0], hsl[:,:,1], hsl[:,:,2]
        c = (1.0 - np.abs(2.0 * l - 1.0)) * s
        x = c * (1.0 - np.abs(np.mod(h / 60.0, 2) - 1.0))
        m = l - c / 2.0
        r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)
        
        mask = (0 <= h) & (h < 60)
        r[mask], g[mask], b[mask] = c[mask], x[mask], 0
        mask = (60 <= h) & (h < 120)
        r[mask], g[mask], b[mask] = x[mask], c[mask], 0
        mask = (120 <= h) & (h < 180)
        r[mask], g[mask], b[mask] = 0, c[mask], x[mask]
        mask = (180 <= h) & (h < 240)
        r[mask], g[mask], b[mask] = 0, x[mask], c[mask]
        mask = (240 <= h) & (h < 300)
        r[mask], g[mask], b[mask] = x[mask], 0, c[mask]
        mask = (300 <= h) & (h < 360)
        r[mask], g[mask], b[mask] = c[mask], 0, x[mask]
        
        r, g, b = r + m, g + m, b + m
        return np.stack([r, g, b], axis=2)

    img_float = input_image.astype(np.float32) / 255.0

    # 直接调用 OpenCV
    if mode == "OpenCV HLS":
        hls = cv2.cvtColor(img_float, cv2.COLOR_RGB2HLS)
        h, l, s = cv2.split(hls)

        h = np.mod(h + h_slider * 180.0, 360.0)
        l = np.clip(l + l_slider, 0, 1)
        s = np.clip(s + s_slider, 0, 1)

        output_float = cv2.cvtColor(cv2.merge([h, l, s]), cv2.COLOR_HLS2RGB)
    # 归一化到 [0, 1] 方便计算
    elif mode == "色相/饱和度":
        # 手写转换 + PS自然饱和度
        hsl = rgb_to_hsl(img_float)
        h, s, l = hsl[:,:,0], hsl[:,:,1], hsl[:,:,2]
        
        h = np.mod(h + h_slider * 180.0, 360.0)
        l = np.clip(l + l_slider, 0, 1)

        if s_slider > 0:
            mask = s > 1e-6
            s[mask] = s[mask] + s_slider * (1.0 - s[mask]) * s[mask]
        else:
            s = s * (1.0 + s_slider)
        
        s = np.clip(s, 0, 1)   
        hsl[:,:,0], hsl[:,:,1], hsl[:,:,2] = h, s, l
        output_float = hsl_to_rgb(hsl)
    else:
        # 拟合 Photoshop 亮度/对比度
        if brightness > 0:
            output_float = 1.0 - np.power(1.0 - img_float, 1.0 + brightness * 2.0)
        else:
            output_float = np.power(img_float, 1.0 - brightness * 2.0)
            
        if contrast > 0:
            # 上限保护
            gamma_contrast = 1.0 - contrast * 0.9
            shifted = output_float - 0.5
            output_float = np.sign(shifted) * np.power(np.abs(shifted) * 2.0, gamma_contrast) / 2.0 + 0.5
        else:
            # 降低对比度：像素向中性灰(0.5)平滑聚拢
            output_float = (output_float - 0.5) * (1.0 + contrast) + 0.5
            
        output_float = np.clip(output_float, 0, 1)

    output_image = np.clip(output_float * 255.0, 0, 255).astype(np.uint8)
    
    # 提取六个通道展示图
    r_channel = np.stack([output_image[:,:,0]]*3, axis=-1)
    g_channel = np.stack([output_image[:,:,1]]*3, axis=-1)
    b_channel = np.stack([output_image[:,:,2]]*3, axis=-1)
    
    out_hsl = rgb_to_hsl(output_float)
    h_channel = np.clip((out_hsl[:,:,0] / 360.0) * 255, 0, 255).astype(np.uint8)
    h_channel = np.stack([h_channel]*3, axis=-1)
    s_channel = np.clip(out_hsl[:,:,1] * 255, 0, 255).astype(np.uint8)
    s_channel = np.stack([s_channel]*3, axis=-1)
    l_channel = np.clip(out_hsl[:,:,2] * 255, 0, 255).astype(np.uint8)
    l_channel = np.stack([l_channel]*3, axis=-1)

    return [output_image, r_channel, g_channel, b_channel, h_channel, s_channel, l_channel]

def function_hw2(mode="generate", seed_gen=42, seed_grid=888, pos_str="", neg_str="", intensity=1.0):
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.utils as vutils

    # 加载模型
    global netG_hw2, global_vA_hw2
    netG_hw2 = globals().get('netG_hw2')
    global_vA_hw2 = globals().get('global_vA_hw2')

    if netG_hw2 is None:
        nz, ngf, nc = 100, 64, 3
        device = torch.device("cpu")

        class Generator(nn.Module):
            def __init__(self):
                super(Generator, self).__init__()
                self.main = nn.Sequential(
                    nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True),
                    nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh()
                )
            def forward(self, input):
                return self.main(input)

        netG_hw2 = Generator().to(device)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = ["dcgan_checkpoint.pth", os.path.join(base_dir, "dcgan_checkpoint.pth")]
        ckpt_path = next((p for p in possible_paths if os.path.exists(p)), None)

        if ckpt_path:
            try:
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(ckpt_path, map_location=device)
                
            if isinstance(ckpt, dict):
                clean_dict = {k.replace('module.', ''): v for k, v in ckpt.get('state_dict', ckpt.get('generator', ckpt)).items()}
                netG_hw2.load_state_dict(clean_dict, strict=False)
            else:
                netG_hw2 = ckpt.to(device)
        else:
            raise gr.Error("找不到 dcgan_checkpoint.pth，请确保文件在同目录下。", duration=10)
        netG_hw2.eval()

    def tensor_to_numpy(t, nrow=8, upscale=1):
        if upscale > 1:
            t = F.interpolate(t, scale_factor=upscale, mode='nearest')
        grid = vutils.make_grid(t, nrow=nrow, normalize=True, padding=2)
        img = grid.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        return (img * 255.0).clip(0, 255).astype(np.uint8)

    device = torch.device("cpu")

    # 路由分配
    if mode == "generate":
        torch.manual_seed(int(seed_gen))
        z = torch.randn(1, 100, 1, 1, device=device)
        with torch.no_grad():
            fake = netG_hw2(z)
        # 放大 4 倍
        return tensor_to_numpy(fake, nrow=1, upscale=4), None

    elif mode == "grid":
        torch.manual_seed(int(seed_grid))
        z_64 = torch.randn(64, 100, 1, 1, device=device)
        with torch.no_grad():
            fake_64 = netG_hw2(z_64)
        # 放大 2 倍
        return tensor_to_numpy(fake_64, nrow=8, upscale=2), None

    elif mode == "calc":
        torch.manual_seed(int(seed_grid))
        z_64 = torch.randn(64, 100, 1, 1, device=device)
        
        # 1~64 转为 0~63
        pos_idx = [int(i.strip()) - 1 for i in str(pos_str).split() if i.strip().isdigit() and 1 <= int(i.strip()) <= 64]
        neg_idx = [int(i.strip()) - 1 for i in str(neg_str).split() if i.strip().isdigit() and 1 <= int(i.strip()) <= 64]

        if not pos_idx or not neg_idx:
            raise gr.Error("编号不能为空且必须在 1~64 之间！", duration=5)

        global_vA_hw2 = z_64[pos_idx].mean(dim=0, keepdim=True) - z_64[neg_idx].mean(dim=0, keepdim=True)

        with torch.no_grad():
            fake_pos = netG_hw2(z_64[pos_idx])
            fake_neg = netG_hw2(z_64[neg_idx])
            
        # 每行最多4张，upscale=2 固定尺寸
        img_pos = tensor_to_numpy(fake_pos, nrow=4, upscale=2)
        img_neg = tensor_to_numpy(fake_neg, nrow=4, upscale=2)
        return img_pos, img_neg

    elif mode == "edit":
        torch.manual_seed(int(seed_gen))
        z_new = torch.randn(1, 100, 1, 1, device=device)
        
        if 'global_vA_hw2' in globals() and global_vA_hw2 is not None:
            z_edited = z_new + float(intensity) * global_vA_hw2
        else:
            z_edited = z_new
            
        with torch.no_grad():
            img_orig = tensor_to_numpy(netG_hw2(z_new), nrow=1, upscale=4)
            img_edit = tensor_to_numpy(netG_hw2(z_edited), nrow=1, upscale=4)
            
        # 左右拼接
        sep = np.full((img_orig.shape[0], 6, 3), 255, dtype=np.uint8)
        img_res = np.concatenate([img_orig, sep, img_edit], axis=1)
        return img_res, None

def function_hw3(task_type, bg_image, fg_image, mode, calc_impl, bg_video):
    def get_corners_from_mask(mask_layer):
        """画笔蒙版中提取并排序4个角点"""
        _, binary = cv2.threshold(mask_layer, 127, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        centers = [tuple(centroids[i]) for i in range(1, num_labels)]
        if len(centers) != 4:
            raise gr.Error(f"输入错误：请在背景图像上用画笔准确标出屏幕的 4 个顶点！当前检测到 {len(centers)} 个点", duration=5)
        
        centers = np.array(centers, dtype=np.float32)
        centers = centers[np.argsort(centers[:, 1])]  # 按 y 排序分为上下两组
        top, bottom = centers[:2], centers[2:]
        tl = top[np.argsort(top[:, 0])][0]
        tr = top[np.argsort(top[:, 0])][1]
        bl = bottom[np.argsort(bottom[:, 0])][0]
        br = bottom[np.argsort(bottom[:, 0])][1]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def get_homography(src, dst, manual=False):
        """计算单应性矩阵"""
        if manual:
            A = []
            for i in range(4):
                x, y = src[i]
                u, v = dst[i]
                A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
                A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
            A = np.array(A, dtype=np.float64)
            _, _, V = np.linalg.svd(A)
            H = V[-1].reshape(3, 3)
            return H / H[2, 2]
        else:
            H, _ = cv2.findHomography(src, dst)
            return H

    def warp_and_blend(bg, fg, H, interp_mode, manual_calc):
        """图像几何变换与像素插值拼贴"""
        # 确保通道清洗为 3 通道 (RGB)
        if bg.shape[-1] == 4: bg = bg[:, :, :3]
        if fg.shape[-1] == 4: fg = fg[:, :, :3]
        
        h_bg, w_bg = bg.shape[:2]
        h_fg, w_fg = fg.shape[:2]
        src_corners = np.array([[0, 0], [w_fg, 0], [w_fg, h_fg], [0, h_fg]], dtype=np.float32)

        if not manual_calc:
            flags = cv2.INTER_NEAREST if interp_mode == 'Nearest' else (cv2.INTER_LINEAR if interp_mode == 'Bilinear' else cv2.INTER_CUBIC)
            warped_fg = cv2.warpPerspective(fg, H, (w_bg, h_bg), flags=flags)
            mask = np.ones((h_fg, w_fg), dtype=np.uint8) * 255
            warped_mask = cv2.warpPerspective(mask, H, (w_bg, h_bg), flags=cv2.INTER_NEAREST)
        else:
            # 空间坐标逆映射与插值算法
            warped_fg = np.zeros_like(bg)
            warped_mask = np.zeros((h_bg, w_bg), dtype=np.uint8)
            
            dst_corners = cv2.perspectiveTransform(src_corners.reshape(-1, 1, 2), H).squeeze()
            x_min = max(0, int(np.floor(np.min(dst_corners[:, 0]))))
            x_max = min(w_bg - 1, int(np.ceil(np.max(dst_corners[:, 0]))))
            y_min = max(0, int(np.floor(np.min(dst_corners[:, 1]))))
            y_max = min(h_bg - 1, int(np.ceil(np.max(dst_corners[:, 1]))))
            
            H_inv = np.linalg.inv(H)
            X, Y = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
            pts_bg = np.stack([X.ravel(), Y.ravel(), np.ones_like(X).ravel()], axis=-1)
            
            pts_fg = pts_bg.dot(H_inv.T)
            pts_fg = pts_fg[:, :2] / pts_fg[:, 2:]
            x_fg, y_fg = pts_fg[:, 0], pts_fg[:, 1]
            
            valid = (x_fg >= 0) & (x_fg < w_fg - 1) & (y_fg >= 0) & (y_fg < h_fg - 1)
            x_v, y_v = x_fg[valid], y_fg[valid]
            
            if interp_mode == 'Nearest':
                x_idx, y_idx = np.round(x_v).astype(int), np.round(y_v).astype(int)
                vals = fg[y_idx, x_idx]
            elif interp_mode == 'Bilinear':
                x0, y0 = np.floor(x_v).astype(int), np.floor(y_v).astype(int)
                x1, y1 = x0 + 1, y0 + 1
                dx, dy = x_v - x0, y_v - y0
                w00, w10 = (1 - dx) * (1 - dy), dx * (1 - dy)
                w01, w11 = (1 - dx) * dy, dx * dy
                vals = (fg[y0, x0] * w00[:, None] + fg[y0, x1] * w10[:, None] + 
                        fg[y1, x0] * w01[:, None] + fg[y1, x1] * w11[:, None])
            else: # Bicubic
                def cubic_weight(t):
                    t = np.abs(t)
                    w = np.zeros_like(t)
                    m1, m2 = t <= 1, (t > 1) & (t <= 2)
                    w[m1] = 1.5 * t[m1]**3 - 2.5 * t[m1]**2 + 1
                    w[m2] = -0.5 * t[m2]**3 + 2.5 * t[m2]**2 - 4 * t[m2] + 2
                    return w

                x0, y0 = np.floor(x_v).astype(int), np.floor(y_v).astype(int)
                vals = np.zeros((len(x_v), 3))
                for i in range(-1, 3):
                    for j in range(-1, 3):
                        xx = np.clip(x0 + i, 0, w_fg - 1)
                        yy = np.clip(y0 + j, 0, h_fg - 1)
                        wx, wy = cubic_weight(x_v - (x0 + i)), cubic_weight(y_v - (y0 + j))
                        vals += fg[yy, xx] * (wx * wy)[:, None]
                        
            valid_X, valid_Y = X.ravel()[valid], Y.ravel()[valid]
            warped_fg[valid_Y, valid_X] = np.clip(vals, 0, 255).astype(np.uint8)
            warped_mask[valid_Y, valid_X] = 255

        mask_3c = np.stack([warped_mask]*3, axis=-1) / 255.0
        blended = bg * (1 - mask_3c) + warped_fg * mask_3c
        return blended.astype(np.uint8)

    if task_type == "image":
        # === 静态图像模式 (70分/85分) ===
        if bg_image is None or bg_image["background"] is None:
            raise gr.Error('输入错误：请先输入一张背景图像并在上面点 4 个红点', duration=5)  
        if fg_image is None:
            raise gr.Error('输入错误：请先输入一张前景展示图像', duration=5)

        bg = bg_image["background"]
        dst_pts = get_corners_from_mask(bg_image["layers"][0][:, :, 0])
        h_fg, w_fg = fg_image.shape[:2]
        src_pts = np.array([[0, 0], [w_fg, 0], [w_fg, h_fg], [0, h_fg]], dtype=np.float32)

        is_manual = (calc_impl == '手动实现(矩阵计算+像素插值)')
        H = get_homography(src_pts, dst_pts, manual=is_manual)
        output_image = warp_and_blend(bg, fg_image, H, mode, manual_calc=is_manual)
        return output_image

    else:
        # 视频追踪替换模式
        if bg_image is None or bg_image["background"] is None:
            raise gr.Error('输入错误：请在视频第一帧图像上标出屏幕 4 个顶点', duration=5)
        if fg_image is None:
            raise gr.Error('输入错误：请上传要替换的前景展示图像', duration=5)
        if bg_video is None:
            raise gr.Error('输入错误：请上传要处理的背景视频文件 (.mp4)', duration=5)

        first_frame = bg_image["background"]
        if first_frame.shape[-1] == 4: first_frame = first_frame[:, :, :3]
        if fg_image.shape[-1] == 4: fg_image = fg_image[:, :, :3]

        dst_pts = get_corners_from_mask(bg_image["layers"][0][:, :, 0])
        h_fg, w_fg = fg_image.shape[:2]
        src_pts = np.array([[0, 0], [w_fg, 0], [w_fg, h_fg], [0, h_fg]], dtype=np.float32)

        cap = cv2.VideoCapture(bg_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 系统临时文件夹
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        out_video_path = temp_file.name
        temp_file.close()

        # 使用 avc1 (H.264) 编码
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

        old_gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)

        mask_features = np.zeros((h, w), dtype=np.uint8)
        x_min, y_min = np.min(dst_pts, axis=0)
        x_max, y_max = np.max(dst_pts, axis=0)
        bw, bh = x_max - x_min, y_max - y_min
        
        # 扩大检测区域
        x0, y0 = max(0, int(x_min - bw * 0.15)), max(0, int(y_min - bh * 0.15))
        x1, y1 = min(w, int(x_max + bw * 0.15)), min(h, int(y_max + bh * 0.35))
        mask_features[y0:y1, x0:x1] = 255

        pts_gftt = cv2.goodFeaturesToTrack(
            old_gray, maxCorners=250, qualityLevel=0.01, minDistance=8, mask=mask_features
        )

        # 沿 4 条屏幕边均匀插值采样边缘特征点
        edge_pts = []
        for i in range(4):
            p_start, p_end = dst_pts[i], dst_pts[(i + 1) % 4]
            for frac in np.linspace(0.1, 0.9, 8):
                edge_pts.append(p_start * (1 - frac) + p_end * frac)
        edge_pts = np.array(edge_pts, dtype=np.float32).reshape(-1, 1, 2)

        # 合并 4 个角点、边框点和角点群
        p_list = [dst_pts.reshape(-1, 1, 2), edge_pts]
        if pts_gftt is not None and len(pts_gftt) > 0:
            p_list.append(pts_gftt)
        p_init = np.vstack(p_list)

        # p_track 记录当前帧坐标，p_orig 永远锚定第 0 帧原始坐标
        p_track = p_init.copy()
        p_orig = p_init.copy()
        last_good_dst = dst_pts.copy()

        lk_params = dict(
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        mask_fg_float = np.ones((h_fg, w_fg), dtype=np.float32)

        while True:
            ret, frame = cap.read()
            if not ret: break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

            # 前向光流与后向光流双向验证
            p_curr, st1, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p_track, None, **lk_params)
            p_back, st2, _ = cv2.calcOpticalFlowPyrLK(frame_gray, old_gray, p_curr, None, **lk_params)

            # 剔除单向滑动或光照异常漂移的点
            fb_err = np.linalg.norm(p_track - p_back, axis=2).squeeze()
            valid = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb_err < 1.0)

            p_track_valid = p_curr[valid]
            p_orig_valid = p_orig[valid]

            if len(p_orig_valid) >= 10:
                # 锚定第 0 帧进行 RANSAC 单应性矩阵求解
                H_0_to_t, inliers = cv2.findHomography(p_orig_valid, p_track_valid, cv2.RANSAC, 2.5)
                if H_0_to_t is not None:
                    curr_dst = cv2.perspectiveTransform(dst_pts.reshape(-1, 1, 2), H_0_to_t).squeeze()
                    last_good_dst = curr_dst
                    inlier_idx = (inliers.ravel() == 1)
                    p_track = p_track_valid[inlier_idx]
                    p_orig = p_orig_valid[inlier_idx]
                else:
                    curr_dst = last_good_dst
                    p_track, p_orig = p_track_valid, p_orig_valid
            else:
                curr_dst = last_good_dst
                p_track, p_orig = p_track_valid, p_orig_valid

            # 渲染前景至当前帧屏幕区域
            H_fg, _ = cv2.findHomography(src_pts, curr_dst)
            if H_fg is not None:
                warped_fg = cv2.warpPerspective(fg_image, H_fg, (w, h), flags=cv2.INTER_LINEAR)
                # 边缘抗锯齿羽化融合
                warped_mask = cv2.warpPerspective(mask_fg_float, H_fg, (w, h), flags=cv2.INTER_LINEAR)
                mask_3c = np.stack([warped_mask] * 3, axis=-1)
                blended = (frame_rgb * (1.0 - mask_3c) + warped_fg * mask_3c).clip(0, 255).astype(np.uint8)
                out.write(cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
            else:
                out.write(frame)

            old_gray = frame_gray

        cap.release()
        out.release()
        return out_video_path

def function_hw4(input_image, mode="手动导向滤波", radius=4, eps=0.02, sigma_s=3.0, sigma_r=0.1, subsample=4, sigma_bm3d=25.0, sharpen_amount=2.5):
    if input_image is None:
        raise gr.Error('输入错误：在处理之前请先输入一张图像', duration=5)

    # 统一剥离可能存在的 Alpha 通道，保证 3 通道 RGB
    if input_image.ndim == 3 and input_image.shape[-1] == 4:
        input_image = input_image[:, :, :3]

    # 调用 OpenCV
    # # 调用 bilateralFilter 实现双边滤波
    # cv_bilateral = cv2.bilateralFilter(input_image, d=int(radius)*2+1, sigmaColor=float(sigma_r)*255.0, sigmaSpace=float(sigma_s))
    #
    # 调用 fastNlMeansDenoisingColored 实现 NLM 滤波
    # cv_nlm = cv2.fastNlMeansDenoisingColored(input_image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

    img_float = input_image.astype(np.float32) / 255.0

    # 盒式滤波
    def manual_box_filter(img, r):
        r = max(1, int(r))
        if img.ndim == 2:
            img_pad = np.pad(img, ((r, r), (r, r)), mode='reflect')
        else:
            img_pad = np.pad(img, ((r, r), (r, r), (0, 0)), mode='reflect')

        ksize = 2 * r + 1
        # 沿 Y 轴利用差分累积和计算滑动窗口和
        cum_y = np.cumsum(img_pad, axis=0, dtype=np.float64)
        diff_y = cum_y[ksize - 1:, ...] - np.concatenate([np.zeros((1,) + img_pad.shape[1:], dtype=np.float64), cum_y[:-ksize, ...]], axis=0)

        # 沿 X 轴利用差分累积和计算滑动窗口和
        cum_x = np.cumsum(diff_y, axis=1, dtype=np.float64)
        diff_x = cum_x[:, ksize - 1:, ...] - np.concatenate([np.zeros((diff_y.shape[0], 1) + img_pad.shape[2:], dtype=np.float64), cum_x[:, :-ksize, ...]], axis=1)

        return (diff_x / float(ksize * ksize)).astype(np.float32)

    # 导向滤波
    def manual_guided_filter(I, p, r, eps_val):
        mean_I = manual_box_filter(I, r)
        mean_p = manual_box_filter(p, r)
        mean_Ip = manual_box_filter(I * p, r)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = manual_box_filter(I * I, r)
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + float(eps_val))
        b = mean_p - a * mean_I

        mean_a = manual_box_filter(a, r)
        mean_b = manual_box_filter(b, r)

        q = mean_a * I + mean_b
        return np.clip(q, 0.0, 1.0)
    
    # 基于盒式滤波优化的快速导向滤波
    def manual_fast_guided_filter(I, p, r, eps_val, s):
        s = max(1, int(s))
        h, w = I.shape[:2]
        small_w, small_h = max(2, w // s), max(2, h // s)

        # 下采样引导图与输入图
        I_sub = cv2.resize(I, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        p_sub = cv2.resize(p, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        r_sub = max(1, int(round(r / float(s))))

        # 低分辨率空间盒式滤波计算局部线性系数
        mean_I = manual_box_filter(I_sub, r_sub)
        mean_p = manual_box_filter(p_sub, r_sub)
        mean_Ip = manual_box_filter(I_sub * p_sub, r_sub)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = manual_box_filter(I_sub * I_sub, r_sub)
        var_I = mean_II - mean_I * mean_I

        a_sub = cov_Ip / (var_I + float(eps_val))
        b_sub = mean_p - a_sub * mean_I

        mean_a_sub = manual_box_filter(a_sub, r_sub)
        mean_b_sub = manual_box_filter(b_sub, r_sub)

        # 双线性上采样系数至原始分辨率
        mean_a = cv2.resize(mean_a_sub, (w, h), interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(mean_b_sub, (w, h), interpolation=cv2.INTER_LINEAR)

        q = mean_a * I + mean_b
        return np.clip(q, 0.0, 1.0)

    # 双边滤波算法
    def manual_bilateral_filter(img, r, s_space, s_color):
        r = max(1, int(r))
        s_space = max(1e-5, float(s_space))
        s_color = max(1e-5, float(s_color))
        h, w, c = img.shape
        img_pad = np.pad(img, ((r, r), (r, r), (0, 0)), mode='reflect')

        out = np.zeros_like(img, dtype=np.float32)
        w_sum = np.zeros((h, w, 1), dtype=np.float32)

        two_ss2 = 2.0 * (s_space ** 2)
        two_sc2 = 2.0 * (s_color ** 2)

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                spatial_w = np.exp(-(dx * dx + dy * dy) / two_ss2)
                shifted = img_pad[r + dy:r + dy + h, r + dx:r + dx + w, :]
                color_diff2 = np.sum((img - shifted) ** 2, axis=2, keepdims=True)
                range_w = np.exp(-color_diff2 / two_sc2)

                weight = spatial_w * range_w
                out += shifted * weight
                w_sum += weight

        return np.clip(out / (w_sum + 1e-8), 0.0, 1.0)

    # BM3D 算法
    def manual_bm3d(img_rgb, sigma_val=25.0):
        # 转换至 YCrCb 空间：对亮度通道 Y 执行完整 BM3D，对色度通道保边去噪
        ycrcb = cv2.cvtColor((img_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
        Y = ycrcb[:, :, 0]
        h, w = Y.shape

        # 构建 8x8 正交余弦变换矩阵与 1D Haar 小波变换矩阵
        N = 8
        dct_mat = np.zeros((N, N), dtype=np.float32)
        for k in range(N):
            alpha = np.sqrt(1.0 / N) if k == 0 else np.sqrt(2.0 / N)
            for n in range(N):
                dct_mat[k, n] = alpha * np.cos(np.pi * (2 * n + 1) * k / (2.0 * N))
        dct_mat_T = dct_mat.T

        def get_haar_matrix(size):
            if size == 1:
                return np.array([[1.0]], dtype=np.float32)
            H_mat = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32) / np.sqrt(2.0)
            while H_mat.shape[0] < size:
                cur = H_mat.shape[0]
                I_mat = np.eye(cur, dtype=np.float32)
                top = np.kron(H_mat[0:1, :], np.array([[1.0, 1.0]], dtype=np.float32) / np.sqrt(2.0))
                if cur > 1:
                    mid = np.kron(H_mat[1:, :], np.array([[1.0, 1.0]], dtype=np.float32) / np.sqrt(2.0))
                    top = np.vstack([top, mid])
                bot = np.kron(I_mat, np.array([[1.0, -1.0]], dtype=np.float32) / np.sqrt(2.0))
                H_mat = np.vstack([top, bot])
            return H_mat

        S = 13 # 搜索窗口半宽
        K_max = 8 # 3D 组最大相似块数 (2的幂次)
        step = 4 # 参考块滑动步长
        sigma_val = max(2.0, float(sigma_val))
        lam_3d = 2.6 * sigma_val  # 3D 硬阈值系数

        H_z = get_haar_matrix(K_max)
        H_z_T = H_z.T

        pad_size = S + N
        Y_pad = np.pad(Y, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')
        hp, wp = Y_pad.shape

        # 预计算全图所有 8x8 块的 2D DCT 频域表示，加速 3D 变换
        from numpy.lib.stride_tricks import sliding_window_view
        all_patches = sliding_window_view(Y_pad, (N, N))
        # 2D DCT: D @ P @ D^T
        all_dct = np.einsum('ij,xyjk,kl->xyil', dct_mat, all_patches, dct_mat_T, optimize=True)

        # 基础估计
        num_1 = np.zeros_like(Y_pad, dtype=np.float32)
        den_1 = np.zeros_like(Y_pad, dtype=np.float32)
        kaiser_win = np.outer(np.kaiser(N, 2.0), np.kaiser(N, 2.0)).astype(np.float32)

        ref_ys = list(range(pad_size, pad_size + h - N + 1, step))
        if ref_ys[-1] != pad_size + h - N:
            ref_ys.append(pad_size + h - N)
        ref_xs = list(range(pad_size, pad_size + w - N + 1, step))
        if ref_xs[-1] != pad_size + w - N:
            ref_xs.append(pad_size + w - N)

        # 每个参考块匹配位置
        matched_coords = {}

        for ry in ref_ys:
            for rx in ref_xs:
                ref_patch = all_patches[ry, rx]
                y0, y1 = ry - S, ry + S + 1
                x0, x1 = rx - S, rx + S + 1

                # 块匹配：向量化计算搜索窗口内的欧氏距离
                search_patches = all_patches[y0:y1:2, x0:x1:2]
                diff = search_patches - ref_patch
                dists = np.sum(diff * diff, axis=(2, 3))

                flat_idx = np.argpartition(dists.ravel(), K_max - 1)[:K_max]
                sy_idx, sx_idx = np.unravel_index(flat_idx, dists.shape)
                cy = y0 + sy_idx * 2
                cx = x0 + sx_idx * 2
                matched_coords[(ry, rx)] = (cy, cx)

                # 相似块的 2D DCT 堆叠成 3D 组，沿第 3 维做 1D Haar
                group_2d_dct = all_dct[cy, cx]
                group_3d = np.einsum('kz,zij->kij', H_z, group_2d_dct)

                # 3D 硬阈值滤波与非零计数权重
                mask_keep = np.abs(group_3d) > lam_3d
                n_nonzero = np.sum(mask_keep)
                group_3d_filt = group_3d * mask_keep
                weight_ht = (1.0 / (n_nonzero * (sigma_val ** 2))) if n_nonzero >= 1 else (1.0 / (sigma_val ** 2))

                # 逆 3D 变换: 先逆 1D Haar，再逆 2D DCT
                group_inv_1d = np.einsum('zk,kij->zij', H_z_T, group_3d_filt)
                group_spatial = np.einsum('ij,zjk,kl->zil', dct_mat_T, group_inv_1d, dct_mat)

                w_patch = weight_ht * kaiser_win
                for k in range(K_max):
                    py, px = cy[k], cx[k]
                    num_1[py:py + N, px:px + N] += group_spatial[k] * w_patch
                    den_1[py:py + N, px:px + N] += w_patch

        Y_basic_pad = num_1 / (den_1 + 1e-8)
        Y_basic_pad[:pad_size, :] = Y_pad[:pad_size, :]
        Y_basic_pad[pad_size + h:, :] = Y_pad[pad_size + h:, :]
        Y_basic_pad[:, :pad_size] = Y_pad[:, :pad_size]
        Y_basic_pad[:, pad_size + w:] = Y_pad[:, pad_size + w:]

        # 最终估计
        basic_patches = sliding_window_view(Y_basic_pad, (N, N))
        basic_dct = np.einsum('ij,xyjk,kl->xyil', dct_mat, basic_patches, dct_mat_T, optimize=True)

        num_2 = np.zeros_like(Y_pad, dtype=np.float32)
        den_2 = np.zeros_like(Y_pad, dtype=np.float32)
        sigma2 = sigma_val ** 2

        for (ry, rx), (cy, cx) in matched_coords.items():
            # 3D 变换以计算经验维纳收缩系数
            basic_group_2d = basic_dct[cy, cx]
            basic_group_3d = np.einsum('kz,zij->kij', H_z, basic_group_2d)

            energy = basic_group_3d ** 2
            wiener_w = energy / (energy + sigma2)

            # 原始含噪组 3D 变换 / 维纳收缩
            noisy_group_2d = all_dct[cy, cx]
            noisy_group_3d = np.einsum('kz,zij->kij', H_z, noisy_group_2d)
            filt_group_3d = noisy_group_3d * wiener_w

            norm_w2 = np.sum(wiener_w ** 2)
            weight_wie = (1.0 / (norm_w2 * sigma2)) if norm_w2 > 1e-6 else (1.0 / sigma2)

            # 逆 3D 变换还原空间块
            inv_1d = np.einsum('zk,kij->zij', H_z_T, filt_group_3d)
            spatial_patches = np.einsum('ij,zjk,kl->zil', dct_mat_T, inv_1d, dct_mat)

            w_patch = weight_wie * kaiser_win
            for k in range(K_max):
                py, px = cy[k], cx[k]
                num_2[py:py + N, px:px + N] += spatial_patches[k] * w_patch
                den_2[py:py + N, px:px + N] += w_patch

        Y_final = (num_2 / (den_2 + 1e-8))[pad_size:pad_size + h, pad_size:pad_size + w]
        ycrcb[:, :, 0] = np.clip(Y_final, 0, 255)

        # 色度通道 Cr, Cb 快速导向滤波联合消噪，消除彩色噪斑
        Y_guide = np.clip(Y_final / 255.0, 0.0, 1.0)
        for ch in [1, 2]:
            ch_norm = ycrcb[:, :, ch] / 255.0
            ycrcb[:, :, ch] = manual_fast_guided_filter(Y_guide, ch_norm, r=4, eps_val=0.01, s=2) * 255.0

        out_rgb = cv2.cvtColor(np.clip(ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        return out_rgb

    # 功能路由
    if mode == "导向滤波":
        out_float = manual_guided_filter(img_float, img_float, radius, eps)
        output_image = np.clip(out_float * 255.0, 0, 255).astype(np.uint8)

    elif mode == "快速导向滤波":
        out_float = manual_fast_guided_filter(img_float, img_float, radius, eps, subsample)
        output_image = np.clip(out_float * 255.0, 0, 255).astype(np.uint8)

    elif mode == "双边滤波":
        out_float = manual_bilateral_filter(img_float, radius, sigma_s, sigma_r)
        output_image = np.clip(out_float * 255.0, 0, 255).astype(np.uint8)

    elif mode == "图像锐化对比":
        # 图像细节增强公式: I_enhanced = I_base + k * (I - I_base)
        base_bilateral = manual_bilateral_filter(img_float, radius, sigma_s, sigma_r)
        base_guided = manual_guided_filter(img_float, img_float, radius, eps)

        k = float(sharpen_amount)
        sharp_bilateral = np.clip(base_bilateral + k * (img_float - base_bilateral), 0.0, 1.0)
        sharp_guided = np.clip(base_guided + k * (img_float - base_guided), 0.0, 1.0)

        img_b = (sharp_bilateral * 255.0).astype(np.uint8)
        img_g = (sharp_guided * 255.0).astype(np.uint8)

        # 拼接对比
        sep = np.full((img_b.shape[0], 6, 3), 255, dtype=np.uint8)
        output_image = np.concatenate([img_b, sep, img_g], axis=1)

    elif mode == "BM3D算法":
        output_image = manual_bm3d(img_float, sigma_bm3d)

    else:
        output_image = input_image

    return output_image

def function_hw5(input_image, mode="直方图均衡化", clip_limit=2.0, grid_size=8, ref_image=None, color_space="YCrCb亮度通道 (保持色彩)"):
    if input_image is None:
        raise gr.Error('输入错误：在处理之前请先输入一张图像', duration=5)

    # 剥离 Alpha 通道
    if input_image.ndim == 3 and input_image.shape[-1] == 4:
        input_image = input_image[:, :, :3]

    # OpenCV 官方函数调用
    # equalizeHist 实现直方图均衡化
    # ycrcb_cv = cv2.cvtColor(input_image, cv2.COLOR_RGB2YCrCb)
    # ycrcb_cv[:, :, 0] = cv2.equalizeHist(ycrcb_cv[:, :, 0])
    # cv_he_out = cv2.cvtColor(ycrcb_cv, cv2.COLOR_YCrCb2RGB)
    #
    # createCLAHE 实现 CLAHE
    # clahe_obj = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(grid_size), int(grid_size)))
    # ycrcb_clahe = cv2.cvtColor(input_image, cv2.COLOR_RGB2YCrCb)
    # ycrcb_clahe[:, :, 0] = clahe_obj.apply(ycrcb_clahe[:, :, 0])
    # cv_clahe_out = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2RGB)

    # 直方图均衡化
    def manual_hist_equalization_1ch(ch):
        # 0~255 灰度直方图与累积分布函数
        hist = np.bincount(ch.ravel(), minlength=256)
        cdf = np.cumsum(hist)
        # 非零最小值归一化映射，防止全黑偏移
        cdf_nonzero = cdf[cdf > 0]
        if len(cdf_nonzero) == 0 or cdf[-1] == cdf_nonzero[0]:
            return ch.copy()
        cdf_min = cdf_nonzero[0]
        lut = np.round((cdf - cdf_min) / float(cdf[-1] - cdf_min) * 255.0)
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        return lut[ch]

    # CLAHE
    def manual_clahe_1ch(ch, clip_lim=2.0, n_tiles=8):
        h, w = ch.shape
        ny = max(1, int(n_tiles))
        nx = max(1, int(n_tiles))
        clip_lim = max(1.0, float(clip_lim))

        # 高宽能被网格数整除
        pad_h = (ny - h % ny) % ny
        pad_w = (nx - w % nx) % nx
        ch_pad = np.pad(ch, ((0, pad_h), (0, pad_w)), mode='reflect')
        hp, wp = ch_pad.shape
        th, tw = hp // ny, wp // nx
        num_pixels = th * tw

        # 截断阈值
        limit = max(1, int(clip_lim * num_pixels / 256.0))

        # 每个局部块计算对比度受限直方图与 CDF 映射表
        luts = np.zeros((ny, nx, 256), dtype=np.float32)
        for ty in range(ny):
            for tx in range(nx):
                tile = ch_pad[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw]
                hist = np.bincount(tile.ravel(), minlength=256).astype(np.int64)

                # 截断超出阈值的部分，均匀重分配到所有灰度级
                excess = np.maximum(0, hist - limit)
                total_excess = np.sum(excess)
                hist = np.minimum(hist, limit)
                hist += total_excess // 256
                rem = total_excess % 256
                if rem > 0:
                    step = max(1, 256 // rem)
                    idx_rem = np.arange(0, 256, step)[:rem]
                    hist[idx_rem] += 1

                # Tile 累积分布函数
                cdf = np.cumsum(hist).astype(np.float32)
                cdf_min = cdf[cdf > 0][0] if np.any(cdf > 0) else 0.0
                denom = max(1.0, float(num_pixels) - cdf_min)
                luts[ty, tx] = np.clip((cdf - cdf_min) / denom * 255.0, 0.0, 255.0)

        # 块中心对齐的二维双线性插值消除块效应
        y_coords = (np.arange(h, dtype=np.float32) + 0.5) / th - 0.5
        x_coords = (np.arange(w, dtype=np.float32) + 0.5) / tw - 0.5

        ty0 = np.clip(np.floor(y_coords).astype(np.int32), 0, ny - 1)
        ty1 = np.clip(ty0 + 1, 0, ny - 1)
        wy = np.clip(y_coords - np.floor(y_coords), 0.0, 1.0)[:, None]

        tx0 = np.clip(np.floor(x_coords).astype(np.int32), 0, nx - 1)
        tx1 = np.clip(tx0 + 1, 0, nx - 1)
        wx = np.clip(x_coords - np.floor(x_coords), 0.0, 1.0)[None, :]

        Y_idx, X_idx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        val = ch[Y_idx, X_idx]

        # 相邻四个块中心对应的灰度映射值
        v00 = luts[ty0[:, None], tx0[None, :], val]
        v01 = luts[ty0[:, None], tx1[None, :], val]
        v10 = luts[ty1[:, None], tx0[None, :], val]
        v11 = luts[ty1[:, None], tx1[None, :], val]

        # 双线性加权融合
        out = (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * ((1.0 - wx) * v10 + wx * v11)
        return np.clip(np.round(out), 0, 255).astype(np.uint8)

    # 两张图直方图匹配
    def manual_hist_matching_1ch(src_ch, ref_ch):
        # 计算源图与参考图的归一化累积分布函数
        src_hist = np.bincount(src_ch.ravel(), minlength=256)
        ref_hist = np.bincount(ref_ch.ravel(), minlength=256)

        src_cdf = np.cumsum(src_hist).astype(np.float64) / src_ch.size
        ref_cdf = np.cumsum(ref_hist).astype(np.float64) / ref_ch.size

        # 二分查找寻找满足 ref_cdf[z] >= src_cdf[k] 最接近灰度级映射表
        lut = np.searchsorted(ref_cdf, src_cdf, side='left')
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        return lut[src_ch]

    # 功能路由与色彩空间处理
    use_ycrcb = (color_space == "YCrCb亮度通道")

    if mode == "直方图均衡化":
        if input_image.ndim == 2:
            output_image = manual_hist_equalization_1ch(input_image)
        elif use_ycrcb:
            ycrcb = cv2.cvtColor(input_image, cv2.COLOR_RGB2YCrCb)
            ycrcb[:, :, 0] = manual_hist_equalization_1ch(ycrcb[:, :, 0])
            output_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        else:
            output_image = np.stack([manual_hist_equalization_1ch(input_image[:, :, c]) for c in range(3)], axis=-1)

    elif mode == "CLAHE":
        if input_image.ndim == 2:
            output_image = manual_clahe_1ch(input_image, clip_limit, grid_size)
        elif use_ycrcb:
            ycrcb = cv2.cvtColor(input_image, cv2.COLOR_RGB2YCrCb)
            ycrcb[:, :, 0] = manual_clahe_1ch(ycrcb[:, :, 0], clip_limit, grid_size)
            output_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        else:
            output_image = np.stack([manual_clahe_1ch(input_image[:, :, c], clip_limit, grid_size) for c in range(3)], axis=-1)

    elif mode == "直方图匹配":
        if ref_image is None:
            raise gr.Error('输入错误：进行直方图匹配前请先上传一张【参考图像】！', duration=5)
        if ref_image.ndim == 3 and ref_image.shape[-1] == 4:
            ref_image = ref_image[:, :, :3]

        if input_image.ndim == 2:
            ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY) if ref_image.ndim == 3 else ref_image
            output_image = manual_hist_matching_1ch(input_image, ref_gray)
        elif use_ycrcb:
            src_ycrcb = cv2.cvtColor(input_image, cv2.COLOR_RGB2YCrCb)
            ref_ycrcb = cv2.cvtColor(ref_image, cv2.COLOR_RGB2YCrCb)
            src_ycrcb[:, :, 0] = manual_hist_matching_1ch(src_ycrcb[:, :, 0], ref_ycrcb[:, :, 0])
            output_image = cv2.cvtColor(src_ycrcb, cv2.COLOR_YCrCb2RGB)
        else:
            # RGB 三通道独立匹配，迁移参考图的明暗对比度与整体色调风格
            output_image = np.stack([manual_hist_matching_1ch(input_image[:, :, c], ref_image[:, :, c]) for c in range(3)], axis=-1)

    else:
        output_image = input_image

    return output_image