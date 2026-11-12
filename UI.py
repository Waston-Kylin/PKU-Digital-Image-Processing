#-*-coding:utf-8 -*-
from __future__ import annotations

import argparse
import pathlib
import gradio as gr
import numpy as np

def create_demo_hw1(process):
    with gr.Blocks() as demo:
        gr.Markdown("## 作业一: 图像色彩综合处理工具")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='输入图像')
                mode = gr.Radio(["色相/饱和度", "亮度/对比度", "OpenCV HLS"], label="模式", value="色相/饱和度")
                h_slider = gr.Slider(minimum=-1, maximum=1, label="色相", value=0, step=0.01)
                s_slider = gr.Slider(minimum=-1, maximum=1, label="饱和度", value=0, step=0.01)
                l_slider = gr.Slider(minimum=-1, maximum=1, label="明度", value=0, step=0.01)
                brightness = gr.Slider(minimum=-1, maximum=1, label="亮度", value=0, step=0.01, visible=False)
                contrast = gr.Slider(minimum=-1, maximum=1, label="对比度", value=0, step=0.01, visible=False)
            with gr.Column():
                with gr.Row():
                    output_image = gr.Image(type='numpy', label='输出图像', interactive=False)
                with gr.Row():
                    r_channel = gr.Image(type='numpy', label='R通道', interactive=False)
                    g_channel = gr.Image(type='numpy', label='G通道', interactive=False)
                    b_channel = gr.Image(type='numpy', label='B通道', interactive=False)
                with gr.Row():
                    h_channel = gr.Image(type='numpy', label='H通道', interactive=False)
                    s_channel = gr.Image(type='numpy', label='S通道', interactive=False)
                    l_channel = gr.Image(type='numpy', label='L通道', interactive=False)

        input_image.change(fn=process, 
                           inputs=[input_image, mode, h_slider, s_slider, l_slider, brightness, contrast], 
                           outputs=[output_image, r_channel, g_channel, b_channel, h_channel, s_channel, l_channel])

        def update_mode(mode):
            if mode == "亮度/对比度":
                return (
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=True), gr.update(visible=True)
                )
            else:
                return (
                    gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
                    gr.update(visible=False), gr.update(visible=False)
                )
        mode.change(update_mode, inputs=[mode], outputs=[h_slider, s_slider, l_slider, brightness, contrast])

        def slider_update(slider):
            return slider.release(
                fn=process, 
                inputs=[input_image, mode, h_slider, s_slider, l_slider, brightness, contrast], 
                outputs=[output_image, r_channel, g_channel, b_channel, h_channel, s_channel, l_channel]
            )

        slider_update(h_slider)
        slider_update(s_slider)
        slider_update(l_slider)
        slider_update(brightness)
        slider_update(contrast)

    return demo

def create_demo_hw2(process):
    import gradio as gr
    with gr.Blocks() as demo:
        gr.Markdown('## 作业二: 基于DCGAN的图像生成与编辑') 

        # 图像生成
        with gr.Tab("图像生成功能"):
            with gr.Row():
                with gr.Column():
                    seed_gen = gr.Number(value=42, label='随机种子 (Seed)', precision=0)
                    btn_gen = gr.Button("生成图像", variant="primary")
                with gr.Column():
                    out_gen = gr.Image(label='输出图像', interactive=False)
            
            # 参数对齐到 process 接口
            btn_gen.click(fn=lambda s: process("generate", s, 0, "", "", 0)[0], inputs=[seed_gen], outputs=[out_gen])

        # 图像编辑
        with gr.Tab("图像编辑功能"):
            gr.Markdown("### 生成 64 张候选图")
            with gr.Row():
                with gr.Column():
                    seed_grid = gr.Number(value=42, label='候选图生成种子 (Seed)', precision=0)
                    gr.Markdown("""
                    **图像编号排布（1~64）：**
                    从左到右，从上到下顺序递增。
                    - **第 1 行**:  1, 2, 3, 4, 5, 6, 7, 8
                    - **第 2 行**:  9, 10, 11, 12, 13, 14, 15, 16
                    - ......
                    """)
                    btn_grid = gr.Button("生成 64 张图", variant="primary")
                with gr.Column():
                    out_grid = gr.Image(label="64 张候选图像 (点击右上角放大查看)", interactive=False)
            btn_grid.click(fn=lambda sg: process("grid", 0, sg, "", "", 0)[0], inputs=[seed_grid], outputs=[out_grid])

            gr.Markdown("### 寻找相反性质图像并提取编辑向量")
            with gr.Row():
                with gr.Column():
                    pos_input = gr.Textbox(value="1", label="具有 [+A] 特征的图像编号 (1~64，若多对空格分隔)")
                    neg_input = gr.Textbox(value="2", label="具有 [-A] 特征的图像编号 (1~64，若多对空格分隔)")
                    btn_calc = gr.Button("计算 v_A 并预览选择", variant="primary")
                with gr.Column():
                    with gr.Row():
                        out_pos = gr.Image(label="+A 样本预览", interactive=False)
                        out_neg = gr.Image(label="-A 样本预览", interactive=False)
            btn_calc.click(fn=lambda sg, p, n: process("calc", 0, sg, p, n, 0), inputs=[seed_grid, pos_input, neg_input], outputs=[out_pos, out_neg])

            gr.Markdown("### 输入新噪声 z，用向量生成编辑版本 z+v_A")
            with gr.Row():
                with gr.Column():
                    seed_edit = gr.Number(value=42, label='新噪声随机种子 (Seed)', precision=0)
                    intensity = gr.Slider(minimum=-5.0, maximum=5.0, value=1.0, step=0.1, label="编辑强度 α (α=1 即标准编辑)")
                    btn_edit = gr.Button("生成编辑对比图", variant="primary")
                with gr.Column():
                    out_edit = gr.Image(label="左侧: 原图 z，右侧: 编辑后 z + α*v_A", interactive=False)
            btn_edit.click(fn=lambda se, i: process("edit", se, 0, "", "", i)[0], inputs=[seed_edit, intensity], outputs=[out_edit])

    return demo

def create_demo_hw3(process):
    with gr.Blocks() as demo:
        gr.Markdown("## 作业三: 替换屏幕内容工具")

        video_frame_state = gr.State(None)
        
        with gr.Tabs():
            # 静态图像替换
            with gr.Tab("静态图像屏幕替换"):
                with gr.Row():
                    with gr.Column():
                        bg_img = gr.ImageEditor(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='背景图像 (请在此标出屏幕的 4 个顶点位置)', brush=gr.Brush(default_size=4, colors=["#FF0000"], color_mode="fixed"), eraser=False, layers=False)
                        reset_btn = gr.Button('重新选择顶点位置', size="sm")
                        fg_img = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='前景展示图像')
                        
                    with gr.Column():
                        with gr.Row():
                            mode = gr.Radio(choices=['Nearest', 'Bilinear', 'Bicubic'], value='Bilinear', label='像素插值算法')
                            calc_impl = gr.Radio(choices=['OpenCV 官方函数', '手动实现（矩阵计算+像素插值）'], value='OpenCV 官方函数', label='几何实现方式')
                        
                        run_btn_img = gr.Button('替换屏幕内容为前景图像', variant="primary")
                        out_img = gr.Image(type='numpy', label='输出合成图像', interactive=False)
                
                def reset_points(b): return b["background"] if b else None
                reset_btn.click(fn=reset_points, inputs=[bg_img], outputs=[bg_img])
                
                def wrap_img(b, f, m, c):
                    return process("image", b, f, m, c, None)
                run_btn_img.click(fn=wrap_img, inputs=[bg_img, fg_img, mode, calc_impl], outputs=[out_img])

            # 视频追踪替换
            with gr.Tab("视频屏幕目标追踪替换"):
                gr.Markdown("用 Lucas-Kanade 稀疏光流法追踪视频中笔记本屏幕的 4 个特征角点并逐帧贴图透视变换")
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="上传底图视频 (.mp4)", format='mp4')
                        video_fg = gr.Image(sources=['upload', 'webcam', 'clipboard'],
                                            type='numpy',
                                            label='前景展示图像')
                        
                    with gr.Column():
                        video_first_frame = gr.ImageEditor(
                            sources=[],
                            type='numpy', 
                            label='自动提取视频第 1 帧，请在此标出屏幕的 4 个顶点位置', 
                            brush=gr.Brush(default_size=4, colors=["#FF0000"], color_mode="fixed"), 
                            eraser=False, 
                            layers=False
                        )
                        video_reset_btn = gr.Button('重新标点', size="sm")
                        run_btn_video = gr.Button('追踪并生成完整替换视频', variant="primary")
                        out_video = gr.Video(label='输出替换后视频', interactive=False)
                        
                video_reset_btn.click(fn=reset_points, inputs=[video_first_frame], outputs=[video_first_frame])
                
                def extract_first_frame(video_path):
                    if video_path is None:
                        return None
                    import cv2
                    cap = cv2.VideoCapture(video_path)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        return {
                            "background": frame_rgb,
                            "layers": [],
                            "composite": frame_rgb
                        }
                    return None

                # 视频上传
                video_input.change(fn=extract_first_frame, inputs=[video_input], outputs=[video_first_frame])

                def reset_video_points(b):
                    if b and b.get("background") is not None:
                        bg = b["background"]
                        return {
                            "background": bg,
                            "layers": [],
                            "composite": bg
                        }
                    return None
                video_reset_btn.click(fn=reset_video_points, inputs=[video_first_frame], outputs=[video_first_frame])
                
                def wrap_video(b, f, v):
                    return process("video", b, f, "Bilinear", "OpenCV官方函数", v)
                run_btn_video.click(fn=wrap_video, inputs=[video_first_frame, video_fg, video_input], outputs=[out_video])

    return demo

def create_demo_hw4(process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业四: 图像去噪与保边滤波工具') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='输入图像')
                mode = gr.Radio(
                    choices=[
                        "导向滤波",
                        "快速导向滤波",
                        "双边滤波",
                        "图像锐化对比",
                        "BM3D算法"
                    ],
                    value="导向滤波",
                    label="算法模式选择"
                )
                
                radius = gr.Slider(minimum=1, maximum=15, value=4, step=1, label="滤波窗口半径 r", visible=True)
                eps = gr.Slider(minimum=0.001, maximum=0.2, value=0.02, step=0.001, label="导向滤波正则化参数 eps", visible=True)
                subsample = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="快速导向滤波下采样率 s", visible=False)
                sigma_s = gr.Slider(minimum=0.5, maximum=15.0, value=3.0, step=0.5, label="双边滤波空间域标准差 sigma_s", visible=False)
                sigma_r = gr.Slider(minimum=0.01, maximum=0.5, value=0.1, step=0.01, label="双边滤波值域标准差 sigma_r", visible=False)
                sharpen_amount = gr.Slider(minimum=1.0, maximum=5.0, value=2.5, step=0.1, label="细节增强/锐化倍数 k", visible=False)
                sigma_bm3d = gr.Slider(minimum=5.0, maximum=50.0, value=25.0, step=1.0, label="BM3D 噪声标准差估计 sigma", visible=False)

            with gr.Column():
                output_image = gr.Image(type='numpy', label='输出图像 (对比模式：左为双边滤波，右为导向滤波)', interactive=False)
                run_button = gr.Button(value='运行', variant="primary")

        def update_hw4_mode(selected_mode):
            if selected_mode == "导向滤波":
                return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]
            elif selected_mode == "快速导向滤波":
                return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]
            elif selected_mode == "双边滤波":
                return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)]
            elif selected_mode == "图像锐化对比":
                return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=False),
                        gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)]
            else: # BM3D
                return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)]

        mode.change(
            fn=update_hw4_mode,
            inputs=[mode],
            outputs=[radius, eps, subsample, sigma_s, sigma_r, sharpen_amount, sigma_bm3d]
        )

        run_button.click(
            fn=process,
            inputs=[input_image, mode, radius, eps, sigma_s, sigma_r, subsample, sigma_bm3d, sharpen_amount],
            outputs=[output_image]
        )
    return demo

def create_demo_hw5(process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业五: 图像提亮与直方图增强工具') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='输入待处理图像')
                mode = gr.Radio(
                    choices=[
                        "直方图均衡化",
                        "CLAHE",
                        "直方图匹配"
                    ],
                    value="直方图均衡化",
                    label="算法模式选择"
                )
                color_space = gr.Radio(
                    choices=[
                        "YCrCb亮度通道",
                        "RGB三通道独立处理"
                    ],
                    value="YCrCb亮度通道",
                    label="作用通道模式"
                )
                
                # CLAHE参数
                clip_limit = gr.Slider(minimum=1.0, maximum=10.0, value=2.0, step=0.5, label="CLAHE 对比度限制阈值", visible=False)
                grid_size = gr.Slider(minimum=2, maximum=16, value=8, step=1, label="CLAHE 网格划分尺寸", visible=False)
                
                # 直方图匹配参考图
                ref_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='参考图像 ', visible=False)

            with gr.Column():
                output_image = gr.Image(type='numpy', label='输出图像', interactive=False)
                run_button = gr.Button(value='运行', variant="primary")

        def update_hw5_mode(selected_mode):
            if selected_mode == "直方图均衡化":
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value="YCrCb亮度通道")
            elif selected_mode == "CLAHE":
                return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(value="YCrCb亮度通道")
            else:  # 直方图匹配
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(value="RGB三通道独立处理")

        mode.change(
            fn=update_hw5_mode,
            inputs=[mode],
            outputs=[clip_limit, grid_size, ref_image, color_space]
        )

        run_button.click(
            fn=process,
            inputs=[input_image, mode, clip_limit, grid_size, ref_image, color_space],
            outputs=[output_image]
        )
    return demo