from __future__ import annotations

import os
import argparse
import pathlib
import gradio as gr
from UI import *
from functions import *


HTML_DESCRIPTION = '''
<div align=center>
<h1 style="font-weight: 900; margin-bottom: 7px;">
   图像处理网页演示工具</a>
</h1>
<p>使用方式：在浏览器中打开http://127.0.0.1:8088/即可</p>
<p>作者: Rivélan</p>
</div>
'''
MD_DESCRIPTION = '''
## 此网页演示提供以下图像处理工具:
- 作业1：图像色相/饱和度/亮度/对比度调整工具（手动 RGB-HSL 转换、拟合 PS 亮度对比度曲线、自然饱和度）
- 作业2：基于GAN的图像生成编辑工具（支持 Seed 生成、64宫格预览、特征向量 vA 提取与可视化编辑）
- 作业3：替换屏幕内容工具（单应性计算矩阵、双线性/双三次插值，及视频光流追踪渲染)
- 作业4：图像去噪工具（手动导向滤波、盒式滤波优化快速导向滤波、双边滤波、图像锐化对比及 BM3D 算法）
- 作业5：图像提亮工具（手动实现直方图均衡化、限制对比度自适应直方图均衡化 CLAHE 及双图直方图匹配）
'''

def main():
    os.environ['no_proxy'] = 'localhost, 127.0.0.1, ::1'
    os.environ['NO_PROXY'] = 'localhost, 127.0.0.1, ::1'
    with gr.Blocks(css='style.css') as demo:
        gr.Markdown(HTML_DESCRIPTION) 
        gr.Markdown(MD_DESCRIPTION)
        with gr.Tabs():
            with gr.TabItem('作业1: 图像色相/饱和度/亮度/对比度调整工具'):
                create_demo_hw1(function_hw1)          
            with gr.TabItem('作业2: 基于GAN的图像生成编辑工具'):
                create_demo_hw2(function_hw2)   
            with gr.TabItem('作业3: 替换屏幕内容工具'):
                create_demo_hw3(function_hw3)  
            with gr.TabItem('作业4: 图像去噪工具'):
                create_demo_hw4(function_hw4) 
            with gr.TabItem('作业5: 图像提亮工具'):
                create_demo_hw5(function_hw5)                                    

    demo.launch(server_port=8088, inbrowser=True)

if __name__ == '__main__':
    main()