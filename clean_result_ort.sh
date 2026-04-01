#!/bin/bash

# 查找当前目录及子目录下的所有 result_ort 文件夹，并删除其中的图片
# 并且在所有目录中删除 onnx 模型文件
echo "开始清理 result_ort 目录中的图片文件，以及所有目录下的 ONNX 模型文件..."

# 1. 查找名为 result_ort 的目录并删除其中的图片
echo "清理 result_ort 中的图片..."
find . -type d -name "result_ort" | while read -r dir; do
    echo "正在清理图片: $dir"
    find "$dir" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.bmp" \) -exec rm -v {} +
done

# 2. 查找并删除所有的 .onnx 文件
echo "清理工作区中的 ONNX 模型..."
find . -type f -name "*.onnx" -exec rm -v {} +

echo "清理完成！"
