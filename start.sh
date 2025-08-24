#!/bin/bash

# PaperPal 启动脚本
echo "🚀 启动 PaperPal - AI论文助手"
echo "================================"

# 检查虚拟环境
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ 虚拟环境已激活: $VIRTUAL_ENV"
else
    echo "⚠️  激活虚拟环境..."
    source venv/bin/activate
fi

# 检查.env文件
if [ ! -f ".env" ]; then
    echo "⚠️  未找到.env文件，请先配置API密钥："
    echo "   cp env_example.txt .env"
    echo "   然后编辑.env文件填入您的API密钥"
    exit 1
fi

echo "🔧 启动Streamlit应用..."
echo "📍 应用地址: http://localhost:8501"
echo "⏹️  按 Ctrl+C 停止应用"
echo "================================"

streamlit run app.py
