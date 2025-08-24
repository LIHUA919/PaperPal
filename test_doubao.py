#!/usr/bin/env python
"""测试豆包API连接"""

import os
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_doubao_api():
    api_key = os.getenv("DOUBAO_API_KEY")
    endpoint_id = os.getenv("DOUBAO_ENDPOINT_ID")
    
    print(f"API Key: {api_key}")
    print(f"Endpoint ID: {endpoint_id}")
    
    if not api_key or not endpoint_id:
        print("❌ 缺少API配置")
        return
    
    # 构建URL - 使用正确的火山方舟API格式
    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    print(f"URL: {url}")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": endpoint_id,
        "messages": [
            {"role": "user", "content": "你好，请简单介绍一下你自己"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    print("\n正在调用豆包API...")
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API调用成功!")
            print(f"回复: {result.get('choices', [{}])[0].get('message', {}).get('content', '无内容')}")
        else:
            print(f"❌ API调用失败: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ 请求超时")
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    test_doubao_api()
