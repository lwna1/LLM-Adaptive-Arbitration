import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

import time
import requests
import sys

# --- 配置区 ---
API_URL = "http://127.0.0.1:11434"
# 将原本的单模型改为模型列表（按参数量从小到大测试）
TEST_MODELS = ["qwen-0.5b", "llama-1b", "qwen-1.5b"]
LOG_FILE = "ollama_service.log"

def kill_existing_ollama():
    print("🧹 [1/4] 正在清理历史 Ollama 进程...")
    os.system("sudo pkill -9 ollama > /dev/null 2>&1")
    time.sleep(2)

def start_ollama_server():
    print("🚀 [2/4] 正在启动 Ollama 服务端 (注入防死锁环境变量)...")
    start_cmd = f"OLLAMA_NO_AVX2=1 OLLAMA_NO_AVX=1 nohup /usr/local/bin/ollama serve > {LOG_FILE} 2>&1 &"
    os.system(start_cmd)
    time.sleep(3) 

def wait_for_ready(timeout=30):
    print("⏳ [3/4] 等待 API 端口就绪", end="", flush=True)
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{API_URL}/api/tags", timeout=2)
            if response.status_code == 200:
                print(" [就绪!]")
                return True
        except requests.exceptions.ConnectionError:
            pass
        
        print(".", end="", flush=True)
        time.sleep(2)
        
    print("\n❌ 启动超时，请检查服务日志。")
    return False

def test_all_models():
    print("🧠 [4/4] 正在对三大基座模型依次发起推理测试...")
    print("⚠️  注意：由于涉及模型在内存中的加载与切换，整体耗时可能在一两分钟左右，请耐心等待。\n")
    print("="*60)
    
    for model_name in TEST_MODELS:
        print(f"▶️ 正在呼叫 {model_name:<12} ...", end="", flush=True)
        
        payload = {
            "model": model_name,
            "prompt": "你好，请回复收到。",
            "stream": False
        }
        
        try:
            start_time = time.time()
            # 设置较长的超时时间，应对模型的云盘加载与内存置换
            response = requests.post(f"{API_URL}/api/generate", json=payload, timeout=180).json()
            latency = time.time() - start_time
            
            reply = response.get("response", "").strip()
            # 去除可能产生的换行符，让输出更整洁
            reply_clean = reply.replace("\n", " ")
            
            print(f" [成功! 耗时: {latency:>5.2f}s] -> 回复: {reply_clean}")
            
        except requests.exceptions.Timeout:
            print(f"\n❌ 请求超时！模型 {model_name} 加载失败。")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ 测试 {model_name} 时发生未知错误: {e}")
            sys.exit(1)
            
    print("="*60 + "\n")

if __name__ == "__main__":
    print("\n=== 初始化 LLM 三级架构运行环境 ===")
    kill_existing_ollama()
    start_ollama_server()
    
    if wait_for_ready():
        test_all_models()
        print("🎉 所有底层基座运作正常！端侧大语言模型自适应仲裁平台已就绪。")