import os
import time
import requests
import sys

# --- 配置区 ---
# 强制使用 127.0.0.1，彻底避开 localhost 的 IPv6 解析黑洞
API_URL = "http://127.0.0.1:11434"
TEST_MODEL = "qwen-0.5b"
LOG_FILE = "/root/test/ollama_service.log"

def kill_existing_ollama():
    print("🧹 [1/4] 正在清理历史 Ollama 进程...")
    os.system("sudo pkill -9 ollama > /dev/null 2>&1")
    time.sleep(2)

def start_ollama_server():
    print("🚀 [2/4] 正在启动 Ollama 服务端 (使用原生 Bash 强注入环境变量)...")
    # 直接使用之前我们手动验证过 100% 成功的 Bash 启动命令！
    start_cmd = f"OLLAMA_NO_AVX2=1 OLLAMA_NO_AVX=1 nohup /usr/local/bin/ollama serve > {LOG_FILE} 2>&1 &"
    os.system(start_cmd)
    time.sleep(3) # 留出初始化时间

def wait_for_ready(timeout=30):
    print("⏳ [3/4] 等待 API 端口就绪", end="", flush=True)
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # 使用 127.0.0.1 进行探测
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

def test_model_inference():
    print(f"🧠 [4/4] 正在对 {TEST_MODEL} 发起 API 接口推理测试...")
    
    payload = {
        "model": TEST_MODEL,
        "prompt": "你好，请回复收到。",
        "stream": False
    }
    
    try:
        start_time = time.time()
        # API 测试：设置 60 秒超时（由于是小模型，只要不卡死，通常几秒钟就会返回）
        response = requests.post(f"{API_URL}/api/generate", json=payload, timeout=60).json()
        latency = time.time() - start_time
        
        reply = response.get("response", "").strip()
        print("\n" + "="*50)
        print(f"✅ 测试大成功! (API 响应耗时: {latency:.2f}秒)")
        print(f"🤖 模型回复: {reply}")
        print("="*50 + "\n")
        
    except requests.exceptions.Timeout:
        print("\n❌ API 请求再次超时！")
        print("🔍 自动为您抓取后台崩溃日志的最后 15 行：")
        os.system(f"tail -n 15 {LOG_FILE}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生未知错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("\n=== 初始化 LLM 运行环境 ===")
    kill_existing_ollama()
    start_ollama_server()
    
    if wait_for_ready():
        test_model_inference()
        print("🎉 所有底层系统运作正常！")