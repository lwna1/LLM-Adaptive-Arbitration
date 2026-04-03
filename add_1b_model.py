import os
from modelscope.hub.file_download import model_file_download

print("⏳ 正在通过阿里内网高速下载 Llama-3.2-1B-Instruct (约 800MB)...")
try:
    # 【已修复】更换为 second-state 维护的 100% 存在的官方搬运库
    path_1b = model_file_download(
        model_id="second-state/Llama-3.2-1B-Instruct-GGUF",
        file_path="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        revision="master"
    )
    print(f"✅ 下载成功！文件路径: {path_1b}\n")
except Exception as e:
    print(f"❌ 下载失败: {e}")
    exit(1)

# Llama 3 专属的系统对话模板 (防止模型胡言乱语)
llama3_template = """
TEMPLATE \"\"\"<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

\"\"\"
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
"""

print("✍️ 正在生成 Modelfile 并注入模板...")
with open("Modelfile_1B", "w") as f:
    f.write(f"FROM {path_1b}\n{llama3_template}")

print("⚙️ 正在将 1B 模型注册到 Ollama...")
os.system("/usr/local/bin/ollama create llama-1b -f Modelfile_1B")

print("\n🎉 三大基座集结完毕！当前 Ollama 模型列表：")
os.system("/usr/local/bin/ollama list")