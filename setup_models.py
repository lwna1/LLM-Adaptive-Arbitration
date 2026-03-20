import os
from modelscope.hub.file_download import model_file_download

# 1. 下载并获取精确路径
print("⏳ 开始下载 Qwen2.5-0.5B (约 350MB)...")
path_05b = model_file_download(
    model_id="qwen/Qwen2.5-0.5B-Instruct-GGUF", 
    file_path="qwen2.5-0.5b-instruct-q4_k_m.gguf", 
    revision="master"
)
print(f"✅ 0.5B 下载成功！绝对路径: {path_05b}\n")

print("⏳ 开始下载 Qwen2.5-1.5B (约 1.1GB)...")
path_15b = model_file_download(
    model_id="qwen/Qwen2.5-1.5B-Instruct-GGUF", 
    file_path="qwen2.5-1.5b-instruct-q4_k_m.gguf", 
    revision="master"
)
print(f"✅ 1.5B 下载成功！绝对路径: {path_15b}\n")

# 2. 自动生成 Modelfile
with open("Modelfile_05B", "w") as f:
    f.write(f"FROM {path_05b}\n")
with open("Modelfile_15B", "w") as f:
    f.write(f"FROM {path_15b}\n")

# 3. 自动调用 Ollama 导入
print("⚙️ 正在将 0.5B 模型注册到 Ollama...")
os.system("/usr/local/bin/ollama create qwen-0.5b -f Modelfile_05B")

print("⚙️ 正在将 1.5B 模型注册到 Ollama...")
os.system("/usr/local/bin/ollama create qwen-1.5b -f Modelfile_15B")

# 4. 展示结果
print("\n🎉 导入全部完成！当前 Ollama 中的模型列表：")
os.system("/usr/local/bin/ollama list")