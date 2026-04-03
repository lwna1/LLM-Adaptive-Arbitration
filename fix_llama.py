import os
import glob

# 1. 查找已下载的 Llama 1B 模型文件
cache_dir = "/root/.cache/modelscope"
gguf_files = glob.glob(f"{cache_dir}/**/*.gguf", recursive=True)
path_1b = [f for f in gguf_files if "1b" in f.lower() and "llama" in f.lower()]

if not path_1b:
    print("❌ 找不到 Llama 1B 模型文件，请检查之前是否下载成功。")
    exit(1)

# 2. 带有强制 System Prompt 的改进版 Llama-3 模板
llama3_strict_template = """
TEMPLATE \"\"\"{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

\"\"\"
SYSTEM \"\"\"你是一个高效、极其精简的AI执行程序。必须严格、字面地执行用户的指令。不要擅自补充上下文，不要输出任何客服话术，不要中英夹杂，不要提供解释。\"\"\"
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
"""

# 3. 生成新的 Modelfile 并重新打包覆盖
print("✍️ 正在生成带有强制系统提示词的 Modelfile...")
with open("Modelfile_1B_fix", "w") as f:
    f.write(f"FROM {path_1b[0]}\n{llama3_strict_template}")

print("⚙️ 正在向 Ollama 重新注册并覆盖 llama-1b (注入紧箍咒)...")
os.system("/usr/local/bin/ollama create llama-1b -f Modelfile_1B_fix")

print("\n🎉 Llama-1B 修复完成！它现在学会闭嘴了。")