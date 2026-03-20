import os
import glob

# 1. 自动寻找之前下载的 GGUF 模型文件
cache_dir = "/root/.cache/modelscope"
path_05b = glob.glob(f"{cache_dir}/**/*0.5b*q4_k_m.gguf", recursive=True)
path_15b = glob.glob(f"{cache_dir}/**/*1.5b*q4_k_m.gguf", recursive=True)

if not path_05b or not path_15b:
    print("❌ 找不到模型文件，请确认之前是否下载成功。")
    exit(1)

# Qwen 专属的 ChatML 规范对话模板与停止词
qwen_template = """
TEMPLATE \"\"\"{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
\"\"\"
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
"""

# 2. 重新生成标准化的 Modelfile
print("✍️ 正在注入 Qwen 专属对话模板...")
with open("Modelfile_05b_fix", "w") as f:
    f.write(f"FROM {path_05b[0]}\n{qwen_template}")

with open("Modelfile_15b_fix", "w") as f:
    f.write(f"FROM {path_15b[0]}\n{qwen_template}")

# 3. 让 Ollama 重新打包模型 (这会覆盖之前坏掉的模型)
print("⚙️ 正在重新编译 qwen-0.5b...")
os.system("/usr/local/bin/ollama create qwen-0.5b -f Modelfile_05b_fix")

print("⚙️ 正在重新编译 qwen-1.5b...")
os.system("/usr/local/bin/ollama create qwen-1.5b -f Modelfile_15b_fix")

print("\n🎉 修复完成！现在它们知道该怎么好好聊天了。")