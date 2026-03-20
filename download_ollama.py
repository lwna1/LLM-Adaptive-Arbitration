from modelscope.hub.file_download import model_file_download

# 使用 ModelScope 官方维护的二进制镜像
print("正在通过 ModelScope 官方源高速下载 Ollama...")
try:
    file_path = model_file_download(
        model_id='ZhipuAI/ollama-linux', # 借用智谱或官方维护的路径
        file_path='ollama-linux-amd64',
        revision='master'
    )
    print(f"下载成功！文件位于: {file_path}")
except Exception as e:
    print(f"下载失败，请检查网络或路径: {e}")