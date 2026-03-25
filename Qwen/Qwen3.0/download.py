from modelscope import snapshot_download

# 下载 Qwen3-0.6B-Instruct 模型（最小的Qwen3版本）
model_dir = snapshot_download(
    'Qwen/Qwen3-0.6B',           # Qwen3模型ID
    cache_dir='./model_cache'     # 指定下载目录
)

print(f"模型已下载到: {model_dir}")