from modelscope import snapshot_download

# 下载Qwen2.5-0.5B-Instruct模型
model_dir = snapshot_download(
    'qwen/Qwen2.5-0.5B-Instruct',
    cache_dir='./model_cache'  # 指定下载目录
)

print(f"模型已下载到: {model_dir}")