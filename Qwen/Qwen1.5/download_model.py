from modelscope import snapshot_download

# 下载到当前项目目录，不会占C盘
model_dir = snapshot_download(
    "qwen/Qwen2.5-0.5B-Chat",
    cache_dir="./"
)
print("模型下载完成，路径：", model_dir)