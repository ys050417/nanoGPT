# Qwen1.5、Qwen2.5、Qwen3.0 开源大语言模型学习

## 一、案例简介

​		Qwen作为覆盖入门、进阶、高阶的开源大语言模型学习案例，结合不同学习者的硬件条件差异，分别选用适配各阶段的Qwen系列模型权重作为学习核心。本案例将学习三个模型：

- Qwen1.5-0.5B（5亿参数量）：可在普通个人电脑上流畅运行，帮助学习者掌握模型加载、基础对话应用构建及ChatBot核心逻辑，同时其多参数量版本支持阶梯式进阶，初步接触大模型核心技术；
- Qwen2.5-0.5B（5亿参数量，适配普通个人电脑）：该版本在1.5版本基础上升级，支持长文本处理、优化数学推理与代码生成能力，助力学习者巩固基础技能的同时，掌握模型性能优化、多语言交互等进阶内容；
- Qwen3.0-0.6B（6亿参数量，适配普通个人电脑）：其采用MoE架构，支持混合推理，在多方面能力大幅提升，可帮助深入学习高阶技术，掌握智能体开发、部署优化等技能，具备独立开发高阶大模型应用的能力。

## 二、Qwen1.5-0.5B模型

### （一）创建环境

```
conda create -n qwen1.5-gpu python=3.8
```

```
conda activate qwen1.5-gpu
```

### （二）安装 GPU 版 PyTorch

- 检验CUDA 版本

```
nvidia-smi
```

- 安装对应 PyTorch

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### （三）安装相关库

```
pip install transformers>=4.32
pip install accelerate>=0.26.0
pip install modelscope
pip install gradio
```

### （四）下载模型

- 创建download.py

```python
from modelscope import snapshot_download

# 下载到当前项目目录，不会占C盘
model_dir = snapshot_download(
    "qwen/Qwen1.5-0.5B-Chat",
    cache_dir="./"
)
print("模型下载完成，路径：", model_dir)
```

### （五）加载模型

- 创建run_qwen_gpu.py
- model_path 模型路径需要修改成下载路径

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 自动判断 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用设备：", device)

# 模型路径
model_path = r"E:\python\大模型应用开发\qianwen1.5\qwen\Qwen1.5-0.5B-Chat"

# 加载模型（自动用 GPU）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 对话输入
prompt = "简单介绍一下你自己"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# 构造对话模板
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 转为模型输入
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# 生成回答
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)

# 截断输入部分，只保留模型生成内容
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 解码输出
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("\n回答：", response)
```

### （六） 下载Qwen1.5

- 方法一：进入github网站下载（需要使用加速器，可以使用steam++）
- 下载网址

```
https://github.com/QwenLM/Qwen3/tree/v1.5
```

- 方法二：pycharm中直接下载官方 Qwen1.5 项目

```
git clone https://github.com/QwenLM/Qwen1.5.git
```

### （七）创建web_demo.py

- DEFAULT_CKPT_PATH路径需要修改为模型下载路径，与model_path 模型路径保持一致

```python
# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""

from argparse import ArgumentParser
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

DEFAULT_CKPT_PATH =  r"E:\python\大模型应用开发\qianwen1.5\qwen\Qwen1.5-0.5B-Chat"


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")

    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=8000,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="127.0.0.1",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        torch_dtype="auto",
        device_map=device_map,
        resume_download=True,
    ).eval()
    model.generation_config.max_new_tokens = 2048   # For chat.

    return model, tokenizer


def _chat_stream(model, tokenizer, query, history):
    conversation = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
    ]
    for query_h, response_h in history:
        conversation.append({'role': 'user', 'content': query_h})
        conversation.append({'role': 'assistant', 'content': response_h})
    conversation.append({'role': 'user', 'content': query})
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors='pt',
    )
    inputs = inputs.to(model.device)
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=inputs,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _launch_demo(args, model, tokenizer):

    def predict(_query, _chatbot, _task_history):
        print(f"User: {_query}")
        _chatbot.append((_query, ""))
        full_response = ""
        response = ""
        for new_text in _chat_stream(model, tokenizer, _query, history=_task_history):
            response += new_text
            _chatbot[-1] = (_query, response)

            yield _chatbot
            full_response = response

        print(f"History: {_task_history}")
        _task_history.append((_query, full_response))
        print(f"Qwen1.5-Chat: {full_response}")

    def regenerate(_chatbot, _task_history):
        if not _task_history:
            yield _chatbot
            return
        item = _task_history.pop(-1)
        _chatbot.pop(-1)
        yield from predict(item[0], _chatbot, _task_history)

    def reset_user_input():
        return gr.update(value="")

    def reset_state(_chatbot, _task_history):
        _task_history.clear()
        _chatbot.clear()
        _gc()
        return _chatbot

    with gr.Blocks() as demo:
        gr.Markdown("""\
<p align="center"><img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/logo_qwen1.5.jpg" style="height: 80px"/><p>""")
        gr.Markdown("""<center><font size=8>Qwen1.5-Chat Bot</center>""")
        gr.Markdown(
            """\
<center><font size=3>This WebUI is based on Qwen1.5-Chat, developed by Alibaba Cloud. \
(本WebUI基于Qwen1.5-Chat打造，实现聊天机器人功能。)</center>""")
        gr.Markdown("""\
<center><font size=4>
Qwen1.5-7B <a href="https://modelscope.cn/models/qwen/Qwen1.5-7B/summary">🤖 </a> | 
<a href="https://huggingface.co/Qwen/Qwen1.5-7B">🤗</a>&nbsp ｜ 
Qwen1.5-7B-Chat <a href="https://modelscope.cn/models/qwen/Qwen1.5-7B-Chat/summary">🤖 </a> | 
<a href="https://huggingface.co/Qwen/Qwen1.5-7B-Chat">🤗</a>&nbsp ｜ 
Qwen1.5-14B <a href="https://modelscope.cn/models/qwen/Qwen1.5-14B/summary">🤖 </a> | 
<a href="https://huggingface.co/Qwen/Qwen1.5-14B">🤗</a>&nbsp ｜ 
Qwen1.5-14B-Chat <a href="https://modelscope.cn/models/qwen/Qwen1.5-14B-Chat/summary">🤖 </a> | 
<a href="https://huggingface.co/Qwen/Qwen1.5-14B-Chat">🤗</a>&nbsp ｜ 
&nbsp<a href="https://github.com/QwenLM/Qwen1.5">Github</a></center>""")

        chatbot = gr.Chatbot(label='Qwen1.5-Chat', elem_classes="control-height")
        query = gr.Textbox(lines=2, label='Input')
        task_history = gr.State([])

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History (清除历史)")
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")

        submit_btn.click(predict, [query, chatbot, task_history], [chatbot], show_progress=True)
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(reset_state, [chatbot, task_history], outputs=[chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)

        gr.Markdown("""\
<font size=2>Note: This demo is governed by the original license of Qwen1.5. \
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
including hate speech, violence, pornography, deception, etc. """)

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()

    model, tokenizer = _load_model_tokenizer(args)

    _launch_demo(args, model, tokenizer)


if __name__ == '__main__':
    main()
```

### （八）运行访问

- 运行web_demo.py后进行访问
- 访问地址

```
http://localhost:8000
```

## 三、Qwen2.5-0.5B模型

### （一）创建环境

```
conda create -n Qwen2.5 python=3.10
```

```
conda activate Qwen2.5 
```

### （二） 安装GPU 版 PyTorch

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### （三）安装相关库

- 打开PyCharm的终端，执行命令
- 安装transformers

```
pip install transformers
```

- 安装modelscope

```
pip install modelscope
```

- 安装gradio

```
pip install gradio
```

- 安装加速库（可选，提升性能）

```
pip install accelerate
```

### （四）下载模型

- 创建 `download_model.py` 

```python
from modelscope import snapshot_download

# 下载Qwen2.5-0.5B-Instruct模型
model_dir = snapshot_download(
    'qwen/Qwen2.5-0.5B-Instruct',
    cache_dir='./model_cache'  # 指定下载目录
)

print(f"模型已下载到: {model_dir}")
```

### （五）加载模型

- 创建qwen_inference.py
- model_path模型路径需要修改为实际路径

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")
if device == 'cuda':
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

# 模型路径
model_path = r'E:\python\大模型应用开发\qianwen2.5\model_cache\qwen\Qwen2.5-0.5B-Instruct'

# 检查模型是否存在
if not os.path.exists(model_path):
    print(f"错误: 模型路径不存在: {model_path}")
    print("请先运行 download_model.py 下载模型")
    exit(1)

print("正在加载模型...")
# 加载模型和tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("模型加载完成！")


def chat_with_qwen(prompt, max_new_tokens=512):
    """与Qwen模型对话的函数"""
    # 构建消息格式
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 编码输入
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 生成回复
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.7,  # 控制随机性
        top_p=0.9,  # 核采样
        do_sample=True  # 启用采样
    )

    # 提取生成的回复（去除输入部分）
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 解码回复
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# 测试对话
if __name__ == "__main__":
    print("\n=== Qwen2.5-0.5B ChatBot ===\n")
    print("输入 'quit' 退出对话\n")

    while True:
        user_input = input("用户: ")
        if user_input.lower() == 'quit':
            break

        response = chat_with_qwen(user_input)
        print(f"Qwen: {response}\n")
```

### （六）下载Qwen2.5

- 进入github网站下载（需要使用加速器，可以使用steam++）
- 下载网址

```
https://github.com/QwenLM/Qwen3/tree/v2.5
```

### （七）创建web_demo.py

- DEFAULT_CKPT_PATH路径需要修改为模型下载路径，与model_path 模型路径保持一致

```python
# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""

from argparse import ArgumentParser
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

DEFAULT_CKPT_PATH = r'E:\python\大模型应用开发\qianwen2.5\model_cache\qwen\Qwen2.5-0.5B-Instruct'


def _get_args():
    parser = ArgumentParser(description="Qwen2.5-Instruct web chat demo.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default=DEFAULT_CKPT_PATH,
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )

    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a publicly shareable link for the interface.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        default=False,
        help="Automatically launch the interface in a new tab on the default browser.",
    )
    parser.add_argument(
        "--server-port", type=int, default=8000, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="127.0.0.1", help="Demo server name."
    )

    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    # 修复1: 移除 resume_download=True 参数
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,  # 添加 trust_remote_code
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    # 修复2: 移除 resume_download=True 参数
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        torch_dtype="auto",
        device_map=device_map,
        trust_remote_code=True,  # 添加 trust_remote_code
    ).eval()
    model.generation_config.max_new_tokens = 2048  # For chat.

    return model, tokenizer


def _chat_stream(model, tokenizer, query, history):
    conversation = []
    for query_h, response_h in history:
        conversation.append({"role": "user", "content": query_h})
        conversation.append({"role": "assistant", "content": response_h})
    conversation.append({"role": "user", "content": query})
    input_text = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True
    )
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text


def _gc():
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _launch_demo(args, model, tokenizer):
    def predict(_query, _chatbot, _task_history):
        print(f"User: {_query}")
        _chatbot.append((_query, ""))
        full_response = ""
        response = ""
        for new_text in _chat_stream(model, tokenizer, _query, history=_task_history):
            response += new_text
            _chatbot[-1] = (_query, response)

            yield _chatbot
            full_response = response

        print(f"History: {_task_history}")
        _task_history.append((_query, full_response))
        print(f"Qwen: {full_response}")

    def regenerate(_chatbot, _task_history):
        if not _task_history:
            yield _chatbot
            return
        item = _task_history.pop(-1)
        _chatbot.pop(-1)
        yield from predict(item[0], _chatbot, _task_history)

    def reset_user_input():
        return gr.update(value="")

    def reset_state(_chatbot, _task_history):
        _task_history.clear()
        _chatbot.clear()
        _gc()
        return _chatbot

    with gr.Blocks() as demo:
        gr.Markdown("""\
<p align="center"><img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/assets/logo/qwen2.5_logo.png" style="height: 120px"/><p>""")
        gr.Markdown(
            """\
<center><font size=3>This WebUI is based on Qwen2.5-Instruct, developed by Alibaba Cloud. \
(本WebUI基于Qwen2.5-Instruct打造，实现聊天机器人功能。)</center>"""
        )
        gr.Markdown("""\
<center><font size=4>
Qwen2.5-0.5B-Instruct <a href="https://modelscope.cn/models/qwen/Qwen2.5-0.5B-Instruct/summary">🤖 </a> | 
<a href="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct">🤗</a>&nbsp ｜ 
Qwen2.5-7B-Instruct <a href="https://modelscope.cn/models/qwen/Qwen2.5-7B-Instruct/summary">🤖 </a> | 
<a href="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct">🤗</a>&nbsp ｜ 
Qwen2.5-32B-Instruct <a href="https://modelscope.cn/models/qwen/Qwen2.5-32B-Instruct/summary">🤖 </a> | 
<a href="https://huggingface.co/Qwen/Qwen2.5-32B-Instruct">🤗</a>&nbsp ｜ 
Qwen2.5-72B-Instruct <a href="https://modelscope.cn/models/qwen/Qwen2.5-72B-Instruct/summary">🤖 </a> | 
<a href="https://huggingface.co/Qwen/Qwen2.5-72B-Instruct">🤗</a>&nbsp ｜ 
&nbsp<a href="https://github.com/QwenLM/Qwen2.5">Github</a></center>""")

        chatbot = gr.Chatbot(label="Qwen", elem_classes="control-height")
        query = gr.Textbox(lines=2, label="Input")
        task_history = gr.State([])

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History (清除历史)")
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")

        submit_btn.click(
            predict, [query, chatbot, task_history], [chatbot], show_progress=True
        )
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(
            reset_state, [chatbot, task_history], outputs=[chatbot], show_progress=True
        )
        regen_btn.click(
            regenerate, [chatbot, task_history], [chatbot], show_progress=True
        )

        gr.Markdown("""\
<font size=2>Note: This demo is governed by the original license of Qwen2.5. \
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
including hate speech, violence, pornography, deception, etc. """)

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()

    # 检查模型路径是否存在
    import os
    if not os.path.exists(args.checkpoint_path):
        print(f"错误: 模型路径不存在: {args.checkpoint_path}")
        print("请检查模型路径是否正确")
        print("当前设置的路径:", args.checkpoint_path)
        return

    print(f"正在加载模型: {args.checkpoint_path}")
    print(f"使用设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    model, tokenizer = _load_model_tokenizer(args)
    print("模型加载成功！")
    print(f"启动Web界面，访问地址: http://{args.server_name}:{args.server_port}")

    _launch_demo(args, model, tokenizer)


if __name__ == "__main__":
    main()
```

### （八）运行访问

- 运行web_demo.py后进行访问
- 访问地址

```
http://localhost:8000
```

## 四、Qwen3.0-0.6B模型

### （一）创建环境

```
winget install Python.Python.3.12
```

### （二）安装GPU 版 PyTorch

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### （三）安装相关库

- PyCharm 终端直接运行：

```
pip install transformers accelerate sentencepiece protobuf
```

```
pip install numpy pandas pillow
```

### （四）下载模型

- 创建download.py

```python
from modelscope import snapshot_download

# 下载 Qwen3-0.6B-Instruct 模型（最小的Qwen3版本）
model_dir = snapshot_download(
    'Qwen/Qwen3-0.6B',           # Qwen3模型ID
    cache_dir='./model_cache'     # 指定下载目录
)

print(f"模型已下载到: {model_dir}")
```

### （五）加载模型

- 创建loading .py
- model_path模型路径需要修改为实际路径

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")
if device == 'cuda':
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

# 模型路径
model_path = r'E:\python\大模型应用开发\qianwen3.0\model_cache\Qwen\Qwen3-0.6B'

# 检查模型是否存在
if not os.path.exists(model_path):
    print(f"错误: 模型路径不存在: {model_path}")
    print("请先运行 download_model.py 下载模型")
    exit(1)

print("正在加载模型...")
# 加载模型和tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("模型加载完成！")


def chat_with_qwen(prompt, max_new_tokens=512):
    """与Qwen模型对话的函数"""
    # 构建消息格式
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 编码输入
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 生成回复
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.7,  # 控制随机性
        top_p=0.9,  # 核采样
        do_sample=True  # 启用采样
    )

    # 提取生成的回复（去除输入部分）
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 解码回复
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# 测试对话
if __name__ == "__main__":
    print("\n=== Qwen2.5-0.5B ChatBot ===\n")
    print("输入 'quit' 退出对话\n")

    while True:
        user_input = input("用户: ")
        if user_input.lower() == 'quit':
            break

        response = chat_with_qwen(user_input)
        print(f"Qwen: {response}\n")
```

### （六）下载Qwen3.0

- 进入github网站下载（需要使用加速器，可以使用steam++）
- 下载网址

```
https://github.com/QwenLM/Qwen3/tree/v3.0
```

### （七）创建web_demo.py

- DEFAULT_CKPT_PATH路径需要修改为模型下载路径，与model_path 模型路径保持一致

```
# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""

from argparse import ArgumentParser
from threading import Thread
import traceback
import re

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

DEFAULT_CKPT_PATH = r"E:\python\大模型应用开发\qianwen3.0\model_cache\Qwen\Qwen3-0.6B"


def _get_args():
    parser = ArgumentParser(description="Qwen3-Instruct web chat demo.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default=DEFAULT_CKPT_PATH,
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        default=True,
        help="Disable thinking mode (default: True)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a publicly shareable link for the interface.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        default=False,
        help="Automatically launch the interface in a new tab on the default browser.",
    )
    parser.add_argument(
        "--server-port", type=int, default=8000, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="127.0.0.1", help="Demo server name."
    )

    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    print(f"正在加载模型: {args.checkpoint_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint_path,
            trust_remote_code=True,
        )
        print("✓ 分词器加载成功")
    except Exception as e:
        print(f"✗ 分词器加载失败: {e}")
        raise

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_path,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True,
        ).eval()
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        raise

    model.generation_config.max_new_tokens = 2048

    # 打印模型信息
    print(f"模型设备: {model.device}")
    print(f"模型数据类型: {model.dtype}")

    return model, tokenizer


def _format_messages(history):
    """格式化消息为Qwen3要求的格式"""
    messages = []
    for msg in history:
        # history 现在是字典列表格式
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                messages.append({
                    "role": "user",
                    "content": msg.get("content", "")
                })
            elif msg.get("role") == "assistant":
                messages.append({
                    "role": "assistant",
                    "content": msg.get("content", "")
                })
    return messages


def _remove_think_tags(text):
    """移除思考标签内容"""
    # 移除 <think>...</think> 标签及其内容
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 移除可能的空白行
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()


def _chat_stream(model, tokenizer, query, history, disable_thinking=True):
    try:
        # 构建对话历史
        messages = _format_messages(history)

        # 添加当前用户消息
        if query and query.strip():
            messages.append({
                "role": "user",
                "content": query.strip()
            })

        print(f"消息数量: {len(messages)}")

        # 使用 tokenizer 的聊天模板
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 如果禁用思考模式，添加指令
        if disable_thinking:
            # 在系统消息中添加禁用思考的指令
            # 注意：这需要在消息列表开头添加系统消息
            system_msg = {"role": "system",
                          "content": "You are a helpful assistant. Do not use <think> tags in your response. Answer directly without thinking process."}
            messages_with_system = [system_msg] + messages
            input_text = tokenizer.apply_chat_template(
                messages_with_system,
                tokenize=False,
                add_generation_prompt=True
            )

        print(f"输入文本前100字符: {input_text[:100]}...")

        # 编码输入
        inputs = tokenizer(
            [input_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)

        # 创建流式生成器
        streamer = TextIteratorStreamer(
            tokenizer=tokenizer,
            skip_prompt=True,
            timeout=60.0,
            skip_special_tokens=True
        )

        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "streamer": streamer,
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # 用于累积响应，以便移除思考标签
        accumulated_response = ""

        for new_text in streamer:
            if new_text:
                accumulated_response += new_text
                # 实时移除思考标签
                cleaned_response = _remove_think_tags(accumulated_response)
                if cleaned_response:
                    yield cleaned_response

    except Exception as e:
        error_msg = f"生成错误: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        yield f"[错误] {str(e)}"


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _launch_demo(args, model, tokenizer):
    def predict(_query, _chatbot, _task_history):
        if not _query or _query.strip() == "":
            yield _chatbot
            return

        print(f"\n{'=' * 50}")
        print(f"用户: {_query}")
        print(f"{'=' * 50}")

        # 添加用户消息（使用字典格式）
        _chatbot.append({"role": "user", "content": _query})
        # 添加空的助手消息用于流式更新
        _chatbot.append({"role": "assistant", "content": ""})

        full_response = ""
        response = ""

        try:
            for new_text in _chat_stream(model, tokenizer, _query, history=_task_history,
                                         disable_thinking=args.disable_thinking):
                if new_text:
                    response = new_text  # 直接使用清理后的文本
                    # 检查是否有错误
                    if response.startswith("[错误]"):
                        _chatbot[-1] = {"role": "assistant", "content": response}
                        yield _chatbot
                        return
                    # 更新助手回复
                    _chatbot[-1] = {"role": "assistant", "content": response}
                    yield _chatbot
                    full_response = response

        except Exception as e:
            error_msg = f"生成出错: {str(e)}"
            print(error_msg)
            _chatbot[-1] = {"role": "assistant", "content": error_msg}
            yield _chatbot
            return

        print(f"Qwen3 响应长度: {len(full_response)} 字符")
        # 保存到历史记录（保存为字典格式）
        _task_history.append({"role": "user", "content": _query})
        _task_history.append({"role": "assistant", "content": full_response})

        # 清理显存
        _gc()

    def regenerate(_chatbot, _task_history):
        if len(_task_history) < 2:
            yield _chatbot
            return
        # 移除最后两条消息（用户和助手）
        _task_history.pop()  # 移除助手回复
        last_user = _task_history.pop()  # 移除用户消息
        # 移除聊天记录中的最后两条
        _chatbot.pop()  # 移除助手回复
        _chatbot.pop()  # 移除用户消息
        # 重新生成
        yield from predict(last_user["content"], _chatbot, _task_history)

    def reset_user_input():
        return gr.update(value="")

    def reset_state(_chatbot, _task_history):
        _task_history.clear()
        _chatbot.clear()
        _gc()
        return _chatbot

    # 创建界面
    with gr.Blocks(title="Qwen3 Chat Demo") as demo:
        # 使用文字 Logo
        gr.Markdown("""\
<p align="center">
<font size=6><b>✨ Qwen3 ✨</b></font><br>
<font size=3>通义千问 3.0 (已禁用思考模式)</font>
</p>
""")

        gr.Markdown(
            """\
<center><font size=3>This WebUI is based on Qwen3-Instruct, developed by Alibaba Cloud. \
(本WebUI基于Qwen3-Instruct打造，实现聊天机器人功能。)</center>"""
        )

        gr.Markdown("""\
<center><font size=4>
<a href="https://modelscope.cn/models/Qwen/Qwen3-0.6B/summary">🤖 Qwen3-0.6B</a> | 
<a href="https://modelscope.cn/models/Qwen/Qwen3-1.7B/summary">🤖 Qwen3-1.7B</a> | 
<a href="https://modelscope.cn/models/Qwen/Qwen3-8B/summary">🤖 Qwen3-8B</a> | 
<a href="https://modelscope.cn/models/Qwen/Qwen3-32B/summary">🤖 Qwen3-32B</a> | 
<a href="https://github.com/QwenLM/Qwen3">📚 Github</a>
</center>""")

        # Chatbot 使用默认格式
        chatbot = gr.Chatbot(
            label="Qwen3",
            height=500
        )
        query = gr.Textbox(
            lines=2,
            label="Input",
            placeholder="请输入您的问题...",
            interactive=True
        )
        task_history = gr.State([])

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History (清除历史)", variant="secondary")
            submit_btn = gr.Button("🚀 Submit (发送)", variant="primary")
            regen_btn = gr.Button("🤔️ Regenerate (重试)", variant="secondary")

        submit_btn.click(
            predict, [query, chatbot, task_history], [chatbot], show_progress=True
        )
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(
            reset_state, [chatbot, task_history], outputs=[chatbot], show_progress=True
        )
        regen_btn.click(
            regenerate, [chatbot, task_history], [chatbot], show_progress=True
        )

        # 添加回车提交功能
        query.submit(
            predict, [query, chatbot, task_history], [chatbot], show_progress=True
        )
        query.submit(reset_user_input, [], [query])

        gr.Markdown("""\
<font size=2>Note: This demo is governed by the original license of Qwen3. \
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
including hate speech, violence, pornography, deception, etc. """)

    # 打印启动信息
    print(f"\n🚀 启动 Qwen3 Web 界面...")
    print(f"📱 本地访问地址: http://{args.server_name}:{args.server_port}")
    if args.share:
        print("🌐 将创建公共链接...")

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()

    # 打印设备信息
    print("=" * 50)
    print("Qwen3 Web Demo 启动中...")
    print("=" * 50)
    print(f"Python 版本: {__import__('sys').version}")
    print(f"Transformers 版本: {__import__('transformers').__version__}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"Gradio 版本: {gr.__version__}")
    print(f"禁用思考模式: {args.disable_thinking}")

    if torch.cuda.is_available():
        print(f"✅ 使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ 使用 CPU 运行（速度较慢）")
    print("=" * 50)

    try:
        model, tokenizer = _load_model_tokenizer(args)
        print("✅ 模型加载完成！")
        print("=" * 50)
        print("开始启动 Web 界面...")
        print("=" * 50)

        _launch_demo(args, model, tokenizer)

    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print(traceback.format_exc())
        input("按回车键退出...")


if __name__ == "__main__":
    main()
```

### （八）运行访问

- 运行web_demo.py后进行访问
- 访问地址

```
http://localhost:8000
```

