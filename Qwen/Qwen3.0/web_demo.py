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
including hate speech, violence, pornography, deception, etc. \
(注：本演示受Qwen3的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)""")

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