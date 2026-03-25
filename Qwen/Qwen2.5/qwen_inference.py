from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")
if device == 'cuda':
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

# 模型路径（修改为你的实际路径）
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