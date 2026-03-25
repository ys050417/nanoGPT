from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 自动判断 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用设备：", device)

# 模型路径（和下载路径一致）
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