# quick_test.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型路径
model_path = "/root/autodl-tmp/model/Llama-3.1-8B-Instruct"

print("=== 开始测试 ===")

try:
    # 1. 加载tokenizer
    print("1. 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("   ✓ Tokenizer加载成功")
    
    # 2. 加载模型
    print("2. 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("   ✓ 模型加载成功")
    
    # 3. 测试推理
    print("3. 测试推理...")
    
    prompt = "请用中文回答：什么是人工智能？"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = inputs.to('cuda')
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\n问: {prompt}")
    print(f"答: {response}")
    print("\n✓ 模型运行正常！")
    
except Exception as e:
    print(f"\n✗ 出错: {e}")