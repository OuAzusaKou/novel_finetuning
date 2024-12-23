import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from peft import PeftModel, PeftConfig
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint


def load_model(checkpoint_path, device="cuda:3", use_lora=False):
    """
    加载模型
    
    Args:
        checkpoint_path: 模型检查点路径
        device: 使用的设备，默认为"cuda:3"
        use_lora: 是否使用LoRA模型
    """
    # 加载分词器
    model_name = "../ChatGLM3/THUDM/chatglm3-6b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    if use_lora:
        # 加载LoRA配置
        config = PeftConfig.from_pretrained(checkpoint_path)
        # 加载LoRA权重
        model = PeftModel.from_pretrained(model, checkpoint_path)
    else:
        # 加载完整的模型权重
        model = load_state_dict_from_zero_checkpoint(model, checkpoint_path)
    
    # 将模型移动到指定设备
    model = model.to(device)
    model.eval()
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=2048, temperature=0.6, top_p=0.85, repetition_penalty=1.1, no_repeat_ngram_size=3):
    """
    生成回答
    
    Args:
        model: 模型
        tokenizer: 分词器
        prompt: 输入的问题
        max_length: 最大生成长度
        temperature: 温度参数，控制生成的随机性（0.8表示适度随机）
        top_p: top-p采样参数（0.95表示保留累积概率前95%的token）
    """
    message = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        message,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)

    generate_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": max_length,
        "do_sample": True,  # 启用采样
        "temperature": temperature,  # 控制随机性
        "top_p": top_p,    # 控制采样范围
        "repetition_penalty": repetition_penalty,  # 添加重复惩罚
        "no_repeat_ngram_size": no_repeat_ngram_size   # 避免重复的n-gram
    }
    out = model.generate(**generate_kwargs)
    
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

def main():
    # 根据模型类型选择检查点路径和加载方式
    use_lora = True  # 设置是否使用LoRA模型
    if use_lora:
        checkpoint_path = "./glm_lora_finetuned/checkpoint-1700"  # LoRA权重路径
    else:
        checkpoint_path = "./glm_finetuned/checkpoint-300"  # 完整模型检查点路径
    
    try:
        model, tokenizer = load_model(checkpoint_path, use_lora=use_lora)
        print("模型加载完成，开始对话！(输入 'quit' 退出)")
        
        while True:
            user_input = input("\n用户: ")
            if user_input.lower() == 'quit':
                break
                
            try:
                response = generate_response(model, tokenizer, user_input)
                print("\n助手:", response)
            except Exception as e:
                print(f"生成回答时发生错误: {e}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("对话结束！")

if __name__ == "__main__":
    main() 