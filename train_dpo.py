import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType
from trl import DPOTrainer, DPOConfig
from typing import Dict, List

def create_dpo_dataset() -> Dataset:
    """
    创建DPO训练数据集
    返回格式：Dataset包含 {
        "prompt": str,
        "chosen": str,
        "rejected": str
    }
    """
    # 加载原始数据集
    dataset = load_dataset("json", data_files="training_data.json")
    
    # 创建DPO数据集
    dpo_data: List[Dict[str, str]] = []
    texts = dataset["train"]["text"]
    
    for i in range(0, len(texts)-1, 2):
        if i+1 < len(texts):
            # 假设相邻的两个样本，第一个是更好的回答
            dpo_data.append({
                "prompt": "请继续写下去：",
                "chosen": texts[i],
                "rejected": texts[i+1]
            })
    
    # 转换为Dataset格式
    return Dataset.from_list(dpo_data)

def main():
    # 初始化模型和分词器
    model_name = "glm-edge-1.5b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=None
    )

    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=['gate_up_proj','down_proj']
    )
    model = get_peft_model(model, lora_config)
    

    
    # DPO配置
    dpo_config = DPOConfig(
        learning_rate=5e-5,
        weight_decay=0.01,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        max_length=1024,
        max_prompt_length=256,
        num_train_epochs=300,
        neftune_noise_alpha=5,
        output_dir="./glm_dpo_finetuned",
        fp16=True,
        logging_steps=10,
        save_steps=100,
        deepspeed="ds_config_dpo.json",
        local_rank=int(os.getenv("LOCAL_RANK", -1)),
        ddp_backend="nccl",
        remove_unused_columns=False,
        report_to="none"
    )

    # 创建DPO数据集
    train_dataset = create_dpo_dataset()
    
    # 初始化DPO训练器
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # 使用同一个模型作为参考
        args=dpo_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # 开始训练
    dpo_trainer.train()
    
    # 保存模型
    dpo_trainer.save_model("./glm_dpo_weights")

if __name__ == "__main__":
    main() 