import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)

def main():
    # 初始化模型和分词器
    model_name = "glm-edge-1.5b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=None  # 使用DeepSpeed时不要设置device_map
    )

    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                     # LoRA 秩
        lora_alpha=32,          # LoRA alpha参数
        lora_dropout=0.1,       # Dropout概率
        bias="none",
        # target_modules=['q_proj', 'v_proj', 'k_proj'],  # 需要根据模型结构调整
        target_modules=['gate_up_proj','down_proj']
    )

    # 准备模型进行LoRA训练
    model = prepare_model_for_kbit_training(model)
    print(model)
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数信息
    model.print_trainable_parameters()

    # 加载数据集
    dataset = load_dataset("json", data_files="training_data.json")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=1024  # 与数据处理保持一致
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./glm_lora_finetuned",
        num_train_epochs=300,          # 减少训练轮数
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,          # LoRA通常使用更大的学习率
        weight_decay=0.01,           # 添加权重衰减
        lr_scheduler_type="cosine",  # 使用cosine学习率调度
        logging_steps=10,
        save_steps=100,
        deepspeed="ds_config_lora.json",
        fp16=True,
        local_rank=int(os.getenv("LOCAL_RANK", -1)),
        ddp_backend="nccl"
    )

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # 开始训练
    trainer.train()
    
    # 保存LoRA权重
    model.save_pretrained("./glm_lora_weights")

if __name__ == "__main__":
    main() 