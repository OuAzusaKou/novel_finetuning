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
import deepspeed

def main():
    # 初始化模型和分词器
    model_name = "glm-edge-1.5b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=None  # 使用DeepSpeed时不要设置device_map
    )

    # 加载数据集
    dataset = load_dataset("json", data_files="training_data.json")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=1024
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names  # 移除原始数据集中的所有列
    )

    # DeepSpeed配置
    training_args = TrainingArguments(
        output_dir="./glm_finetuned",
        num_train_epochs=300,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        weight_decay=0,
        lr_scheduler_type="constant",
        logging_steps=10,
        save_steps=100,
        deepspeed="ds_config.json",
        fp16=True,
        local_rank=int(os.getenv("LOCAL_RANK", -1)),
        ddp_backend="nccl"
    )

    # 自定义Trainer类来处理DeepSpeed Zero-3
    # class CustomTrainer(Trainer):
    #     def training_step(self, model, inputs, num_items_in_batch=None):
    #         model.train()
    #         inputs = self._prepare_inputs(inputs)
    #         loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

    #         if self.args.gradient_accumulation_steps > 1:
    #             loss = loss / self.args.gradient_accumulation_steps

    #         if self.is_deepspeed_enabled:
    #             model.backward(loss)
    #         else:
    #             loss.backward()

    #         return loss

    #     def on_step_end(self, args, state, control, **kwargs):
    #         # 在步骤结束时获取学习率
    #         if state.global_step % args.logging_steps == 0:
    #             if self.is_deepspeed_enabled:
    #                 lr = self.deepspeed.lr_scheduler.get_lr()[0]
    #             else:
    #                 lr = self.optimizer.param_groups[0]["lr"]
    #             print(f"Step {state.global_step}, Current learning rate: {lr}")

    # # 初始化CustomTrainer
    # trainer = CustomTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_dataset["train"],
    #     data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    # )
        # 初始化标准Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main() 