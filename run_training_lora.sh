deepspeed --num_gpus=3 train_lora.py \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=8 \
    --num_train_epochs=300 \
    --learning_rate=2e-4 \
    --fp16 