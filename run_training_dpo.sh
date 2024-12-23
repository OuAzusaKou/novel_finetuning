deepspeed --num_gpus=4 train_dpo.py \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=8 \
    --num_train_epochs=300 \
    --learning_rate=5e-5 \
    --fp16 