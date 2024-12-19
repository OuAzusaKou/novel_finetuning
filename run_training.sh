deepspeed --num_gpus=3 train.py \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --num_train_epochs=300 \
    --learning_rate=1e-4 \
    --fp16 