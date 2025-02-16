 CUDA_VISIBLE_DEVICES=0 API_PORT=8000 llamafactory-cli api \
    --model_name_or_path models/chatglm3-6b \
    --adapter_name_or_path src/LLaMA-Factory/saves/ChatGLM3-6B-Chat/lora/*** \
    --template chatglm3 \
    --finetuning_type lora