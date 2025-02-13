# CORA

This repository contains the dataset and code for our paper: "CORA: A Cognitive Reframing Dialogue Agent Powered by Large Language Models".

## Enviroment Setup

The basic environment settings are as follows:

- Operating System: Ubuntu 18.04
- CUDA: 12.4
- Python: 3.10.0
- LLaMA-Factory: 0.9.2

You can install the python dependencies through the following command:

```
pip install -r requirements.txt
```

## Dataset Access

Due to ethical considerations, the raw data files of our Sim-CBT dataset have been encrypted. Only a limited numer of example dialogues are provided in the `data/samples.json` file for demonstration purpose. Researchers interested in accessing the complete dataset are required to adhere to the following conditions to obtain the dataset password:

1. Obtain authorization from the source data provider of the [C2D2 dataset](https://github.com/bcwangavailable/C2D2-Cognitive-Distortion).
2. Agree to restrict the use of the dataset to academic purposes only, and ensure that it will not be shared or disclosed to any third parties without explicit permission.
3. Submit the source data license and a signed data usage statement to our designated email address for verification.

## Running the Code

- Download the [ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b) model and place it in the `models/` directory.

- Set up the LLaMA-Factory environment following the instructions provided in the [LLaMA-Factory repository](https://github.com/hiyouga/LLaMA-Factory), and navigate to the LLaMA-Factory directory by executing: `cd src/LLaMA-Factory`.

- Place the JSON files from our dataset into the `**/LLaMA-Factory/data/` directory, and update the data configuration in `**/LLaMA-Factory/data/dataset_info.json`.

- To initiate model training, execute the following command:

  ```
  llamafactory-cli train \
      --stage sft \
      --do_train True \
      --model_name_or_path [base_model_path] \
      --preprocessing_num_workers 16 \
      --finetuning_type lora \
      --template chatglm3 \
      --flash_attn auto \
      --dataset_dir data \
      --dataset [data_name] \
      --cutoff_len 1024 \
      --learning_rate 0.0003 \
      --num_train_epochs 5.0 \
      --max_samples 100000 \
      --per_device_train_batch_size 4 \
      --gradient_accumulation_steps 8 \
      --lr_scheduler_type cosine \
      --max_grad_norm 1.0 \
      --logging_steps 5 \
      --save_steps 100 \
      --warmup_steps 0 \
      --optim adamw_torch \
      --packing False \
      --report_to none \
      --output_dir [save_model_path] \
      --bf16 True \
      --plot_loss True \
      --ddp_timeout 180000000 \
      --include_num_input_tokens_seen True \
      --lora_rank 8 \
      --lora_alpha 16 \
      --lora_dropout 0.05 \
      --loraplus_lr_ratio 16 \
      --lora_target all 
  ```

- To perform model evaluation, execute the following command:

  ```
  llamafactory-cli train \
      --stage sft \
      --model_name_or_path [base_model_path] \
      --preprocessing_num_workers 16 \
      --finetuning_type lora \
      --quantization_method bitsandbytes \
      --template chatglm3 \
      --flash_attn auto \
      --dataset_dir data \
      --eval_dataset [eval_data_name] \
      --cutoff_len 1024 \
      --max_samples 100000 \
      --per_device_eval_batch_size 16 \
      --predict_with_generate True \
      --max_new_tokens 512 \
      --top_p 0.7 \
      --temperature 0.95 \
      --output_dir [eval_result_path] \
      --do_predict True \
      --adapter_name_or_path [save_model_path]
  ```

- To interact with the trained model, execute the following command:

  ```
  llamafactory-cli chat \
      --model_name_or_path [base_model_path] \
      --adapter_name_or_path  [save_model_path] \
      --template default \
      --finetuning_type lora
  ```

Additionally, the trained model can be accessed via a website interface located in the `web/` directory.
