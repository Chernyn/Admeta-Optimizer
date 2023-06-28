TASK_NAME=stsb
CUDA_VISIBLE_DEVICES=0 python run_glue.py   \
    --model_name_or_path bert-base-cased   \
    --task_name $TASK_NAME  \
    --do_train   \
    --do_eval   \
    --max_seq_length 128   \
    --per_device_train_batch_size 32   \
    --num_train_epochs 3  \
    --output_dir ./tmp/admetar_$TASK_NAME/ \
    --overwrite_output_dir \
    --eval_steps 400 \
    --evaluation_strategy steps \
    --optim admetar \
    --learning_rate 7e-4  \
    --admetar_lamda 0.02



CUDA_VISIBLE_DEVICES=0,1,2 python run_qa.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name squad \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 12 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./tmp/debug_squad/ \
    --overwrite_output_dir \
    --eval_steps 400 \
    --evaluation_strategy steps \
    --optim admetar \
    --learning_rate 4e-4  \
    --admetar_lamda 0.05



CUDA_VISIBLE_DEVICES=0,1 python run_qa.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name squad_v2 \
    --version_2_with_negative \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 12 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./tmp/debug_squad_v2 \
    --overwrite_output_dir \
    --eval_steps 400 \
    --evaluation_strategy steps \
    --optim admetar \
    --learning_rate 3e-4  \
    --admetar_lamda 0.2



CUDA_VISIBLE_DEVICES=0 python3 run_ner.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name conll2003 \
    --do_train \
    --do_eval \
    --output_dir ./tmp/test-ner/ \
    --overwrite_output_dir \
    --eval_steps 400 \
    --evaluation_strategy steps \
    --optim admetar \
    --learning_rate 2e-4  \
    --admetar_lamda 0.3



CUDA_VISIBLE_DEVICES=0 python run_audio_classification.py \
    --model_name_or_path facebook/wav2vec2-base \
    --dataset_name superb \
    --dataset_config_name ks \
    --output_dir wav2vec2-base-ft-keyword-spotting \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --fp16 \
    --max_length_seconds 1 \
    --attention_mask False \
    --warmup_ratio 0.1 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 32 \
    --dataloader_num_workers 4 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --save_total_limit 3 \
    --seed 0 \
    --optim admetar \
    --learning_rate 5e-4 \
    --admetar_lamda 0.05



CUDA_VISIBLE_DEVICES=0,1,2,3 python run_audio_classification.py \
    --model_name_or_path facebook/wav2vec2-base \
    --dataset_name common_language \
    --audio_column_name audio \
    --label_column_name language \
    --output_dir wav2vec2-base-lang-id \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --fp16 \
    --max_length_seconds 16 \
    --attention_mask False \
    --warmup_ratio 0.1 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 1 \
    --dataloader_num_workers 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --save_total_limit 3 \
    --seed 0 \
    --optim admetar \
    --learning_rate 2e-3 \
    --admetar_lamda 0.2