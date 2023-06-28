# cd /data1/users/chenyineng/admeta-transformers/examples/pytorch/text-classification

# TASK_NAME=stsb

# for lr in 8e-4 5e-4 6e-4;
# do
#     for mt in 0.02 0.03 0.05 0.08 0.1 0.2 0.3;
#     do
#         CUDA_VISIBLE_DEVICES=0 python run_glue.py   \
#             --model_name_or_path bert-large-cased   \
#             --task_name $TASK_NAME  \
#             --do_train   \
#             --do_eval   \
#             --max_seq_length 128   \
#             --per_device_train_batch_size 32   \
#             --num_train_epochs 3  \
#             --output_dir ./tmp/admeta_$TASK_NAME/ \
#             --overwrite_output_dir \
#             --eval_steps 2000 \
#             --evaluation_strategy steps \
#             --optim admeta \
#             --learning_rate $lr  \
#             --admeta_momentum $mt | tee ./large_stsb_admeta_logs/${TASK_NAME}_lr=${lr}_mt=${mt}.log
#     done    
# done
    
# cd /data1/users/chenyineng/admeta-transformers/examples/pytorch/question-answering
# for lr in 4e-4;
# do
#     for mt in 0.05;
#     do
#         CUDA_VISIBLE_DEVICES=2,3,4 python run_qa.py \
#           --model_name_or_path bert-base-uncased \
#           --dataset_name squad \
#           --do_train \
#           --do_eval \
#           --per_device_train_batch_size 12 \
#           --num_train_epochs 2 \
#           --max_seq_length 384 \
#           --doc_stride 128 \
#           --output_dir ./tmp/debug_squad/ \
#           --overwrite_output_dir \
#           --eval_steps 800 \
#           --evaluation_strategy steps \
#           --optim admeta \
#           --learning_rate $lr  \
#           --admeta_momentum $mt | tee ./test/lr=${lr}_mt=${mt}.log
#     done
# done

cd /data1/users/chenyineng/admeta-transformers/examples/pytorch/question-answering
for lr in 4e-4;
do
    for mt in 0.3;
    do
        CUDA_VISIBLE_DEVICES=6,7 python run_qa.py \
          --model_name_or_path bert-large-uncased \
          --dataset_name squad_v2 \
          --version_2_with_negative \
          --do_train \
          --do_eval \
          --per_device_train_batch_size 12 \
          --num_train_epochs 2 \
          --max_seq_length 384 \
          --doc_stride 128 \
          --output_dir ./tmp/debug_squad_v2_${lr}_${mt}/ \
          --overwrite_output_dir \
          --eval_steps 2000 \
          --evaluation_strategy steps \
          --optim admeta \
          --learning_rate $lr  \
          --admeta_momentum $mt | tee ./large_q-a2v_admeta_logs/lr=${lr}_mt=${mt}.log
    done
done


# CUDA_VISIBLE_DEVICES=2,3,4 python run_qa.py \
#     --model_name_or_path bert-base-uncased \
#     --dataset_name squad_v2 \
#     --do_train \
#     --do_eval \
#     --version_2_with_negative \
#     --per_device_train_batch_size 12 \
#     --num_train_epochs 2 \
#     --max_seq_length 384 \
#     --doc_stride 128 \
#     --output_dir ./tmp/debug_squadv2/ \
#     --overwrite_output_dir \
#     --eval_steps 800 \
#     --evaluation_strategy steps \
#     --optim admeta \
#     --learning_rate $lr  \
#     --admeta_momentum $mt | tee ./test/lr=${lr}_mt=${mt}.log


# cd /data1/users/chenyineng/admeta-transformers/examples/pytorch/semantic-segmentation
# for lr in 1.5e-4 2e-4;
# do
#     for mt in 0.2;
#     do
#         CUDA_VISIBLE_DEVICES=1 python run_semantic_segmentation.py \
#             --model_name_or_path nvidia/mit-b0 \
#             --dataset_name segments/sidewalk-semantic \
#             --output_dir ./tmp/segformer_outputs/ \
#             --overwrite_output_dir \
#             --remove_unused_columns False \
#             --do_train \
#             --do_eval \
#             --evaluation_strategy steps \
#             --max_steps 10000 \
#             --learning_rate 0.00006 \
#             --lr_scheduler_type polynomial \
#             --per_device_train_batch_size 8 \
#             --per_device_eval_batch_size 8 \
#             --logging_strategy steps \
#             --logging_steps 100 \
#             --evaluation_strategy epoch \
#             --save_strategy epoch \
#             --seed 1337 \
#             --optim admeta \
#             --learning_rate $lr  \
#             --admeta_momentum $mt | tee ./seg_admeta_logs/lr=${lr}_mt=${mt}.log
#     done
# done

# cd /data1/users/chenyineng/admeta-transformers/examples/pytorch/semantic-segmentation
# for lr in 5e-4;
# do
#     for mt in 0.1;
#     do 
#         CUDA_VISIBLE_DEVICES=4 python run_semantic_segmentation.py \
#             --model_name_or_path nvidia/mit-b0 \
#             --dataset_name segments/sidewalk-semantic \
#             --output_dir ./tmp/segformer_outputs/ \
#             --overwrite_output_dir \
#             --remove_unused_columns False \
#             --do_train \
#             --do_eval \
#             --evaluation_strategy steps \
#             --max_steps 10000 \
#             --lr_scheduler_type polynomial \
#             --per_device_train_batch_size 8 \
#             --per_device_eval_batch_size 8 \
#             --logging_strategy steps \
#             --logging_steps 100 \
#             --evaluation_strategy epoch \
#             --save_strategy epoch \
#             --seed 1337 \
#             --optim admeta \
#             --learning_rate $lr \
#             --admeta_momentum $mt | tee ./seg_admeta1to0.8_logs/lr=${lr}_mt=${mt}.log
#     done
# done



# cd /data1/users/chenyineng/admeta-transformers/examples/pytorch/audio-classification
# for lr in 8e-4 1e-3;
# do
#     for mt in 0.02 0.05 0.08;
#     do
#         CUDA_VISIBLE_DEVICES=2 python run_audio_classification.py \
#             --model_name_or_path facebook/wav2vec2-base \
#             --dataset_name superb \
#             --dataset_config_name ks \
#             --output_dir wav2vec2-base-ft-keyword-spotting_admeta_lr=$lr_mt=$mt \
#             --overwrite_output_dir \
#             --remove_unused_columns False \
#             --do_train \
#             --do_eval \
#             --fp16 \
#             --max_length_seconds 1 \
#             --attention_mask False \
#             --warmup_ratio 0.1 \
#             --num_train_epochs 5 \
#             --per_device_train_batch_size 32 \
#             --gradient_accumulation_steps 4 \
#             --per_device_eval_batch_size 32 \
#             --dataloader_num_workers 4 \
#             --logging_strategy steps \
#             --logging_steps 10 \
#             --evaluation_strategy epoch \
#             --save_strategy epoch \
#             --load_best_model_at_end True \
#             --metric_for_best_model accuracy \
#             --save_total_limit 3 \
#             --seed 0 \
#             --optim admeta \
#             --learning_rate $lr \
#             --admeta_momentum $mt | tee ./audio_admeta_logs/lr=${lr}_mt=${mt}.log
#     done
# done

# cd /data1/users/chenyineng/admeta-transformers/examples/pytorch/audio-classification
# for lr in 2e-3;
# do
#     for mt in 0.2;
#     do
#         CUDA_VISIBLE_DEVICES=4,5,6,7 python run_audio_classification.py \
#             --model_name_or_path facebook/wav2vec2-base \
#             --dataset_name common_language \
#             --audio_column_name audio \
#             --label_column_name language \
#             --output_dir wav2vec2-base-lang-id_admeta_$lr_$mt \
#             --overwrite_output_dir \
#             --remove_unused_columns False \
#             --do_train \
#             --do_eval \
#             --fp16 \
#             --max_length_seconds 16 \
#             --attention_mask False \
#             --warmup_ratio 0.1 \
#             --num_train_epochs 10 \
#             --per_device_train_batch_size 8 \
#             --gradient_accumulation_steps 4 \
#             --per_device_eval_batch_size 1 \
#             --dataloader_num_workers 8 \
#             --logging_strategy steps \
#             --logging_steps 10 \
#             --evaluation_strategy epoch \
#             --save_strategy epoch \
#             --load_best_model_at_end True \
#             --metric_for_best_model accuracy \
#             --save_total_limit 3 \
#             --seed 0 \
#             --optim admeta \
#             --learning_rate $lr \
#             --admeta_momentum $mt | tee ./audio4_admeta_logs/lr=${lr}_mt=${mt}.log
#     done
# done