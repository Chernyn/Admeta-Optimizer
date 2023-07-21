###example###

cd (your address)

TASK_NAME=stsb

for lr in ...;
do
    for mt in ...;
    do
        CUDA_VISIBLE_DEVICES=0 python run_glue.py   \
            --model_name_or_path bert-large-cased   \
            --task_name $TASK_NAME  \
            --do_train   \
            --do_eval   \
            --max_seq_length 128   \
            --per_device_train_batch_size 32   \
            --num_train_epochs 3  \
            --output_dir ./tmp/admeta_$TASK_NAME/ \
            --overwrite_output_dir \
            --eval_steps 2000 \
            --evaluation_strategy steps \
            --optim admeta \
            --learning_rate $lr  \
            --admeta_momentum $mt | tee ./large_stsb_admeta_logs/${TASK_NAME}_lr=${lr}_mt=${mt}.log
    done    
done
    
