export CUDA_VISIBLE_DEVICES=0

model_name=TriDM
folder_name="${model_name}"

echo "start training..."

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/"$folder_name ]; then
    mkdir ./logs/$folder_name
fi

seq_len=96
label_len=48

for model_name in TriDM
do
for pred_len in 96 192 336 720
do  
    python -u runETT.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ${model_name}_ETTm1_${seq_len}_${pred_len} \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --grid_size 5 \
  --train_epochs 20 \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --gpu 0 > logs/${folder_name}/${model_name}_ETTm1_${seq_len}_${pred_len}.log 
   
done
done

