for sl in '7e-5' '5e-5'
do
		echo ${sl}
		python3.7 MAESC_training.py \
          --dataset twitter17 ./src/data/jsons/twitter17_info.json \
          --checkpoint_dir mytrain/checkpoints/twitter17 \
          --model_config config/pretrain_base.json \
          --log_dir mytrain/log/twitter17 \
          --num_beams 4 \
          --eval_every 1 \
          --lr ${sl} \
          --batch_size 32  \
          --epochs 150 \
          --grad_clip 5 \
          --warmup 0.1 \
          --seed 68 \
          --checkpoint /data/liuxj/aspect_sentiment_detect/pretrain_checkpoint/pretrain/pytorch_model.bin \
          --rank 2 \
          --trc_pretrain_file /data/liuxj/aspect_sentiment_detect/pretrain_checkpoint/trc_pretrain/pytorch_model.bin \
          --nn_attention_on \
          --nn_attention_mode 0 \
          --trc_on \
          --gcn_on \
          --dep_mode 2 \
          --sentinet \
          --no_train

done