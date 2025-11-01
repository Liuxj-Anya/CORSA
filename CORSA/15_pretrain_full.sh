for sl in  '7.5e-5'  #You can change the sl to find the best hyperparameter.
do
		echo ${sl}
		python3.7 MAESC_training.py \
          --dataset twitter15 ./src/data/jsons/twitter15_info.json \
          --checkpoint_dir mytrain/checkpoints/twitter15 \
          --model_config config/pretrain_base.json \
          --log_dir mytrain/log/twitter15 \
          --num_beams 4 \
          --eval_every 1 \
          --lr ${sl} \
          --batch_size 32 \
          --epochs 100 \
          --grad_clip 5 \
          --warmup 0.1 \
          --seed 12 \
          --checkpoint /data/liuxj/aspect_sentiment_detect/pretrain_checkpoint/pretrain/pytorch_model.bin \
          --rank 2 \
          --trc_pretrain_file /data/liuxj/aspect_sentiment_detect/pretrain_checkpoint/trc_pretrain/pytorch_model.bin \
          --nn_attention_on \
          --nn_attention_mode 0 \
          --trc_on \
          --gcn_on \
          --dep_mode 2 \
          --sentinet \

done
