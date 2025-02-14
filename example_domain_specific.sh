python train_cleaned.py \
--train /aircraft \
--batch_size 8 \
--epochs 96 \
--teacher ViT-B/32 \
--model tiny_vit_11m_224 \
--embedding_dim 512 \
--encoder_output_dim 448 \
--dataset aircraft \
--logdir /logs \
--synthetic_data True \
--diverse_prompts True \
--options_per_attribute 15 \
--logname finetuning \
--training_loss CLIP \
--distillation_loss L2 \
--distil_alpha 0.5 \
--learning_rate 0.0005 \
--weight_decay 0.0 \
--devices 1 \
--nodes 1 \
--start_model_path starting_checkpoint.ckpt # checkpoint from domain-agnostic training