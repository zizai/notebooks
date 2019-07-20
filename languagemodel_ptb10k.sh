t2t-trainer \
  --generate_data \
  --data_dir=~/data/languagemodel \
  --output_dir=~/train/languagemodel \
  --problem=languagemodel_ptb10k \
  --model=transformer \
  --hparams_set=transformer_small \
  --train_steps=10000 \
  --eval_steps=1000
