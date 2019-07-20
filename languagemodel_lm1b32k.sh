t2t-trainer \
  --generate_data \
  --data_dir=~/data/languagemodel \
  --output_dir=~/train/languagemodel \
  --problem=languagemodel_lm1b32k \
  --model=transformer \
  --hparams_set=transformer_base \
  --train_steps=10000 \
  --eval_steps=1000
