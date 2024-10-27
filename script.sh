python finetune.py --dataset massive_intent --seed 42 --model FacebookAI/xlm-roberta-base --lr 1e-5 --max_epoch 100 --batch_size 32 --early_stopping 3

python finetune.py --dataset nusax --seed 42 --model FacebookAI/xlm-roberta-base --lr 1e-5 --max_epoch 100 --batch_size 32 --early_stopping 3

python finetune.py --dataset sib200 --seed 42 --model FacebookAI/xlm-roberta-base --lr 1e-5 --max_epoch 100 --batch_size 8 --early_stopping 5