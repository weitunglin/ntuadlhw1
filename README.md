# ntuadlhw1

## experiments steps

* install requirements

```pip install -r requirements.txt```

* download datasets

```bash download.sh```

* start training multiple choice model

```python train_mc.py --dataset_name weitung8/ntuadlhw1 --model_name_or_path hfl/chinese-roberta-wwm-ext --push_to_hub --hub_model_id weitung8/ntuadlhw1-multiple-choice --with_tracking --report_to wandb --max_seq_length 512 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 4 --output_dir weitung8/ntuadlhw1-multiple-choice --num_train_epochs 1 --learning_rate 2e-5 --weight_decay 0.001  --checkpointing_steps epoch```

* start training question answering model

```python train_qa.py --dataset_name weitung8/ntuadlhw1 --model_name_or_path hfl/chinese-roberta-wwm-ext --push_to_hub --hub_model_id weitung8/ntuadlhw1-question-answering --with_tracking --report_to wandb --max_seq_length 512 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 4 --output_dir weitung8/ntuadlhw1-question-answering --num_train_epochs 3 --learning_rate 2e-5 --checkpointing_steps epoch --lr_scheduler_type cosine```

* get the test results

```bash run.sh context.json test.json submission.csv```

* submit results to kaggle

```kaggle competitions submit -c ntuadl2023hw1 -f submission.csv -m "some message"```

