# ntuadlhw1

## quickstart

* install requirements

```pip install -r requirements.txt```

* start training multiple choice model

```python train_mc.py --dataset_name weitung8/ntuadlhw1 --model_name_or_path bert-base-chinese --push_to_hub --hub_model_id weitung8/ntuadlhw1-multiple-choice --with_tracking --report_to wandb --max_seq_length 512 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 4 --output_dir weitung8/ntuadlhw1-multiple-choice --num_train_epochs 1 --learning_rate 3e-5 --pad_to_max_length```

* start training question answering model

```python train_qa.py --dataset_name weitung8/ntuadlhw1 --model_name_or_path bert-base-chinese --push_to_hub --hub_model_id weitung8/ntuadlhw1-question-answering --with_tracking --report_to wandb --max_seq_length 512 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 2 --output_dir weitung8/ntuadlhw1-question-answering --num_train_epochs 3 --learning_rate 3e-5 --pad_to_max_length```

