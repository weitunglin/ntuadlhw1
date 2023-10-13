import sys
from itertools import chain
from copy import deepcopy

from transformers import AutoTokenizer, AutoModelForMultipleChoice, default_data_collator, pipeline
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

padding = 'max_length' # "max_length" or "do_not_pad"
max_seq_length = 512
batch_size = 32

context_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

datasets = load_dataset('json', data_files={'test':test_file})
context_list = pd.read_json(context_file)
context_list = context_list[0]

mc_tokenizer = AutoTokenizer.from_pretrained("weitung8/ntuadlhw1-multiple-choice", local_files_only=True)
mc_model = AutoModelForMultipleChoice.from_pretrained("weitung8/ntuadlhw1-multiple-choice", local_files_only=True)

def preprocess_function(examples):
    paragraphs = [[context_list[context] for context in contexts] for contexts in examples["paragraphs"]]
    questions = [[question] * 4 for question in examples["question"]]

    # Flatten out
    questions = list(chain(*questions))
    paragraphs = list(chain(*paragraphs))

    # Tokenize
    tokenized_examples = mc_tokenizer(
        paragraphs,
        questions,
        max_length=max_seq_length,
        padding=padding,
        truncation=True,
    )
    # Un-flatten
    tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    return tokenized_inputs

test_dataset = deepcopy(datasets['test'])
processed_datasets = datasets.map(preprocess_function, batched=True, remove_columns=["paragraphs"])

test_dataloader = DataLoader(processed_datasets['test'], collate_fn=default_data_collator, batch_size=batch_size)

# print(len(test_dataloader))

mc_model.to('cuda')
mc_model.eval()

mc_outputs = []
for batch in test_dataloader:
    with torch.no_grad():
        batch = { i: batch[i].to('cuda') for i in batch }
        outputs = mc_model(**batch)
    predictions = outputs.logits.argmax(dim=-1)
    mc_outputs.append(predictions.detach().cpu().numpy())

mc_outputs = list(chain(*mc_outputs))

# print(mc_outputs[:3])

context = [context_list[test_dataset['paragraphs'][i][c_idx]] for i, c_idx in enumerate(mc_outputs)]
question = test_dataset['question']
ids = test_dataset['id']

# print(context[:3])
# print(question[:3])

qa_pipe = pipeline("question-answering", model="weitung8/ntuadlhw1-question-answering", device=0)

answer = qa_pipe(question=question, context=context)

# print(answer[:3])

result = pd.DataFrame(data={'id':ids, 'answer':[i['answer'] for i in answer]})

result.to_csv(output_file, index=False)
