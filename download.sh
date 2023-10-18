rm -f 'test.json' 'train.json' 'valid.json' 'context.json'

curl -L -o 'test.json' 'https://huggingface.co/datasets/weitung8/ntuadlhw1/resolve/main/test.json'
curl -L -o 'train.json' 'https://huggingface.co/datasets/weitung8/ntuadlhw1/resolve/main/train.json'
curl -L -o 'valid.json' 'https://huggingface.co/datasets/weitung8/ntuadlhw1/resolve/main/valid.json'
curl -L -o 'context.json' 'https://huggingface.co/datasets/weitung8/ntuadlhw1/resolve/main/context.json'

mkdir -p weitung8/ntuadlhw1-multiple-choice/
mkdir -p weitung8/ntuadlhw1-question-answering/

curl -L -o 'weitung8/ntuadlhw1-multiple-choice/added_tokens.json' 'https://huggingface.co/weitung8/ntuadlhw1-multiple-choice/resolve/main/added_tokens.json'
curl -L -o 'weitung8/ntuadlhw1-multiple-choice/config.json' 'https://huggingface.co/weitung8/ntuadlhw1-multiple-choice/resolve/main/config.json'
curl -L -o 'weitung8/ntuadlhw1-multiple-choice/pytorch_model.bin' 'https://huggingface.co/weitung8/ntuadlhw1-multiple-choice/resolve/main/pytorch_model.bin'
curl -L -o 'weitung8/ntuadlhw1-multiple-choice/speical_tokens_map.json' 'https://huggingface.co/weitung8/ntuadlhw1-multiple-choice/resolve/main/speical_tokens_map.json'
curl -L -o 'weitung8/ntuadlhw1-multiple-choice/tokenizer.json' 'https://huggingface.co/weitung8/ntuadlhw1-multiple-choice/resolve/main/tokenizer.json'
curl -L -o 'weitung8/ntuadlhw1-multiple-choice/tokenizer_config.json' 'https://huggingface.co/weitung8/ntuadlhw1-multiple-choice/resolve/main/tokenizer_config.json'
curl -L -o 'weitung8/ntuadlhw1-multiple-choice/vocab.txt' 'https://huggingface.co/weitung8/ntuadlhw1-multiple-choice/resolve/main/vocab.txt'

curl -L -o 'weitung8/ntuadlhw1-question-answering/added_tokens.json' 'https://huggingface.co/weitung8/ntuadlhw1-question-answering/resolve/main/added_tokens.json'
curl -L -o 'weitung8/ntuadlhw1-question-answering/config.json' 'https://huggingface.co/weitung8/ntuadlhw1-question-answering/resolve/main/config.json'
curl -L -o 'weitung8/ntuadlhw1-question-answering/pytorch_model.bin' 'https://huggingface.co/weitung8/ntuadlhw1-question-answering/resolve/main/pytorch_model.bin'
curl -L -o 'weitung8/ntuadlhw1-question-answering/speical_tokens_map.json' 'https://huggingface.co/weitung8/ntuadlhw1-question-answering/resolve/main/speical_tokens_map.json'
curl -L -o 'weitung8/ntuadlhw1-question-answering/tokenizer.json' 'https://huggingface.co/weitung8/ntuadlhw1-question-answering/resolve/main/tokenizer.json'
curl -L -o 'weitung8/ntuadlhw1-question-answering/tokenizer_config.json' 'https://huggingface.co/weitung8/ntuadlhw1-question-answering/resolve/main/tokenizer_config.json'
curl -L -o 'weitung8/ntuadlhw1-question-answering/vocab.txt' 'https://huggingface.co/weitung8/ntuadlhw1-question-answering/resolve/main/vocab.txt'
