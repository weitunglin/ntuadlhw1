rm -f 'test.json' 'train.json' 'valid.json' 'context.json'

curl -L -o 'test.json' 'https://huggingface.co/datasets/weitung8/ntuadlhw1/resolve/main/test.json'
curl -L -o 'train.json' 'https://huggingface.co/datasets/weitung8/ntuadlhw1/resolve/main/train.json'
curl -L -o 'valid.json' 'https://huggingface.co/datasets/weitung8/ntuadlhw1/resolve/main/valid.json'
curl -L -o 'context.json' 'https://huggingface.co/datasets/weitung8/ntuadlhw1/resolve/main/context.json'
