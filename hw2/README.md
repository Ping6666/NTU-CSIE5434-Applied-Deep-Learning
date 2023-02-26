# ADL HW2

[spec](./spec.pdf)

ref. [GitHub - huggingface/transformers/examples](https://github.com/huggingface/transformers/tree/main/examples)

## env

```
conda env create -f ./env/environment.yml
```

## Train stage

### multiple_choice

```
# for both pretrained & from scratch
bash train_mc.sh ./data/context.json ./data/train.json ./data/valid.json
```

### question_answering

```
# for both pretrained & from scratch
bash train_qa.sh ./data/context.json ./data/train.json ./data/valid.json
```

## Test stage

```
# download model
bash download.sh

# run swag & qa
bash run.sh ./data/context.json ./data/test.json ./qa_pred.csv
```
