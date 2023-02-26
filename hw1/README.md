# ADL HW1

[spec](./spec.pdf)

ref. [GitHub - ntu-adl-ta/ADL21-HW1](https://github.com/ntu-adl-ta/ADL21-HW1)

## env

```shell
conda env create -f ./env/environment.yml
```

## Reproduce

```
bash download.sh
```

## Preprocess

```
bash preprocess.sh
```

## Intent

### Train

```
python ./src/train_intent.py
```

### Test

```
bash intent_cls.sh ./data/intent/test.json pred.intent.csv
```

## Slot

### Train

```
python ./src/train_slot.py
```

### Test

```
bash slot_tag.sh ./data/slot/test.json pred.slot.csv
```
