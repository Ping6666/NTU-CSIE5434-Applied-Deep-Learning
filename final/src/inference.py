import random
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from preprocess import MODES

from preprocess import (
    global_init_workhouse,
    preprocess_workhouse,
    dataset_workhouse,
    inference_prediction,
)
from model import Hahow_Model
from dataset import Hahow_Dataset

## global ##

SEED = 5487
DEVICE = 'cuda:1'

BATCH_SIZE = 64
NUM_WORKER = 8

EMBED_SIZE = 2
FEATURE_NUM = 91
HIDDEN_NUM = 128
DROPOUT = 0.1

TOPK = 50


def set_seed(seed):
    ## insure the reproducibility ##
    # Ref. https://pytorch.org/docs/stable/notes/randomness.html

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return


## predict ##


def predict(test_loader, model):
    _user_ids, _y_topic_preds, _y_course_preds = [], [], []
    for _, data in tqdm(enumerate(test_loader),
                        total=len(test_loader),
                        desc='Test',
                        leave=False):
        # data collate_fn
        _user_id, (_x_gender, _x_vector, _), (_, _) = data
        _x_gender = _x_gender.to(DEVICE)
        _x_vector = _x_vector.to(DEVICE)

        # eval: data -> model -> loss
        with torch.no_grad():
            _y_pred = model(_x_gender, _x_vector)

        # report
        _user_ids.extend(_user_id)
        _y_topic_pred, _y_course_pred = inference_prediction(_y_pred)
        _y_topic_preds.extend(_y_topic_pred)
        _y_course_preds.extend(_y_course_pred)
    return _user_ids, _y_topic_preds, _y_course_preds


def save_prediction(prediction, les, save_file_topic, save_file_course):
    _user_ids, _y_topic_preds, _y_course_preds = prediction
    user_le, course_le = les

    _user_ids = list(user_le.inverse_transform(_user_ids))

    with open(save_file_topic, 'w') as f:
        f.write('user_id,subgroup\n')
        for c_user_id, c_pred in zip(_user_ids, _y_topic_preds):
            c_subgroup = ' '.join([str(c) for c in c_pred])

            f.write(f'{c_user_id},{c_subgroup}\n')

    with open(save_file_course, 'w') as f:
        f.write('user_id,course_id\n')
        for c_user_id, c_pred in zip(_user_ids, _y_course_preds):
            _c_pred = course_le.inverse_transform(np.array(c_pred))
            c_course = ' '.join([str(c) for c in _c_pred])

            f.write(f'{c_user_id},{c_course}\n')
    return


## main ##


def main():
    set_seed(SEED)

    print('***Model***')
    model = Hahow_Model(EMBED_SIZE, FEATURE_NUM, HIDDEN_NUM, FEATURE_NUM,
                        DROPOUT)
    model.load_state_dict(torch.load('./save/topic_25.pt'))
    model.to(DEVICE)
    model.eval()

    print('***Global***')
    global_init_workhouse()

    print('***Data***')
    df_preprocess = preprocess_workhouse()

    print('***Hahow_Dataset***')
    test_seen_datasets = Hahow_Dataset(
        dataset_workhouse(df_preprocess, MODES[3]))
    test_unseen_datasets = Hahow_Dataset(
        dataset_workhouse(df_preprocess, MODES[4]))

    print('***DataLoader***')
    test_seen_loader = DataLoader(
        test_seen_datasets,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
    )
    test_unseen_loader = DataLoader(
        test_unseen_datasets,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
    )

    print('***Predict Seen***')
    test_seen_user_le = test_seen_datasets.get_user_id_labelencoder()
    test_seen_course_le = test_seen_datasets.get_course_id_labelencoder()
    save_prediction(
        predict(test_seen_loader, model),
        (test_seen_user_le, test_seen_course_le),
        './seen_user_topic.csv',
        './seen_user_course.csv',
    )

    print('***Predict UnSeen***')
    test_unseen_user_le = test_unseen_datasets.get_user_id_labelencoder()
    test_unseen_course_le = test_unseen_datasets.get_course_id_labelencoder()
    save_prediction(
        predict(test_unseen_loader, model),
        (test_unseen_user_le, test_unseen_course_le),
        './unseen_user_topic.csv',
        './unseen_user_course.csv',
    )

    print('All Epoch on Test were finished.\n')
    return


if __name__ == "__main__":
    main()

# ./save/topic_25.pt
# 025/050
# Train       | loss = 0.36557, subgroup_acc = 0.35878, course_acc = 0.00360
# Eval_Seen   | loss = 0.37210, subgroup_acc = 0.21223, course_acc = 0.04554
# Eval_UnSeen | loss = 0.24801, subgroup_acc = 0.23404, course_acc = 0.02180
# Test_Seen   | loss =      NC, subgroup_acc = 0.22085, course_acc = 0.04649 (kaggle)
# Test_UnSeen | loss =      NC, subgroup_acc = 0.23088, course_acc =  untest (kaggle)

'''

001/050
Train       | loss = 0.40784, subgroup_acc = 0.30460, course_acc = 0.00371
Eval_Seen   | loss = 0.40782, subgroup_acc = 0.15686, course_acc = 0.01111
Eval_UnSeen | loss = 0.26743, subgroup_acc = 0.15129, course_acc = 0.00676
002/050
Train       | loss = 0.39390, subgroup_acc = 0.34390, course_acc = 0.00357
Eval_Seen   | loss = 0.39455, subgroup_acc = 0.18193, course_acc = 0.01744
Eval_UnSeen | loss = 0.26026, subgroup_acc = 0.17892, course_acc = 0.01069
003/050
Train       | loss = 0.39164, subgroup_acc = 0.35131, course_acc = 0.00356
Eval_Seen   | loss = 0.38632, subgroup_acc = 0.20406, course_acc = 0.02413
Eval_UnSeen | loss = 0.25538, subgroup_acc = 0.20558, course_acc = 0.01375
004/050
Train       | loss = 0.38955, subgroup_acc = 0.35339, course_acc = 0.00353
Eval_Seen   | loss = 0.38855, subgroup_acc = 0.21984, course_acc = 0.03546
Eval_UnSeen | loss = 0.25494, subgroup_acc = 0.21588, course_acc = 0.01803
005/050
Train       | loss = 0.38800, subgroup_acc = 0.35576, course_acc = 0.00354
Eval_Seen   | loss = 0.37797, subgroup_acc = 0.22176, course_acc = 0.04343
Eval_UnSeen | loss = 0.25173, subgroup_acc = 0.22824, course_acc = 0.02099

006/050
Train       | loss = 0.39303, subgroup_acc = 0.35588, course_acc = 0.00358
Eval_Seen   | loss = 0.37880, subgroup_acc = 0.22268, course_acc = 0.03516
Eval_UnSeen | loss = 0.24987, subgroup_acc = 0.22981, course_acc = 0.01729
007/050
Train       | loss = 0.38617, subgroup_acc = 0.35721, course_acc = 0.00355
Eval_Seen   | loss = 0.36998, subgroup_acc = 0.21947, course_acc = 0.03418
Eval_UnSeen | loss = 0.24729, subgroup_acc = 0.21821, course_acc = 0.01654
008/050
Train       | loss = 0.38420, subgroup_acc = 0.35790, course_acc = 0.00362
Eval_Seen   | loss = 0.36536, subgroup_acc = 0.21644, course_acc = 0.04250
Eval_UnSeen | loss = 0.24638, subgroup_acc = 0.22075, course_acc = 0.02035
009/050
Train       | loss = 0.38380, subgroup_acc = 0.35905, course_acc = 0.00357
Eval_Seen   | loss = 0.37551, subgroup_acc = 0.21227, course_acc = 0.04198
Eval_UnSeen | loss = 0.25000, subgroup_acc = 0.22365, course_acc = 0.02083
010/050
Train       | loss = 0.38225, subgroup_acc = 0.35875, course_acc = 0.00358
Eval_Seen   | loss = 0.37245, subgroup_acc = 0.20998, course_acc = 0.04403
Eval_UnSeen | loss = 0.24852, subgroup_acc = 0.22470, course_acc = 0.02145

011/050
Train       | loss = 0.38176, subgroup_acc = 0.35868, course_acc = 0.00358
Eval_Seen   | loss = 0.36965, subgroup_acc = 0.22041, course_acc = 0.04206
Eval_UnSeen | loss = 0.24662, subgroup_acc = 0.23239, course_acc = 0.02030
012/050
Train       | loss = 0.37970, subgroup_acc = 0.35866, course_acc = 0.00361
Eval_Seen   | loss = 0.37575, subgroup_acc = 0.21359, course_acc = 0.04408
Eval_UnSeen | loss = 0.24953, subgroup_acc = 0.22650, course_acc = 0.02154
013/050
Train       | loss = 0.37882, subgroup_acc = 0.35886, course_acc = 0.00360
Eval_Seen   | loss = 0.38264, subgroup_acc = 0.21247, course_acc = 0.04352
Eval_UnSeen | loss = 0.25233, subgroup_acc = 0.23308, course_acc = 0.02118
014/050
Train       | loss = 0.37704, subgroup_acc = 0.35892, course_acc = 0.00354
Eval_Seen   | loss = 0.38462, subgroup_acc = 0.21247, course_acc = 0.04283
Eval_UnSeen | loss = 0.24881, subgroup_acc = 0.23250, course_acc = 0.02063
015/050
Train       | loss = 0.37697, subgroup_acc = 0.35954, course_acc = 0.00359
Eval_Seen   | loss = 0.38585, subgroup_acc = 0.21374, course_acc = 0.04364
Eval_UnSeen | loss = 0.25298, subgroup_acc = 0.23784, course_acc = 0.02115

016/050
Train       | loss = 0.37671, subgroup_acc = 0.35821, course_acc = 0.00362
Eval_Seen   | loss = 0.37421, subgroup_acc = 0.21798, course_acc = 0.04293
Eval_UnSeen | loss = 0.24899, subgroup_acc = 0.23954, course_acc = 0.02073
017/050
Train       | loss = 0.37474, subgroup_acc = 0.35920, course_acc = 0.00361
Eval_Seen   | loss = 0.37548, subgroup_acc = 0.21405, course_acc = 0.04409
Eval_UnSeen | loss = 0.24878, subgroup_acc = 0.23860, course_acc = 0.02141
018/050
Train       | loss = 0.37270, subgroup_acc = 0.35937, course_acc = 0.00349
Eval_Seen   | loss = 0.37217, subgroup_acc = 0.21483, course_acc = 0.04489
Eval_UnSeen | loss = 0.24835, subgroup_acc = 0.23457, course_acc = 0.02164
019/050
Train       | loss = 0.37173, subgroup_acc = 0.35903, course_acc = 0.00356
Eval_Seen   | loss = 0.37770, subgroup_acc = 0.21545, course_acc = 0.04241
Eval_UnSeen | loss = 0.24887, subgroup_acc = 0.24279, course_acc = 0.02060
020/050
Train       | loss = 0.36931, subgroup_acc = 0.35852, course_acc = 0.00361
Eval_Seen   | loss = 0.37466, subgroup_acc = 0.20347, course_acc = 0.04400
Eval_UnSeen | loss = 0.25015, subgroup_acc = 0.22871, course_acc = 0.02148

021/050
Train       | loss = 0.37006, subgroup_acc = 0.35852, course_acc = 0.00358
Eval_Seen   | loss = 0.37854, subgroup_acc = 0.20860, course_acc = 0.04476
Eval_UnSeen | loss = 0.25118, subgroup_acc = 0.23339, course_acc = 0.02195
022/050
Train       | loss = 0.36899, subgroup_acc = 0.35880, course_acc = 0.00356
Eval_Seen   | loss = 0.36902, subgroup_acc = 0.21180, course_acc = 0.04291
Eval_UnSeen | loss = 0.24592, subgroup_acc = 0.23848, course_acc = 0.02063
023/050
Train       | loss = 0.36674, subgroup_acc = 0.35888, course_acc = 0.00363
Eval_Seen   | loss = 0.37689, subgroup_acc = 0.21604, course_acc = 0.04375
Eval_UnSeen | loss = 0.24851, subgroup_acc = 0.23868, course_acc = 0.02096
024/050
Train       | loss = 0.36497, subgroup_acc = 0.35815, course_acc = 0.00358
Eval_Seen   | loss = 0.37530, subgroup_acc = 0.21658, course_acc = 0.04550
Eval_UnSeen | loss = 0.24908, subgroup_acc = 0.23821, course_acc = 0.02169
025/050
Train       | loss = 0.36557, subgroup_acc = 0.35878, course_acc = 0.00360
Eval_Seen   | loss = 0.37210, subgroup_acc = 0.21223, course_acc = 0.04554
Eval_UnSeen | loss = 0.24801, subgroup_acc = 0.23404, course_acc = 0.02180

026/050
Train       | loss = 0.36405, subgroup_acc = 0.35896, course_acc = 0.00359
Eval_Seen   | loss = 0.37854, subgroup_acc = 0.20650, course_acc = 0.04520
Eval_UnSeen | loss = 0.24939, subgroup_acc = 0.22759, course_acc = 0.02147
027/050
Train       | loss = 0.36527, subgroup_acc = 0.35859, course_acc = 0.00359
Eval_Seen   | loss = 0.36153, subgroup_acc = 0.21249, course_acc = 0.04472
Eval_UnSeen | loss = 0.24452, subgroup_acc = 0.23587, course_acc = 0.02087
028/050
Train       | loss = 0.36227, subgroup_acc = 0.35800, course_acc = 0.00368
Eval_Seen   | loss = 0.37228, subgroup_acc = 0.21310, course_acc = 0.04498
Eval_UnSeen | loss = 0.24942, subgroup_acc = 0.23905, course_acc = 0.02123
029/050
Train       | loss = 0.36163, subgroup_acc = 0.35832, course_acc = 0.00361
Eval_Seen   | loss = 0.36934, subgroup_acc = 0.21687, course_acc = 0.04441
Eval_UnSeen | loss = 0.24742, subgroup_acc = 0.24131, course_acc = 0.02081
030/050
Train       | loss = 0.35989, subgroup_acc = 0.35800, course_acc = 0.00364
Eval_Seen   | loss = 0.36166, subgroup_acc = 0.21859, course_acc = 0.04310
Eval_UnSeen | loss = 0.24571, subgroup_acc = 0.24523, course_acc = 0.02030

031/050
Train       | loss = 0.36295, subgroup_acc = 0.35871, course_acc = 0.00362
Eval_Seen   | loss = 0.37208, subgroup_acc = 0.20787, course_acc = 0.04334
Eval_UnSeen | loss = 0.24891, subgroup_acc = 0.23377, course_acc = 0.02060
032/050
Train       | loss = 0.36101, subgroup_acc = 0.35768, course_acc = 0.00364
Eval_Seen   | loss = 0.37482, subgroup_acc = 0.20887, course_acc = 0.04359
Eval_UnSeen | loss = 0.25020, subgroup_acc = 0.23675, course_acc = 0.02088
033/050
Train       | loss = 0.35971, subgroup_acc = 0.35728, course_acc = 0.00361
Eval_Seen   | loss = 0.37446, subgroup_acc = 0.22111, course_acc = 0.04282
Eval_UnSeen | loss = 0.24922, subgroup_acc = 0.24860, course_acc = 0.02062
034/050
Train       | loss = 0.35980, subgroup_acc = 0.35812, course_acc = 0.00361
Eval_Seen   | loss = 0.38481, subgroup_acc = 0.20797, course_acc = 0.04437
Eval_UnSeen | loss = 0.25558, subgroup_acc = 0.22453, course_acc = 0.02149
035/050
Train       | loss = 0.36025, subgroup_acc = 0.35811, course_acc = 0.00363
Eval_Seen   | loss = 0.37186, subgroup_acc = 0.21450, course_acc = 0.04412
Eval_UnSeen | loss = 0.24801, subgroup_acc = 0.23854, course_acc = 0.02141

036/050
Train       | loss = 0.35875, subgroup_acc = 0.35754, course_acc = 0.00363
Eval_Seen   | loss = 0.39161, subgroup_acc = 0.20980, course_acc = 0.04443
Eval_UnSeen | loss = 0.25978, subgroup_acc = 0.23469, course_acc = 0.02125
037/050
Train       | loss = 0.35961, subgroup_acc = 0.35762, course_acc = 0.00356
Eval_Seen   | loss = 0.38280, subgroup_acc = 0.21337, course_acc = 0.04457
Eval_UnSeen | loss = 0.25414, subgroup_acc = 0.23335, course_acc = 0.02151
038/050
Train       | loss = 0.35806, subgroup_acc = 0.35835, course_acc = 0.00359
Eval_Seen   | loss = 0.38624, subgroup_acc = 0.21481, course_acc = 0.04341
Eval_UnSeen | loss = 0.25299, subgroup_acc = 0.24009, course_acc = 0.02111
039/050
Train       | loss = 0.35745, subgroup_acc = 0.35785, course_acc = 0.00363
Eval_Seen   | loss = 0.38982, subgroup_acc = 0.21667, course_acc = 0.04356
Eval_UnSeen | loss = 0.25710, subgroup_acc = 0.23874, course_acc = 0.02084
040/050
Train       | loss = 0.35830, subgroup_acc = 0.35857, course_acc = 0.00358
Eval_Seen   | loss = 0.38690, subgroup_acc = 0.21075, course_acc = 0.04324
Eval_UnSeen | loss = 0.25512, subgroup_acc = 0.23556, course_acc = 0.02080

041/050
Train       | loss = 0.35852, subgroup_acc = 0.35789, course_acc = 0.00365
Eval_Seen   | loss = 0.39269, subgroup_acc = 0.20396, course_acc = 0.04324
Eval_UnSeen | loss = 0.25799, subgroup_acc = 0.23255, course_acc = 0.02059
042/050
Train       | loss = 0.35619, subgroup_acc = 0.35810, course_acc = 0.00362
Eval_Seen   | loss = 0.37578, subgroup_acc = 0.21298, course_acc = 0.04231
Eval_UnSeen | loss = 0.25167, subgroup_acc = 0.23617, course_acc = 0.02019
043/050
Train       | loss = 0.35622, subgroup_acc = 0.35825, course_acc = 0.00362
Eval_Seen   | loss = 0.41038, subgroup_acc = 0.19518, course_acc = 0.04609
Eval_UnSeen | loss = 0.26945, subgroup_acc = 0.21270, course_acc = 0.02247
044/050
Train       | loss = 0.35546, subgroup_acc = 0.35831, course_acc = 0.00366
Eval_Seen   | loss = 0.40116, subgroup_acc = 0.20450, course_acc = 0.04419
Eval_UnSeen | loss = 0.26148, subgroup_acc = 0.23454, course_acc = 0.02160
045/050
Train       | loss = 0.35587, subgroup_acc = 0.35692, course_acc = 0.00363
Eval_Seen   | loss = 0.38524, subgroup_acc = 0.21282, course_acc = 0.04541
Eval_UnSeen | loss = 0.25505, subgroup_acc = 0.23959, course_acc = 0.02147

046/050
Train       | loss = 0.35262, subgroup_acc = 0.35810, course_acc = 0.00360
Eval_Seen   | loss = 0.39293, subgroup_acc = 0.21112, course_acc = 0.04395
Eval_UnSeen | loss = 0.25831, subgroup_acc = 0.22888, course_acc = 0.02098
047/050
Train       | loss = 0.35309, subgroup_acc = 0.35808, course_acc = 0.00358
Eval_Seen   | loss = 0.39854, subgroup_acc = 0.20548, course_acc = 0.04510
Eval_UnSeen | loss = 0.26353, subgroup_acc = 0.23200, course_acc = 0.02147
048/050
Train       | loss = 0.35169, subgroup_acc = 0.35790, course_acc = 0.00366
Eval_Seen   | loss = 0.38437, subgroup_acc = 0.22150, course_acc = 0.04268
Eval_UnSeen | loss = 0.25598, subgroup_acc = 0.24752, course_acc = 0.02040
049/050
Train       | loss = 0.35499, subgroup_acc = 0.35747, course_acc = 0.00359
Eval_Seen   | loss = 0.39759, subgroup_acc = 0.21655, course_acc = 0.04296
Eval_UnSeen | loss = 0.25994, subgroup_acc = 0.24424, course_acc = 0.02077
050/050
Train       | loss = 0.35371, subgroup_acc = 0.35747, course_acc = 0.00362
Eval_Seen   | loss = 0.38958, subgroup_acc = 0.21720, course_acc = 0.04358
Eval_UnSeen | loss = 0.25707, subgroup_acc = 0.24412, course_acc = 0.02110
All Epoch on Train and Eval were finished.

'''
