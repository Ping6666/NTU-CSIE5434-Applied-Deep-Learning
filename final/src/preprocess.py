from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

from torch import topk, Tensor

from text2vec import SentenceModel, semantic_search
from sklearn.preprocessing import LabelEncoder

MODES = ['train', 'val_seen', 'val_unseen', 'test_seen', 'test_unseen']
BASE_DIR = './hahow/data/'
TOPK = 91

## Singleton: guarantee only one copy in the program ##

subgroup_id_pairs: Dict[str, int] = None
sentence_model: SentenceModel = None
subgroups_embeddings = None
vector_table: Dict[str, np.array] = {}

subgroup_course_metrix = None

## read_csv ##


def read_csv_subgroups(name: str = 'subgroups.csv') -> None:
    '''
    store a Dict[str, int] as global variable:
    (key, value) => (subgroup_name, subgroup_id).

        Args:
            name: `subgroups.csv`.
    '''

    global subgroup_id_pairs

    if subgroup_id_pairs is not None:
        return

    pd_csv = pd.read_csv(
        BASE_DIR + name,
        dtype={
            'subgroup_id': int,  # 1~91
            'subgroup_name': str,
        },
        index_col=1,
    )
    subgroup_id_pairs = pd_csv.squeeze().to_dict()
    return


## get_dataframe ##


def get_dataframe_user_course(name: str) -> pd.DataFrame:
    '''
        Args:
            name: `train.csv`, `val_seen.csv`, `val_unseen.csv`, 
            `test_seen.csv`, `test_unseen.csv`.

        Returns:
            pd.DataFrame: columns with raw data, `user_id`, `course_id`, `num_course`.
    '''
    df = pd.DataFrame(
        pd.read_csv(
            BASE_DIR + name,
            dtype={
                'user_id': str,
                'course_id': str,
            },
        ))

    def split_to_list(a: str) -> List[str]:
        _a = []
        if ' ' in a:
            _a = a.split(' ')
        else:
            _a = [a]
        return _a, len(_a)

    df['course_id'].fillna(value='', inplace=True)
    # df['course_id'] = df[['course_id']].progress_apply(
    #     lambda x: split_to_list(*x),
    #     axis=1,
    # )
    df[['course_id', 'num_course']] = df[['course_id']].progress_apply(
        lambda x: split_to_list(*x),
        axis=1,
        result_type='expand',
    )
    # print(df)
    return df


def get_dataframe_flatten_user_course(name: str) -> pd.DataFrame:
    '''
        Args:
            name: `train.csv`, `val_seen.csv`, `val_unseen.csv`, 
            `test_seen.csv`, `test_unseen.csv`.

        Returns:
            pd.DataFrame: columns with raw data (1 to 1), `user_id`, `course_id`.
    '''
    df = get_dataframe_user_course(name)

    flatten_list = []
    for _, c_row in tqdm(df.iterrows(), total=df.shape[0]):
        c_user = c_row['user_id']
        c_courses: List[str] = c_row['course_id']

        # for all course in current user
        for c_course in c_courses:
            # deal with empty space scenario
            _c_course = c_course.strip()
            if _c_course == '':
                continue

            c_train = {
                'user_id': c_user,
                'course_id': _c_course,
            }
            flatten_list.append(c_train)

    df_flatten = pd.DataFrame(flatten_list)
    # print(df_flatten)
    return df_flatten


def get_dataframe_user_subgroup(name: str) -> pd.DataFrame:
    '''
        Args:
            name: `train_group.csv`, `val_seen_group.csv`, `val_unseen_group.csv`, 
            `test_seen_group.csv`, `test_unseen_group.csv`.

        Returns:
            pd.DataFrame: columns with raw data, `user_id`, `subgroup`, `num_subgroup`.
    '''
    df = pd.DataFrame(
        pd.read_csv(
            BASE_DIR + name,
            dtype={
                'user_id': str,
                'subgroup': str,
            },
        ))

    def set_pad_to_list(a: str) -> List[str]:
        # split
        if ' ' in a:
            a = a.split(' ')
        else:
            a = [a]

        # convert to int
        _a = []
        for a_i in a:
            a_i = a_i.strip()
            if a_i == '':
                continue
            _a.append(int(a_i))

        # padding
        n, max_n, pad_num = len(_a), 91, -1
        if n < max_n:
            _a += [pad_num] * (max_n - n)
        _a = _a[:max_n]
        rt = np.array(_a, dtype=np.int8)
        return rt, n

    df['subgroup'].fillna(value='', inplace=True)
    # df['subgroup'] = df[['subgroup']].progress_apply(
    #     lambda x: set_pad_to_list(*x),
    #     axis=1,
    # )
    df[['label_subgroup', 'num_subgroup']] = df[['subgroup']].progress_apply(
        lambda x: set_pad_to_list(*x),
        axis=1,
        result_type='expand',
    )
    # print(df)
    return df


def get_dataframe_users(name: str = 'users.csv') -> pd.DataFrame:
    '''
        Args:
            name: `users.csv`.

        Returns:
            pd.DataFrame: columns with raw data, `user_id`, `gender`,
            `occupation_titles`, `interests`, `recreation_names`.
    '''
    df = pd.DataFrame(
        pd.read_csv(
            BASE_DIR + name,
            dtype={
                'user_id': str,
                'gender': str,
                'occupation_titles': str,
                'interests': str,
                'recreation_names': str,
            },
        ))

    df['gender'].fillna(value='', inplace=True)
    df['occupation_titles'].fillna(value='', inplace=True)
    df['interests'].fillna(value='', inplace=True)
    df['recreation_names'].fillna(value='', inplace=True)
    # print(df)
    return df


def get_dataframe_courses(name: str = 'courses.csv') -> pd.DataFrame:
    '''
        Args:
            name: `courses.csv`.

        Returns:
            pd.DataFrame: columns with raw data, `course_id`, `course_name`,
            `course_price`, `teacher_id`, `teacher_intro`, `groups`, 
            `sub_groups`, `topics`, `course_published_at_local`, 
            `description`, `will_learn`, `required_tools`, 
            `recommended_background`, `target_group`.
    '''
    df = pd.DataFrame(
        pd.read_csv(
            BASE_DIR + name,
            dtype={
                'course_id': str,
                'course_name': str,
                'course_price': int,
                'teacher_id': str,
                'teacher_intro': str,
                'groups': str,
                'sub_groups': str,
                'topics': str,
                'course_published_at_local': str,
                'description': str,
                'will_learn': str,
                'required_tools': str,
                'recommended_background': str,
                'target_group': str,
            },
        ))

    df['groups'].fillna(value='', inplace=True)
    df['sub_groups'].fillna(value='', inplace=True)
    # print(df)
    return df


## getter ##


def subgroup_id_to_idx(a: int) -> int:
    return a - 1


def subgroup_idx_to_id(a: int) -> int:
    return a + 1


def init_model() -> None:
    global sentence_model
    global subgroups_embeddings
    global subgroup_id_pairs

    if ((sentence_model is not None) and (subgroups_embeddings is not None)):
        return

    read_csv_subgroups()

    # model_name_or_path = 'shibing624/text2vec-base-chinese'
    sentence_model = SentenceModel()
    subgroups_embeddings = sentence_model.encode(list(
        subgroup_id_pairs.keys()))
    return


def get_labelencoder_do_fit(x):
    labelencoder = LabelEncoder()
    labelencoder.fit(np.array(x.tolist()))
    return labelencoder


## convertor ##


def convert_gender(a: str) -> int:
    rt = -1
    try:
        if a == 'male':
            rt = 0
        elif a == 'female':
            rt = 1
        elif a == 'other':
            rt = 2
    except:
        print('HI')
        rt = -1
    return rt


def convert_subgroup_strs_to_ids(a: str) -> np.array:
    global subgroup_id_pairs

    rt = np.zeros(len(subgroup_id_pairs), dtype=np.int8)

    # deal with sigle & multi item scenario
    if ',' in a:
        a = a.split(',')
    else:
        a = [a]

    for item in a:
        # deal with empty space scenario
        item = item.strip()
        if item == '':
            continue

        # sep. group & subgroup
        group, subgroup = item.split('_')

        subgroup_id = subgroup_id_pairs.get(subgroup)
        if subgroup_id is not None:
            subgroup_idx = subgroup_id_to_idx(subgroup_id)
            rt[subgroup_idx] = 1
    return rt


def convert_single_text2vec(a: str, topk: int, multiply: float) -> np.array:
    global sentence_model
    global vector_table
    global subgroup_id_pairs

    a_vector: np.array = vector_table.get(a)
    rt = None

    # compute the vector from string (text)
    if a_vector is not None:
        rt = a_vector.copy()
    else:
        item_embedding = sentence_model.encode(a)
        hits = semantic_search(item_embedding,
                               subgroups_embeddings,
                               top_k=topk)[0]

        corpus_ids, scores = [], []
        for hit in hits:
            corpus_ids.append(hit['corpus_id'])
            scores.append(hit['score'])

        # w = np.ones(topk) / np.arange(1, topk + 1)
        # scores = np.multiply(np.array(scores), w).tolist()

        c_rt = np.zeros(len(subgroup_id_pairs), dtype=np.float64)
        for c, s in zip(corpus_ids, scores):
            c_rt[c] = s

        vector_table[a] = c_rt.copy()
        rt = c_rt.copy()

    # do multiply
    rt = np.multiply(rt, multiply)
    return rt


def convert_multiple_text2vec(a: List[str], topk: List[int],
                              multiply: List[float]) -> np.array:
    '''
    do convert_single_text2vec that share same `topk` & `multiply`.
    '''

    # SyntaxWarning: assertion is always true, perhaps remove parentheses?
    # assert (((len(a) == len(topk)) and (len(a) == len(multiply))),
    #         'convert_multiple_text2vec | wrong length!!!')

    _text2vec = []
    for a_i, _topk, _multiply in zip(a, topk, multiply):
        a_i = str(a_i)
        # convert string to List[string]
        if ',' in a_i:
            a_i = a_i.split(',')
        else:
            a_i = [a_i]

        # compute convert_single_text2vec
        for a_i_j in a_i:
            _text2vec.append(convert_single_text2vec(a_i_j, _topk, _multiply))

    # TODO
    # option: sum
    text2vec = np.average(np.array(_text2vec), axis=0).copy()
    return text2vec


def convert_union_subgroups_id(a: str) -> set:
    global subgroup_id_pairs

    # deal with sigle & multi item scenario
    if ',' in a:
        a = a.split(',')
    else:
        a = [a]

    # for all subgroups
    rt = set()
    for subgroup in a:
        # deal with empty space scenario
        subgroup = subgroup.strip()
        if subgroup == '':
            continue

        subgroup_id = subgroup_id_pairs.get(subgroup)
        rt.add(subgroup_id)
    return rt


## manipulate ##


def manipulate_users(df: pd.DataFrame) -> pd.DataFrame:
    '''
        Args:
            df: see get_dataframe_users.

        Returns:
            pd.DataFrame: columns with post data, `users_gender`, `users_interests`
            `users_text2vec`.
    '''
    df['users_gender'] = df[['gender']].progress_apply(
        lambda x: convert_gender(*x),
        axis=1,
    )
    df['users_interests'] = df[['interests']].progress_apply(
        lambda x: convert_subgroup_strs_to_ids(*x),
        axis=1,
    )
    df['users_text2vec'] = df[[
        'occupation_titles', 'interests', 'recreation_names'
    ]].progress_apply(
        lambda x: convert_multiple_text2vec(
            x,
            [2, 1, 2],
            [1, 10, 1],
        ),
        axis=1,
    )
    return df


def manipulate_courses(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    '''
        Args:
            df: see get_dataframe_courses.

        Returns:
            pd.DataFrame: columns with post data, `courses_text2vec`.
    '''
    global subgroup_course_metrix

    all_course_id = np.array(df['course_id'].tolist())
    course_id_labelencoder = get_labelencoder_do_fit(all_course_id)

    df['courses_text2vec'] = df[[
        'course_name', 'teacher_intro', 'groups', 'sub_groups', 'topics',
        'description', 'will_learn', 'required_tools',
        'recommended_background', 'target_group'
    ]].progress_apply(
        lambda x: convert_multiple_text2vec(
            x,
            [2, 5, 1, 1, 2, 20, 5, 5, 5, 5],
            [1, 1, 1, 100, 1, 1, 1, 1, 1, 1],
        ),
        axis=1,
    )

    if subgroup_course_metrix is None:
        subgroup_course_metrix = np.array(df['courses_text2vec'].tolist())
        # print(subgroup_course_metrix.shape)  # (728, 91)

    return df, course_id_labelencoder


def manipulate_merge_flatten(df: pd.DataFrame) -> pd.DataFrame:
    '''
        Args:
            df: from get_dataframe_flatten_user_course & manipulate_courses

                - get_dataframe_flatten_user_course
                    - user_id, course_id
                - manipulate_courses
                    - course_id, course_name, course_price, teacher_id, 
                    - teacher_intro, groups, sub_groups, topics, 
                    - course_published_at_local, description, will_learn, 
                    - required_tools, recommended_background, target_group
                    - courses_text2vec

        Returns:
            pd.DataFrame: columns with post data, `user_id`, `courses_text2vec`.
    '''

    merge_dict = {}
    for _, c_row in tqdm(df.iterrows(), total=df.shape[0]):
        c_user_id = c_row['user_id']
        c_text2vec = c_row['courses_text2vec'].copy()

        c_lv = merge_dict.get(c_user_id)
        if c_lv is None:
            # seq. is VERY important
            # TODO
            # option: average
            c_lv = {}
            c_lv['courses_text2vec'] = c_text2vec
        else:
            c_lv['courses_text2vec'] += c_text2vec

        merge_dict[c_user_id] = c_lv

    df_merge = pd.DataFrame.from_dict(merge_dict, orient="index").reset_index()
    df_merge.columns = ['user_id', 'courses_text2vec']
    # print(df_merge)
    return df_merge


def manipulate_course_id_convertion(
        df: pd.DataFrame,
        course_id_labelencoder: LabelEncoder) -> pd.DataFrame:
    '''
        Args:
            df: see get_dataframe_user_course.

        Returns:
            pd.DataFrame: columns with post data, `label_course_id`.
    '''

    def set_pad_to_list(a: List[str], labelencoder: LabelEncoder) -> np.array:
        a = np.array(a)
        a = labelencoder.transform(a)

        # padding
        n, max_n, pad_num = len(a), 50, -1
        if n < max_n:
            a = np.concatenate((a, [pad_num] * (max_n - n)))
        _a = a[:max_n].copy()

        rt = np.array(_a, dtype=np.int16)
        return rt

    df['label_course_id'] = df['course_id'].progress_apply(
        lambda x: set_pad_to_list(x, course_id_labelencoder))
    return df


## workhouse ##


def global_init_workhouse():
    init_model()
    return


def preprocess_workhouse() -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    get all redundant data from preprocess stage.

        Returns:
            pd.DataFrame: df_users
            pd.DataFrame: df_courses
    '''
    print('******df_users******')
    df_users = get_dataframe_users()
    df_users = manipulate_users(df_users)
    print('df_users.columns', df_users.columns)
    print('df_users', df_users)

    print('******df_courses******')
    df_courses = get_dataframe_courses()
    (df_courses, course_id_labelrncoder) = manipulate_courses(df_courses)
    print('df_courses.columns', df_courses.columns)
    print('df_courses', df_courses)
    return df_users, (df_courses, course_id_labelrncoder)


def _dataset_workhouse(name, df_preprocess) -> Tuple[List, pd.DataFrame, List]:
    '''
        Args:
            name: see get_dataframe_flatten_user_course
            df_preprocess: see preprocess_workhouse

        Returns:
            user_id: List for checking the sequence of dataset
            df: pd.DataFrame the train part (input & label)
            courses_label: List the ground_truth
    '''
    df_users, (df_courses, course_id_labelencoder) = df_preprocess

    # flatten dataframe
    df = get_dataframe_flatten_user_course(name)
    df_flatten = pd.merge(df, df_courses, on='course_id')

    # unflatten dataframe
    df_man_merge = manipulate_merge_flatten(df_flatten)
    df_final = pd.merge(df_man_merge, df_users, on='user_id')

    # get ground truth
    print('File: ' + name[:-4] + '_group.csv')
    df_group = get_dataframe_user_subgroup(name[:-4] + '_group.csv')
    df_final = pd.merge(df_final, df_group, on='user_id')

    # get ground truth
    print('File: ' + name)
    df_course = get_dataframe_user_course(name)
    df_course = manipulate_course_id_convertion(df_course,
                                                course_id_labelencoder)
    df_final = pd.merge(df_final, df_course, on='user_id')

    # drop some other features
    df_final = df_final[[
        'user_id', 'users_gender', 'users_text2vec', 'courses_text2vec',
        'label_subgroup', 'label_course_id'
    ]]
    # print('df_final', df_final)

    # sequence
    user_id = np.array(df_final['user_id'].tolist())
    user_id_labelencoder = get_labelencoder_do_fit(user_id)
    user_id = user_id_labelencoder.transform(user_id)

    # ground_truth
    subgroup = df_final['label_subgroup']
    course_id = df_final['label_course_id']
    # train part (input & label)
    df_final = df_final[['users_gender', 'users_text2vec', 'courses_text2vec']]
    print('df_final', df_final)
    return ((user_id, user_id_labelencoder), df_final, subgroup,
            (course_id, course_id_labelencoder))


def dataset_workhouse(df_preprocess, mode: str):
    '''
        Args:
            df_preprocess: (df_users, df_courses)
                - df_users: see get_dataframe_users
                - df_courses: see get_dataframe_courses

        Returns:
            see _dataset_workhouse
    '''
    # assert mode in MODES, 'dataset_workhouse | wrong mode!!!'

    name = ''
    if mode == MODES[0]:
        name = 'train.csv'
    elif mode == MODES[1]:
        name = 'val_seen.csv'
    elif mode == MODES[2]:
        name = 'val_unseen.csv'
    elif mode == MODES[3]:
        name = 'test_seen.csv'
    elif mode == MODES[4]:
        name = 'test_unseen.csv'

    print(f'File: {name}')
    return _dataset_workhouse(name, df_preprocess)


## inference ##


def predict_course_search(predict: np.array) -> List[int]:
    global subgroup_course_metrix

    # print(predict.shape)  # (91,)
    # print(subgroup_course_metrix.shape)  # (728, 91)

    _predict = predict.reshape(-1, 1)
    _rt = np.matmul(subgroup_course_metrix, _predict)
    rt = _rt.reshape(-1).copy()

    # print(rt.shape)  # (728,)
    return rt


def inference_prediction(
        predicts: Tensor) -> Tuple[List[List[int]], List[List[int]]]:
    c_topic_lists, c_course_lists = [], []
    for predict in predicts:
        # topic
        c_topic_list = (topk(predict, TOPK).indices + 1).tolist()
        c_topic_lists.append(c_topic_list)

        # course
        _predict = predict.detach().cpu().numpy()
        course_predict = predict_course_search(_predict)
        c_course_list = [
            idx + 1 for idx in np.argsort(-np.array(course_predict))
        ]
        c_course_lists.append(c_course_list)
    return c_topic_lists, c_course_lists


## main ##


def printer_unique_counter(n: np.ndarray, pre_str: str = None):
    unique, counts = np.unique(n, return_counts=True)
    print(len(unique), end=' ')
    if pre_str != None:
        print(pre_str, end=' ')
    print(dict(zip(unique, counts)))
    return


def tester_1():
    df_user_course = get_dataframe_user_course('train.csv')
    printer_unique_counter(df_user_course['num_course'].to_numpy(),
                           pre_str='train course |')

    df_user_subgroup = get_dataframe_user_subgroup('train_group.csv')
    printer_unique_counter(df_user_subgroup['num_subgroup'].to_numpy(),
                           pre_str='train subgroup |')

    df_user_course = get_dataframe_user_course('val_seen.csv')
    printer_unique_counter(df_user_course['num_course'].to_numpy(),
                           pre_str='val_seen course |')

    df_user_subgroup = get_dataframe_user_subgroup('val_seen_group.csv')
    printer_unique_counter(df_user_subgroup['num_subgroup'].to_numpy(),
                           pre_str='val_seen subgroup |')

    df_user_course = get_dataframe_user_course('val_unseen.csv')
    printer_unique_counter(df_user_course['num_course'].to_numpy(),
                           pre_str='val_unseen course |')

    df_user_subgroup = get_dataframe_user_subgroup('val_unseen_group.csv')
    printer_unique_counter(df_user_subgroup['num_subgroup'].to_numpy(),
                           pre_str='val_unseen subgroup |')
    return


def main():

    print('***Global***')
    global_init_workhouse()

    print('***Data***')
    df_preprocess = preprocess_workhouse()

    print('***Hahow_Dataset***')
    datasets = dataset_workhouse(df_preprocess, MODES[0])
    print(datasets)
    datasets = dataset_workhouse(df_preprocess, MODES[1])
    print(datasets)
    datasets = dataset_workhouse(df_preprocess, MODES[2])
    print(datasets)
    datasets = dataset_workhouse(df_preprocess, MODES[3])
    print(datasets)
    datasets = dataset_workhouse(df_preprocess, MODES[4])
    print(datasets)
    return


if __name__ == '__main__':
    main()
