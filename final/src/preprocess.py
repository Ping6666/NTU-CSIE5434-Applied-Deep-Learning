from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from tqdm import tqdm

tqdm.pandas()

from torch import topk

from text2vec import SentenceModel, semantic_search

MODES = ['train', 'val_seen', 'val_unseen', 'test_seen', 'test_unseen']
BASE_DIR = './hahow/data/'
TOPK = 91

## Singleton: guarantee only one copy in the program ##

subgroup_id_pairs: Dict[str, int] = None
sentence_model: SentenceModel = None
subgroups_embeddings = None
vector_table: Dict[str, np.array] = {}

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
            pd.DataFrame: columns with raw data, `user_id`, `course_id`.
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
        if ' ' in a:
            return a.split(' ')
        return [a]

    df['course_id'].fillna(value='', inplace=True)
    df['course_id'] = df[['course_id']].progress_apply(
        lambda x: split_to_list(*x),
        axis=1,
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
            name: `val_seen_group.csv`, `val_unseen_group.csv`, 
            `test_seen_group.csv`, `test_unseen_group.csv`.

        Returns:
            pd.DataFrame: columns with raw data, `user_id`, `subgroup`.
    '''
    df = pd.DataFrame(
        pd.read_csv(
            BASE_DIR + name,
            dtype={
                'user_id': str,
                'subgroup': str,
            },
        ))
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


def get_model() -> None:
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


## convertor ##


def convert_gender(a: str) -> int:
    rt = -1
    if a == 'male':
        rt = 0
    elif a == 'female':
        rt = 1
    elif a == 'other':
        rt = 2
    return rt


def convert_subgroup_strs_to_ids(a: str) -> np.array[np.int8]:
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


def convert_single_text2vec(a: str, topk: int,
                            multiply: float) -> np.array[np.float64]:
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

        a_vector[a] = c_rt.copy()
        rt = c_rt.copy()

    # do multiply
    rt = np.multiply(rt, multiply)
    return rt


def convert_multiple_text2vec(a: List[str], topk: List[int],
                              multiply: List[float]) -> np.array[np.float64]:
    '''
    do convert_single_text2vec that share same `topk` & `multiply`.
    '''

    assert (((len(a) == len(topk)) and (len(a) == len(multiply))),
            'convert_multiple_text2vec | wrong length!!!')

    _text2vec = []
    for a_i, _topk, _multiply in zip(a, topk, multiply):
        # convert string to List[string]
        if ',' in a_i:
            a_i = a_i.split(',')
        else:
            a_i = [a_i]

        # compute convert_single_text2vec
        for a_i_j in a_i:
            _text2vec.append(convert_single_text2vec(a_i_j, _topk, _multiply))

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


def convert_predict_to_int_list(predicts) -> List[int]:
    c_group_lists = []
    for predict in predicts:
        c_group_list = (topk(predict, TOPK).indices + 1).tolist()
        c_group_lists.append(c_group_list)
    return c_group_lists


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
        lambda x: convert_multiple_text2vec(x, [10, 1, 2], [1, 5, 1]),
        axis=1,
    )
    return df


def manipulate_courses(df: pd.DataFrame) -> pd.DataFrame:
    '''
        Args:
            df: see get_dataframe_courses.

        Returns:
            pd.DataFrame: columns with post data, `courses_text2vec`, `courses_label`.
    '''
    df['courses_text2vec'] = df[[
        'course_name', 'teacher_intro', 'groups', 'sub_groups', 'topics',
        'description', 'will_learn', 'required_tools',
        'recommended_background', 'target_group'
    ]].progress_apply(
        lambda x: convert_multiple_text2vec(x, [
            1, 91, 1, 1, 91, 91, 91, 91, 91, 91
        ], [5, 1, 5, 5, 1, 1, 1, 1, 1, 1]),
        axis=1,
    )
    # TODO
    df['courses_label'] = df[['sub_groups']].progress_apply(
        lambda x: convert_union_subgroups_id(*x),
        axis=1,
    )
    return df


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
                    - courses_text2vec, courses_label

        Returns:
            pd.DataFrame: columns with post data, ``.
    '''

    merge_dict = {}
    for _, c_row in tqdm(df.iterrows(), total=df.shape[0]):
        c_user_id = c_row['user_id']
        c_text2vec = c_row['courses_text2vec'].copy()
        c_labels: set = c_row['courses_label']

        c_lv = merge_dict.get(c_user_id)
        if c_lv is None:
            # seq. is VERY important
            c_lv = {}
            c_lv['courses_text2vec'] = c_text2vec
            c_lv['courses_label'] = c_labels
        else:
            c_lv['courses_text2vec'] += c_text2vec
            c_lv['courses_label'].update(c_labels)

        merge_dict[c_user_id] = c_lv

    df_merge = pd.DataFrame.from_dict(merge_dict, orient="index")
    df_merge.reset_index()
    df_merge.columns = ['user_id', 'courses_text2vec', 'courses_label']

    def set_pad_to_list(a: set) -> np.array[np.int8]:
        _a = list(a)
        if len(_a) != 91:
            _a += [0] * (91 - len(_a))
        rt = np.array(_a, dtype=np.int8)
        return rt

    df_merge['courses_label'] = df_merge['courses_label'].progress_apply(
        lambda x: set_pad_to_list(x))
    # print(df_merge)
    return df_merge


## workhouse ##


def preprocess_workhouse() -> Tuple(pd.DataFrame, pd.DataFrame):
    '''
    get all redundant data from preprocess stage.

        Returns:
            pd.DataFrame: df_users
            pd.DataFrame: df_courses
    '''
    df_users = get_dataframe_users()
    df_users = manipulate_users(df_users)
    print('df_users.columns', df_users.columns)

    df_courses = get_dataframe_courses()
    df_courses = manipulate_courses(df_courses)
    print('df_courses.columns', df_courses.columns)
    return df_users, df_courses


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
    df_users, df_courses = df_preprocess

    # flatten dataframe
    df = get_dataframe_flatten_user_course(name)
    df_flatten = pd.merge(df, df_courses, on='course_id')

    # unflatten dataframe
    df_man_merge = manipulate_merge_flatten(df_flatten)
    df_final = pd.merge(df_man_merge, df_users, on='user_id')

    # drop some other features
    df_final = df_final[[
        'user_id', 'users_gender', 'users_text2vec', 'courses_text2vec',
        'courses_label'
    ]]

    # sequence
    user_id = df['user_id'].to_list()
    # ground_truth
    courses_label = df['courses_label'].to_list()
    # train part (input & label)
    df = df[['users_gender', 'users_text2vec', 'courses_text2vec']]
    return user_id, df, courses_label


def dataset_workhouse(df_preprocess, mode: str):
    '''
        Args:
            df_preprocess: (df_users, df_courses)
                - df_users: see get_dataframe_users
                - df_courses: see get_dataframe_courses

        Returns:
            see _dataset_workhouse
    '''
    assert mode in MODES, 'dataset_workhouse | wrong mode!!!'

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

    return _dataset_workhouse(name, df_preprocess)


## main ##


def main():
    return


if __name__ == '__main__':
    main()
