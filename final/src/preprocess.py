from typing import List, Dict

import numpy as np
import pandas as pd

from tqdm import tqdm

tqdm.pandas()

from text2vec import SentenceModel, semantic_search

BASE_DIR = './hahow/data/'

## global ##

group_subgroup_pair = {}
group_list = np.array([[0] * 91] * 91, dtype=np.float64)  # dangerous move


def print_list(l):
    for j in range(91):
        for k in range(91):
            print(l[j][k], end=' ')
        print()
    return


def update_group_list(subgroups_dict):
    global group_subgroup_pair
    global group_list
    _group_list = np.array([[0] * 91] * 91, dtype=np.float64)  # dangerous move

    for i in range(len(subgroups_dict)):
        groups = set()

        for k, v in group_subgroup_pair.items():
            if i in v:
                groups.update(set(v))

        if i in groups:
            groups.remove(i)

        for j in range(len(subgroups_dict)):
            if j in groups:
                _group_list[i][j] = 1
            else:
                _group_list[i][j] = 0

    group_list = _group_list
    return


## read csv ##


def read_csv_subgroups(name) -> Dict:
    '''
    name: 'subgroups.csv'
    '''
    subgroups = pd.read_csv(BASE_DIR + name,
                            dtype={
                                'subgroup_id': int,
                                'subgroup_name': str,
                            },
                            index_col=1).squeeze().to_dict()
    return subgroups


## getter ##


def get_model() -> SentenceModel:
    model = SentenceModel('shibing624/text2vec-base-chinese')
    return model


def get_subgroup_id(subgroups_dict, c_subgroup):
    c_subgroup_id = subgroups_dict.get(c_subgroup)
    if c_subgroup_id != None:
        return c_subgroup_id - 1
    # raise KeyError(f"HI {c_subgroup}.")
    return None


## convert ##


def convert_subgroup_vector_from_id(x):
    x = str(x)

    ## preprocess ##
    subgroups = []
    if ' ' in x:
        subgroups = x.split(' ')
    else:
        subgroups = [x]

    ## variable ##
    rt_list = [0] * 91

    for i in subgroups:
        # TODO change check on nan
        try:
            rt_list[int(i) - 1] = 1
        except:
            continue

    return np.array(rt_list, dtype=np.float64)


def convert_list(x: str) -> List:
    '''
    x: course_id
    '''
    try:
        if ' ' in x:
            return x.split(' ')
        return [x]
    except:
        return []


def convert_gender(x: str) -> int:
    '''
    x: gender
    '''
    g = 0
    try:
        if x == 'male':
            g = 0
        elif x == 'female':
            g = 1
        elif x == 'other':
            g = 2
    except:
        g = -1

    return g


def convert_subgroup_vector(x: str, subgroups_dict, split=True) -> List[int]:
    global group_subgroup_pair

    ## preprocess ##
    group_pairs = []
    if ',' in x:
        group_pairs = x.split(',')
    else:
        group_pairs = [x]

    ## variable ##
    rt_list = [0] * len(subgroups_dict)

    for group_pair in group_pairs:
        group_pair = group_pair.strip()
        if group_pair == '':
            continue

        if split:
            group, subgroup = group_pair.split('_')

        else:
            subgroup = group_pair

        subgroup_id = get_subgroup_id(subgroups_dict, subgroup)
        if subgroup_id != None:
            if split:
                c_subgroup = group_subgroup_pair.get(group)
                if c_subgroup == None:
                    group_subgroup_pair[group] = [subgroup_id]
                    update_group_list(subgroups_dict)
                elif subgroup_id not in c_subgroup:
                    c_subgroup.append(subgroup_id)
                    group_subgroup_pair[group] = c_subgroup
                    update_group_list(subgroups_dict)

            rt_list[subgroup_id] = 1

    return np.array(rt_list, dtype=np.float64)


def convert_course_text2vec_score(
    x,
    subgroups_num,
    topk_num,
    model_embedder,
    subgroups_embeddings,
    # subgroups_list,
):
    subgroups_scores = []
    multiplier = 0
    for items in x:
        items = str(items)
        if ',' in items:
            # 'groups', 'sub_groups'
            items = items.split(',')
            _topk_num = 1
            multiplier = 1
        else:
            items = [items]
            _topk_num = topk_num
            multiplier = 0.1

        for item in items:
            item_embedding = model_embedder.encode(str(item))
            hits = semantic_search(item_embedding,
                                   subgroups_embeddings,
                                   top_k=_topk_num)
            hits = hits[0]  # Get the hits for the first query

            corpus_ids, scores = [], []

            c_subgroups_score = np.zeros(subgroups_num)
            for hit in hits:
                corpus_ids.append(hit['corpus_id'])
                scores.append(hit['score'])

            # w = np.ones(_topk_num) / np.arange(1, _topk_num + 1)
            # scores = np.multiply(np.array(scores), w).tolist()
            scores = (np.array(scores) * multiplier).tolist()

            for c, s in zip(corpus_ids, scores):
                c_subgroups_score[c] = s
            subgroups_scores.append(c_subgroups_score)

    subgroups_scores = np.average(np.array(subgroups_scores), axis=0)

    # ## checker ##
    # print(x['course_name'])
    # print(x['groups'])
    # print(x['sub_groups'])
    # subgroups_scores_argmax = np.argsort(-1 * subgroups_scores)
    # for id in subgroups_scores_argmax:
    #     print(subgroups_list[id], subgroups_scores[id])
    # print()
    # print(subgroups_scores.shape)
    # print(subgroups_scores)
    # input()

    return subgroups_scores


def array_norm(a):
    _a = np.array(a).copy()
    _a_max = np.max(_a)
    if _a_max != 0:
        _a = _a / _a_max
    return _a


def convert_merge_stages(a, b):
    '''
    a: 'v_sub_groups'
    b: 'v_text2vec'
    '''
    epsilon = 0.1

    a = np.array(a) + epsilon
    a = array_norm(a)

    b = np.array(b)
    c = np.multiply(a, b)

    # c = array_norm(c)

    return c


## get_dataframe ##


def get_dataframe_group(name) -> pd.DataFrame:
    '''
    name: 'val_seen_group.csv', 'val_unseen_group.csv'
    '''
    df = pd.DataFrame(
        pd.read_csv(BASE_DIR + name, dtype={
            'user_id': str,
            'subgroup': str,
        }))
    df['l_subgroup'] = df[['subgroup'
                           ]].progress_apply(lambda x: convert_list(*x),
                                             axis=1)
    df['v_subgroup'] = df[['subgroup']].progress_apply(
        lambda x: convert_subgroup_vector_from_id(*x), axis=1)
    return df


def get_dataframe_users(name, subgroups_dict) -> pd.DataFrame:
    '''
    name: 'users.csv'
    '''
    df = pd.DataFrame(
        pd.read_csv(BASE_DIR + name,
                    dtype={
                        'user_id': str,
                        'gender': str,
                        'occupation_titles': str,
                        'interests': str,
                        'recreation_names': str,
                    }))

    df['occupation_titles'].fillna(value='', inplace=True)
    df['interests'].fillna(value='', inplace=True)
    df['recreation_names'].fillna(value='', inplace=True)

    df['gender'] = df[['gender']].progress_apply(lambda x: convert_gender(*x),
                                                 axis=1)
    df['v_interests'] = df[['interests']].progress_apply(
        lambda x: convert_subgroup_vector(*x, subgroups_dict), axis=1)
    # print(df['v_interests'])

    return df


def get_dataframe_courses(name, subgroups_dict) -> pd.DataFrame:
    '''
    name: 'courses.csv'
    '''
    df = pd.DataFrame(
        pd.read_csv(BASE_DIR + name,
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
                    }))

    df['groups'].fillna(value='', inplace=True)
    df['sub_groups'].fillna(value='', inplace=True)

    df['v_sub_groups'] = df[['sub_groups']].progress_apply(
        lambda x: convert_subgroup_vector(*x, subgroups_dict, False), axis=1)
    # print(df['v_sub_groups'])

    return df


def get_dataframe_train(name) -> pd.DataFrame:
    '''
    name: 'train.csv'
    '''
    df = pd.DataFrame(
        pd.read_csv(BASE_DIR + name,
                    dtype={
                        'user_id': str,
                        'course_id': str,
                    }))

    df['course_id'] = df[['course_id'
                          ]].progress_apply(lambda x: convert_list(*x), axis=1)
    # print(df)

    train_list = []
    for _, c_row in df.iterrows():
        c_user = c_row['user_id']
        c_courses = c_row['course_id']
        for c_course in c_courses:
            c_train = {
                'user_id': c_user,
                'course_id': c_course,
            }
            train_list.append(c_train)

    df_flatten = pd.DataFrame(train_list)
    # print(df_flatten)

    return df_flatten


def get_dataframe_courses_sub_groups(name, df_courses) -> pd.DataFrame:
    global group_list

    #
    '''
    col: 'user_id', 'course_id'
    '''
    df_train = get_dataframe_train(name)

    df_train = pd.merge(df_train, df_courses, on='course_id')
    df_train = df_train[['user_id', 'v_sub_groups']].copy()

    train_dict = {}
    for i, c_row in df_train.iterrows():
        c_user = c_row['user_id']
        c_v_sub_groups = c_row['v_sub_groups'].copy()
        c_v = train_dict.get(c_user)

        if c_v is None:
            c_v = c_v_sub_groups
        else:
            c_v_sub_groups_mul_mean = np.mean(
                np.multiply(
                    c_v_sub_groups.reshape(1, -1),
                    np.array(group_list),
                ),
                axis=1,
            ).reshape(-1)

            c_v += c_v_sub_groups + 0.2 * c_v_sub_groups_mul_mean
        train_dict[c_user] = c_v

    _df_train = pd.DataFrame(list(train_dict.items()),
                             columns=['user_id', 'v_sub_groups'])
    # print(_df_train)

    return _df_train


def get_dataframe_courses_text2vec(name, df_courses) -> pd.DataFrame:
    # model
    model_embedder = get_model()

    ## constant ##
    subgroups_dict = read_csv_subgroups('subgroups.csv')
    subgroups_num = len(subgroups_dict)
    subgroups_list = list(subgroups_dict.keys())

    topk_num = 91

    subgroups_embeddings = model_embedder.encode(subgroups_list)

    # dataframe
    '''
    col: 'course_id', 'course_name', 'course_price', 'teacher_id',
         'teacher_intro', 'groups', 'sub_groups', 'topics', 'course_published_at_local',
         'description', 'will_learn', 'required_tools', 'recommended_background', 'target_group',
         'v_sub_groups'
    '''

    df_courses['v_text2vec'] = df_courses[[
        'course_name',
        'teacher_intro',
        'groups',
        'sub_groups',
        'topics',
        'description',
        'will_learn',
        'required_tools',
        'recommended_background',
        'target_group',
    ]].progress_apply(
        lambda x: convert_course_text2vec_score(
            x,
            subgroups_num,
            topk_num,
            model_embedder,
            subgroups_embeddings,
            # subgroups_list,
        ),
        axis=1,
    )
    # print(df_courses)

    #
    '''
    col: 'user_id', 'course_id'
    '''
    df_train = get_dataframe_train(name)

    df_train = pd.merge(df_train, df_courses, on='course_id')
    df_train = df_train[['user_id', 'v_text2vec']].copy()

    train_dict = {}
    for _, c_row in df_train.iterrows():
        c_user = c_row['user_id']
        c_v_sub_groups = c_row['v_text2vec'].copy()
        c_v = train_dict.get(c_user)

        if c_v is None:
            c_v = c_v_sub_groups
        else:
            c_v += c_v_sub_groups
        train_dict[c_user] = c_v

    _df_train = pd.DataFrame(list(train_dict.items()),
                             columns=['user_id', 'v_text2vec'])
    # print(_df_train)

    return _df_train


def get_dataframe_test(name) -> pd.DataFrame:
    '''
    name: 'test_seen.csv'
    '''
    df = pd.DataFrame(
        pd.read_csv(BASE_DIR + name,
                    dtype={
                        'user_id': str,
                        'course_id': str,
                    }))

    # df['course_id'] = df[['course_id']].progress_apply(lambda x: get_list(*x), axis=1)

    return df


## workhouse ##


def dataset_workhouse(df_users, df_courses, mode='Train'):
    if mode == 'Train':
        # ## flatten course for each user ##
        # '''
        # col: 'user_id', 'v_sub_groups'
        # '''
        # df_courses_sub_groups = get_dataframe_courses_sub_groups(
        #     'train.csv', df_courses)

        # df = pd.merge(df_courses_sub_groups, df_users, on='user_id')
        # df = df[['gender', 'v_interests', 'v_sub_groups']]

        # ## course text2vec for each user ##
        # '''
        # col: 'user_id', 'v_text2vec'
        # '''
        # df_courses_text2vec = get_dataframe_courses_text2vec(
        #     'train.csv', df_courses)

        # df = pd.merge(df_courses_text2vec, df_users, on='user_id')
        # df = df[['gender', 'v_interests', 'v_text2vec']]

        ## course get text2vec sub_groups for each user ##
        df_courses_sub_groups = get_dataframe_courses_sub_groups(
            'train.csv', df_courses)
        print(df_courses_sub_groups)

        df_courses_text2vec = get_dataframe_courses_text2vec(
            'train.csv', df_courses)
        print(df_courses_text2vec)

        df_merge = pd.merge(df_courses_sub_groups,
                            df_courses_text2vec,
                            on='user_id')
        df_merge['vector'] = df_merge[['v_sub_groups',
                                       'v_text2vec']].progress_apply(
                                           lambda x: convert_merge_stages(*x),
                                           axis=1)

        df = pd.merge(df_merge, df_users, on='user_id')
        df = df[['gender', 'v_interests', 'vector']]
        print(df)

    elif mode == 'Eval_Seen':
        '''
        col: 'user_id', 'subgroup', 'l_subgroup', 'v_subgroup'
        '''
        df_group = get_dataframe_group('val_seen_group.csv')

        df = pd.merge(df_group, df_users, on='user_id')
        df = df[['gender', 'v_interests', 'v_subgroup']]
    elif mode == 'Eval_UnSeen':
        '''
        col: 'user_id', 'subgroup', 'l_subgroup', 'v_subgroup'
        '''
        df_group = get_dataframe_group('val_unseen_group.csv')

        df = pd.merge(df_group, df_users, on='user_id')
        df = df[['gender', 'v_interests', 'v_subgroup']]
    elif mode == 'Test_Seen':
        df_test = get_dataframe_test('test_seen.csv')
        df = pd.merge(df_test, df_users, on='user_id')
        df = df[['gender', 'v_interests']]
    elif mode == 'Test_UnSeen':
        df_test = get_dataframe_test('test_unseen.csv')
        df = pd.merge(df_test, df_users, on='user_id')
        df = df[['gender', 'v_interests']]
    else:
        raise KeyError

    return df


def main():
    return


if __name__ == '__main__':
    main()
