from typing import List, Dict

import numpy as np
import pandas as pd

from tqdm import tqdm

tqdm.pandas()

from text2vec import SentenceModel, semantic_search

BASE_DIR = './hahow/data/'


def get_subgroups(name) -> Dict:
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


def get_subgroup_vector_from_id(x):
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

    return np.array(rt_list)


def get_list(x: str) -> List:
    '''
    x: course_id
    '''
    try:
        if ' ' in x:
            return x.split(' ')
        return [x]
    except:
        return []


def get_group(name) -> pd.DataFrame:
    '''
    name: 'val_seen_group.csv', 'val_unseen_group.csv'
    '''
    df = pd.DataFrame(
        pd.read_csv(BASE_DIR + name, dtype={
            'user_id': str,
            'subgroup': str,
        }))
    df['l_subgroup'] = df[['subgroup']].progress_apply(lambda x: get_list(*x),
                                                       axis=1)
    df['v_subgroup'] = df[['subgroup']].progress_apply(
        lambda x: get_subgroup_vector_from_id(*x), axis=1)
    return df


def get_gender(x: str) -> int:
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


group_subgroup_pair = {}
group_list = [0] * 91  # dangerous move


def update_group_list(subgroups_dict):
    global group_subgroup_pair
    global group_list

    for i in range(len(subgroups_dict)):
        groups = set()

        for k, v in group_subgroup_pair.items():
            if i in v:
                groups.update(set(v))

        if i in groups:
            groups.remove(i)
        group_list[i] = list(groups)
    return


def get_subgroup_id(subgroups_dict, c_subgroup):
    c_subgroup_id = subgroups_dict.get(c_subgroup)
    if c_subgroup_id != None:
        return c_subgroup_id - 1
    # raise KeyError(f"HI {c_subgroup}.")
    return None


def get_subgroup_vector(x: str, subgroups_dict, split=True) -> List[int]:
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

    return np.array(rt_list)


def get_users(name, subgroups_dict) -> pd.DataFrame:
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

    df['gender'] = df[['gender']].progress_apply(lambda x: get_gender(*x),
                                                 axis=1)
    df['v_interests'] = df[['interests']].progress_apply(
        lambda x: get_subgroup_vector(*x, subgroups_dict), axis=1)
    # print(df['v_interests'])

    return df


def get_courses(name, subgroups_dict) -> pd.DataFrame:
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
        lambda x: get_subgroup_vector(*x, subgroups_dict, False), axis=1)
    # print(df['v_sub_groups'])

    return df


def get_train(name) -> pd.DataFrame:
    '''
    name: 'train.csv'
    '''
    df = pd.DataFrame(
        pd.read_csv(BASE_DIR + name,
                    dtype={
                        'user_id': str,
                        'course_id': str,
                    }))

    df['course_id'] = df[['course_id']].progress_apply(lambda x: get_list(*x),
                                                       axis=1)
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

    ## something that get_label will do ##
    # print(df_flatten)

    # df_train = pd.merge(df_flatten, df_courses, on='course_id')
    # df_train = df_train[['user_id', 'v_sub_groups']]
    # # print(df_train)
    # # print(df_train['v_sub_groups'][200])
    # # print(df_train['v_sub_groups'][100])
    # # print(df_train['v_sub_groups'][200] + df_train['v_sub_groups'][100])
    # # print(type(df_train['v_sub_groups'][0]))
    # # input()

    # train_dict = {}
    # for _, c_row in df_train.iterrows():
    #     c_user = c_row['user_id']
    #     c_v_sub_groups = c_row['v_sub_groups']
    #     c_v = train_dict.get(c_user)

    #     if c_v is None:
    #         c_v = c_v_sub_groups
    #     else:
    #         c_v += c_v_sub_groups
    #     train_dict[c_user] = c_v

    # _df_train = pd.DataFrame(list(train_dict.items()),
    #                          columns=['user_id', 'v_sub_groups'])

    # print(_df_train)
    ## end of something ##

    return df_flatten


## deprecate ##


def get_label_deprecate():
    global group_list

    df1 = pd.read_csv(BASE_DIR + 'train.csv')
    dict1 = df1.set_index('user_id').to_dict('index')

    df2 = pd.read_csv(BASE_DIR + 'subgroups.csv')
    dict2 = df2.set_index('subgroup_name').to_dict('index')

    df3 = pd.read_csv(BASE_DIR + 'courses.csv').dropna(subset=['sub_groups'])
    dict3 = df3.set_index('course_id').to_dict('index')

    for c in dict3:
        a = []
        for s in dict3[c]['sub_groups'].split(','):
            a.append(dict2[s]['subgroup_id'] - 1)
        dict3[c] = a

    for u in dict1:
        a = []
        for c in dict1[u]['course_id'].split():
            if c in dict3:
                a.extend(dict3[c])
        dict1[u] = a

    for u in dict1:
        vec = np.zeros(91)
        for i in dict1[u]:
            vec[i] += 1

            for c_id in group_list[i]:
                vec[c_id] += 0.2

        # normalize
        if np.max(vec) != 0:
            vec /= np.max(vec)
        dict1[u] = vec

    # np.save('label.npy', dict1)
    # dict1 = np.load('label1.npy', allow_pickle=True)
    # print(dict1)

    df = pd.DataFrame(list(dict1.items()), columns=['user_id', 'v_sub_groups'])
    # print(df)

    return df


def get_model() -> SentenceModel:
    model = SentenceModel('shibing624/text2vec-base-chinese')
    return model


def get_course_text2vec_score(
    x,
    subgroups_num,
    model_embedder,
    subgroups_embeddings,
    # subgroups_list,
):
    '''
    col: 'course_name', 'teacher_intro', 'groups', 'sub_groups', 'topics',
         'description', 'will_learn', 'recommended_background', 'target_group'
    '''

    subgroups_scores = []

    bad_str = '更多'

    for items in x:
        items = str(items)
        if ',' in items:
            items = items.split(',')
            for i in range(len(items)):
                if bad_str in items[i]:
                    items[i] = items[i].replace(bad_str, '')
        else:
            items = [items]

        for item in items:
            item_embedding = model_embedder.encode(str(item))
            hits = semantic_search(item_embedding,
                                   subgroups_embeddings,
                                   top_k=subgroups_num)
            hits = hits[0]  # Get the hits for the first query

            c_subgroups_score = np.zeros(subgroups_num)
            for hit in hits:
                c_subgroups_score[hit['corpus_id']] = hit['score']
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


def get_courses_text2vec(name, df_courses) -> pd.DataFrame:
    # model
    model_embedder = get_model()

    ## constant ##
    subgroups_dict = get_subgroups('subgroups.csv')
    subgroups_num = len(subgroups_dict)
    subgroups_list = list(subgroups_dict.keys())

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
        lambda x: get_course_text2vec_score(
            x,
            subgroups_num,
            model_embedder,
            subgroups_embeddings,
            # subgroups_list,
        ),
        axis=1,
    )

    #
    '''
    col: 'user_id', 'course_id'
    '''
    df_train = get_train(name)

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

    return _df_train


def get_label_hi() -> pd.DataFrame:
    label = []
    df1 = pd.read_csv('data/train.csv')
    df2 = pd.read_csv('data/courses.csv').fillna('')
    df3 = pd.read_csv('data/subgroups.csv')

    model = get_model()
    vs = model.encode(df3['subgroup_name'])

    for _, row in tqdm(df1.iterrows(), total=df1.shape[0]):
        n = 0
        v = np.zeros(91)

        for i in row['course_id'].split():
            c = df2.loc[df2['course_id'] == i].squeeze()

            if c['groups'] != -1:
                for g in c['groups'].split(','):
                    v1 = model.encode(g)
                    v += [np.linalg.norm(v1 - v2) for v2 in vs]
                n += len(c['groups'].split(','))

            if c['sub_groups'] != -1:
                for s in c['sub_groups'].split(','):
                    v1 = model.encode(s)
                    v += [np.linalg.norm(v1 - v2) for v2 in vs]
                n += len(c['sub_groups'].split(','))

            if c['course_name'] != -1:
                v1 = model.encode(c['course_name'])
                v += [np.linalg.norm(v1 - v2) for v2 in vs]
                n += 1

        v = v if n == 0 else v / n
        label.append([row['user_id'], v])

    label = pd.DataFrame(label, columns=['user_id', 'vector'])
    label.set_index('user_id').to_csv('label.csv')
    return label


def get_test(name) -> pd.DataFrame:
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


def get_dataset(df_users, df_courses, mode='Train'):
    if mode == 'Train':
        ## flatten course for each user ##
        '''
        col: 'user_id', 'course_id'
            'v_sub_groups'
        '''
        df_train = get_train('train.csv')

        df = pd.merge(df_train, df_courses, on='course_id')
        df = df[['user_id', 'course_id', 'v_sub_groups']]

        df = pd.merge(df, df_users, on='user_id')
        df = df[['gender', 'v_interests', 'v_sub_groups']]
        # ## course text2vec for each user ##
        # '''
        # col: 'user_id', 'v_text2vec'
        # '''
        # df_courses_text2vec = get_courses_text2vec('train.csv', df_courses)

        # df = pd.merge(df_courses_text2vec, df_users, on='user_id')
        # df = df[['gender', 'v_interests', 'v_text2vec']]
    elif mode == 'Eval_Seen':
        '''
        col: 'user_id', 'subgroup', 'l_subgroup', 'v_subgroup'
        '''
        df_group = get_group('val_seen_group.csv')

        df = pd.merge(df_group, df_users, on='user_id')
        df = df[['gender', 'v_interests', 'v_subgroup']]
    elif mode == 'Eval_UnSeen':
        '''
        col: 'user_id', 'subgroup', 'l_subgroup', 'v_subgroup'
        '''
        df_group = get_group('val_unseen_group.csv')

        df = pd.merge(df_group, df_users, on='user_id')
        df = df[['gender', 'v_interests', 'v_subgroup']]
    elif mode == 'Test_Seen':
        df_test = get_test('test_seen.csv')
        df = pd.merge(df_test, df_users, on='user_id')
        df = df[['gender', 'v_interests']]
    elif mode == 'Test_UnSeen':
        df_test = get_test('test_unseen.csv')
        df = pd.merge(df_test, df_users, on='user_id')
        df = df[['gender', 'v_interests']]
    else:
        raise KeyError

    return df


def main():
    # df = get_dataset()
    # print(df)

    subgroups_dict = get_subgroups('subgroups.csv')
    df_courses = get_courses('courses.csv', subgroups_dict)
    df = get_courses_text2vec(df_courses)
    print(df)

    return


if __name__ == '__main__':
    main()
