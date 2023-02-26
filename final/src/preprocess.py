from typing import List, Dict

import numpy as np
import pandas as pd

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


def get_subgroup_vector(x: str, subgroups_dict, split=True) -> List[int]:
    ## preprocess ##
    group_pairs = []
    if ',' in x:
        group_pairs = x.split(',')
    else:
        group_pairs = [x]

    ## variable ##
    rt_list = [0] * len(subgroups_dict)
    # subgroup_set = set()

    for group_pair in group_pairs:
        group_pair = group_pair.strip()
        if group_pair == '':
            continue

        if split:
            group, subgroup = group_pair.split('_')
        else:
            subgroup = group_pair

        # subgroup_set.add(subgroup)

        if subgroup in subgroups_dict.keys():
            rt_list[subgroups_dict[subgroup] - 1] = 1

    return np.array(rt_list)


def get_users(name, subgroups_dict):
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

    df['gender'] = df[['gender']].apply(lambda x: get_gender(*x), axis=1)
    df['v_interests'] = df[['interests']].apply(
        lambda x: get_subgroup_vector(*x, subgroups_dict), axis=1)
    # print(df['v_interests'])

    return df


def get_courses(name, subgroups_dict):
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

    df['v_sub_groups'] = df[['sub_groups']].apply(
        lambda x: get_subgroup_vector(*x, subgroups_dict, False), axis=1)
    # print(df['v_sub_groups'])

    return df


def get_list(x: str) -> List:
    '''
    x: course_id
    '''
    if ' ' in x:
        return x.split(' ')
    return [x]


def get_train(name, df_courses):
    '''
    name: 'train.csv'
    '''
    df = pd.DataFrame(
        pd.read_csv(BASE_DIR + name,
                    dtype={
                        'user_id': str,
                        'course_id': str,
                    }))

    df['course_id'] = df[['course_id']].apply(lambda x: get_list(*x), axis=1)
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

    df_train = pd.merge(df_flatten, df_courses, on='course_id')
    df_train = df_train[['user_id', 'v_sub_groups']]
    # print(df_train)
    # print(df_train['v_sub_groups'][200])
    # print(df_train['v_sub_groups'][100])
    # print(df_train['v_sub_groups'][200] + df_train['v_sub_groups'][100])
    # print(type(df_train['v_sub_groups'][0]))
    # input()

    train_dict = {}
    for _, c_row in df_train.iterrows():
        c_user = c_row['user_id']
        c_v_sub_groups = c_row['v_sub_groups']
        c_v = train_dict.get(c_user)

        if c_v is None:
            c_v = c_v_sub_groups
        else:
            c_v += c_v_sub_groups
        train_dict[c_user] = c_v

    _df_train = pd.DataFrame(list(train_dict.items()),
                             columns=['user_id', 'v_sub_groups'])

    print(_df_train)
    return _df_train


def get_label():

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
        if np.max(vec) != 0:
            vec /= np.max(vec)
        dict1[u] = vec

    # np.save('label.npy', dict1)
    # dict1 = np.load('label1.npy', allow_pickle=True)
    # print(dict1)

    df = pd.DataFrame(list(dict1.items()), columns=['user_id', 'v_sub_groups'])
    # print(df)

    return df

def get_test(name):
    '''
    name: 'test_seen.csv'
    '''
    df = pd.DataFrame(
        pd.read_csv(BASE_DIR + name,
                    dtype={
                        'user_id': str,
                        'course_id': str,
                    }))

    # df['course_id'] = df[['course_id']].apply(lambda x: get_list(*x), axis=1)

    return df


def get_dataset(mode='Train'):
    ## constant ##
    subgroups_dict = get_subgroups('subgroups.csv')

    ## dataframe ##

    #
    '''
    col: 'user_id', 'gender', 'occupation_titles', 'interests', 'recreation_names',
         'v_interests'
    '''
    df_users = get_users('users.csv', subgroups_dict)

    #
    '''
    col: 'course_id', 'course_name', 'course_price', 'teacher_id',
         'teacher_intro', 'groups', 'sub_groups', 'topics', 'course_published_at_local',
         'description', 'will_learn', 'required_tools', 'recommended_background', 'target_group',
         'v_sub_groups'
    '''
    df_courses = get_courses('courses.csv', subgroups_dict)

    # #
    # '''
    # col: 'user_id', 'course_id'
    #      'v_sub_groups'
    # '''
    # df_train = get_train('train.csv', df_courses)

    # df = pd.merge(df_train, df_users, on='user_id')
    # df = df[['user_id', 'v_interests', 'v_sub_groups']]

    np_df = None
    if mode=='Train':
        df_label = get_label()

        df = pd.merge(df_label, df_users, on='user_id')
        df = df[['v_interests', 'v_sub_groups']]

        np_df = df.to_numpy()
    else:
        df_test = get_test('test_seen.csv')
        df = pd.merge(df_test, df_courses, left_on='user_id')
        df = df[['v_interests']]

        np_df = df.to_numpy()

    return np_df


def main():
    df = get_dataset()
    print(df)
    return


if __name__ == '__main__':
    main()
