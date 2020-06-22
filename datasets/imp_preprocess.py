# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:39:25 2020

@author: JEFF
"""

import numpy as np
import pandas as pd
import os
import pickle

data = pd.read_csv("./trivago/train.csv")
del data['user_id'], data['platform'], data['city']
del data['device'], data['current_filters'], data['prices']

data = data[data.action_type == "clickout item"]
del data['action_type'], data['step'] 

#%%
# drop the session that only contain one click item
data = data[data['session_id'].duplicated(keep=False)]

# drop the item that appear less than 5 times in all dataset
iid_items = data.groupby('reference').count()
drop_list = []
for index, row in iid_items.iterrows():
    if row['session_id'] < 5:
        drop_list.append(index)

data = data[~data.reference.isin(drop_list)]

# drop the session that only contain one click item
data = data[data['session_id'].duplicated(keep=False)]
# reset index
data = data.reset_index(drop=True)

#%%
# split train and test
s_id = data.session_id.unique()
split_point_sid = s_id[-50000]
# get the index of split_point_sid in the original dataframe
split_point_idx = data.index[data['session_id'] == split_point_sid].tolist()[0]

train = data[:split_point_idx]
test = data[split_point_idx:]
# drop the item in test that doesn't appear in train
iid_items_train = train.reference.unique().tolist()
test = test[test.reference.isin(iid_items_train)]
# drop the session that only contain one click item
test = test[test['session_id'].duplicated(keep=False)]
#%%
item_ctr = 1
item_dict = {}
def split_click_impression(dataset):
    global item_ctr
    # list_ids = []  # store session id
    list_seqs = [] # store item id which start from 1
    list_imp = []
    current_sid = dataset.iloc[0]['session_id']
    outseq_click = []
    outseq_imp = []
    for idx, row in dataset.iterrows():
        if row['session_id'] == current_sid:
            if row['reference'] in item_dict:
                outseq_click += [item_dict[row['reference']]]
            else:
                outseq_click += [item_ctr]
                item_dict[row['reference']] = item_ctr
                item_ctr += 1
            imp_temp = row['impressions'].split("|")
            imp_list = []
            for ref in imp_temp:
                if ref in item_dict:
                    imp_list += [item_dict[ref]]
                else:
                    imp_list += [item_ctr]
                    item_dict[ref] = item_ctr
                    item_ctr += 1
            outseq_imp += [imp_list]
        else:
            current_sid = row['session_id']
            list_seqs += [outseq_click]
            list_imp += [outseq_imp]
            outseq_click = []
            outseq_imp = []
            if row['reference'] in item_dict:
                outseq_click += [item_dict[row['reference']]]
            else:
                outseq_click += [item_ctr]
                item_dict[row['reference']] = item_ctr
                item_ctr += 1
            imp_temp = row['impressions'].split("|")
            imp_list = []
            for ref in imp_temp:
                if ref in item_dict:
                    imp_list += [item_dict[ref]]
                else:
                    imp_list += [item_ctr]
                    item_dict[ref] = item_ctr
                    item_ctr += 1
            outseq_imp += [imp_list]
    return list_seqs, list_imp

train_click, train_imp = split_click_impression(train)
test_click, test_imp = split_click_impression(test)

#%%
def process_click_imp(click, imp):
    out_imp = []
    out_click = []
    out_tar = []
    for idx, seq, imp_ in zip(range(len(click)), click, imp):
        # for i in range(1, len(seq)):
            # out_tar += [seq[-i]]
            # out_click += [seq[:-i]]
            # out_imp += [imp_[-i]]

            # todo 找target在imp的位置, 如果不在就剃除
        for i in range(0, len(seq)):
            if seq[i] in imp_[i]:
                out_tar += [imp_[i].index(seq[i])]  # impression index
                out_click += [seq[:i]]
                out_imp += [imp_[i]]
    return out_imp, out_click, out_tar

# todo target要改成imp的index
new_train_i, new_train_c, new_train_t = process_click_imp(train_click, train_imp)
new_test_i, new_test_c, new_test_t = process_click_imp(test_click, test_imp)
#%%

def fixed_imp_len(imp):
    for imp_ in imp:
        while(len(imp_) != 25):
            imp_ += [0]

fixed_imp_len(new_train_i)
fixed_imp_len(new_test_i)

final_train = (new_train_c, new_train_t, new_train_i)  # session seq, target, impression
final_test = (new_test_c, new_test_t, new_test_i)
print(len(new_train_c))     # old trivago: 414670, new trivago: 397565
print(len(new_test_c))     # old trivago: 86056, new trivago: 109969
# total_new = 507534, but total_old = 500726
print(len(iid_items_train))

if not os.path.exists('trivago_imp'):
    os.makedirs('trivago_imp')
pickle.dump(final_train, open('trivago_imp/train.txt', 'wb'))
pickle.dump(final_test, open('trivago_imp/test.txt', 'wb'))







