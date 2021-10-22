#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Mining final project: data preprocessing

Group: G07
"""
import os
import numpy as np
import pandas as pd

# 1. Read Data
train_dir = "data/yelp_training_set/"
test_dir = "data/yelp_test_set/"
training_set_business = pd.read_json(os.path.join(train_dir, 'yelp_training_set_business.json'), lines=True)
training_set_checkin = pd.read_json(os.path.join(train_dir, 'yelp_training_set_checkin.json'), lines=True)
training_set_review = pd.read_json(os.path.join(train_dir, 'yelp_training_set_review.json'), lines=True)
training_set_user = pd.read_json(os.path.join(train_dir, 'yelp_training_set_user.json'), lines=True)
test_set_business = pd.read_json(os.path.join(test_dir, 'yelp_test_set_business.json'), lines=True)
test_set_checkin = pd.read_json(os.path.join(test_dir, 'yelp_test_set_checkin.json'), lines=True)
test_set_review = pd.read_json(os.path.join(test_dir, 'yelp_test_set_review.json'), lines=True)
test_set_user = pd.read_json(os.path.join(test_dir, 'yelp_test_set_user.json'), lines=True)

# 2. Extract Features for Training
# (a) Review
print("start to extract feature of review from training data...")
X = training_set_review[['business_id','user_id']]
training_record = pd.Timestamp(2013, 1, 19)
X = X.assign(review_days_from_written_to_recorded = [t.days for t in (training_record - training_set_review['date'])])
X = X.assign(review_text_length = training_set_review['text'].apply(len))
# (b) Business
print("start to extract feature of business from training data...")
# sort business_id of training_set_business by business_id of training_set_review
training_set_business_tmp = training_set_business.set_index('business_id')
training_set_business_tmp = training_set_business_tmp.reindex(index=X['business_id'])
training_set_business_tmp = training_set_business_tmp.reset_index()
X = X.assign(business_stars = training_set_business_tmp['stars'])
X = X.assign(business_review_count = training_set_business_tmp['review_count'])
X = X.assign(business_open = training_set_business_tmp['open'].apply(int))
# (c) User
print("start to extract feature of user from training data...")
training_set_user_tmp = training_set_user.set_index('user_id')
training_set_user_tmp = training_set_user_tmp.reindex(index=X['user_id'], fill_value='?')
training_set_user_tmp = training_set_user_tmp.reset_index()
# missing user data preprocessing
median_user_average_stars = np.median(np.hstack([test_set_user['average_stars'], training_set_user['average_stars']]))
training_set_user_tmp['average_stars'].replace('?', median_user_average_stars, inplace=True)
median_user_review_count = np.median(np.hstack([test_set_user['review_count'], training_set_user['review_count']]))
training_set_user_tmp['review_count'].replace('?', median_user_review_count, inplace=True)
median_user_votes = pd.DataFrame(training_set_user['votes'].tolist()).apply(np.median).to_dict()
for i in list(training_set_user_tmp.index):
    if type(training_set_user_tmp.loc[i,'votes']) is not dict:
        training_set_user_tmp.at[i,'votes'] = median_user_votes
# assign after data preprocessing
X = X.assign(user_average_stars = training_set_user_tmp['average_stars'])
X = X.assign(user_review_count = training_set_user_tmp['review_count'])
X = X.assign(user_votes_funny = [v['funny'] for v in training_set_user_tmp['votes']])
X = X.assign(user_votes_useful = [v['useful'] for v in training_set_user_tmp['votes']])
X = X.assign(user_votes_cool = [v['cool'] for v in training_set_user_tmp['votes']])
Y = training_set_review[['business_id','user_id']]
Y = Y.assign(votes_useful = [v['useful'] for v in training_set_review['votes']])

# 3. Extract Features for Test
# (a) Review
print("start to extract feature of review from test data...")
test_X = test_set_review[['business_id','user_id']]
test_record = pd.Timestamp(2013, 3, 12)
test_X = test_X.assign(review_days_from_written_to_recorded = [t.days for t in (test_record - test_set_review['date'])])
test_X = test_X.assign(review_text_length = test_set_review['text'].apply(len))
# (b) Business
print("start to extract feature of business from test data...")
business_tmp = pd.concat([training_set_business, test_set_business], ignore_index=True)
business_tmp = business_tmp.set_index('business_id')
business_tmp = business_tmp.reindex(index=test_X['business_id'])
business_tmp = business_tmp.reset_index()
test_X = test_X.assign(business_stars = business_tmp['stars'])
test_X = test_X.assign(business_review_count = business_tmp['review_count'])
test_X = test_X.assign(business_open = business_tmp['open'].apply(int))
# (c) User
print("start to extract feature of user from test data...")
test_set_user_tmp = test_set_user.copy()
test_set_user_tmp['votes'] = '?'
user_tmp = pd.concat([training_set_user, test_set_user_tmp], ignore_index=True)
user_tmp = user_tmp.set_index('user_id')
user_tmp = user_tmp.reindex(index=test_X['user_id'], fill_value='?')
user_tmp = user_tmp.reset_index()
# missing user data preprocessing
user_tmp['average_stars'].replace('?', median_user_average_stars, inplace=True)
user_tmp['review_count'].replace('?', median_user_review_count, inplace=True)
for i in list(user_tmp.index):
    if type(user_tmp.loc[i,'votes']) is not dict:
        user_tmp.at[i,'votes'] = median_user_votes
# assign after data preprocessing
test_X = test_X.assign(user_average_stars = user_tmp['average_stars'])
test_X = test_X.assign(user_review_count = user_tmp['review_count'])
test_X = test_X.assign(user_votes_funny = [v['funny'] for v in user_tmp['votes']])
test_X = test_X.assign(user_votes_useful = [v['useful'] for v in user_tmp['votes']])
test_X = test_X.assign(user_votes_cool = [v['cool'] for v in user_tmp['votes']])

# 4. Save Features
# Remove user and business id
X = X.iloc[:,2:]
Y = Y.iloc[:,2:]
test_X = test_X.iloc[:,2:]
# Save
X.to_pickle("feature/X.pkl")
Y.to_pickle("feature/Y.pkl")
test_X.to_pickle("feature/test_X.pkl")
print("Finish!")
print("Please check following files in ./feature/")
print("1. X.pkl \n2. Y.pkl \n3. test_X.pkl")