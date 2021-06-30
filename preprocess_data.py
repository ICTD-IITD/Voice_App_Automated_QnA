import config
import requests
from bs4 import BeautifulSoup
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import random
import time

MAX_LEN = config.MAX_LEN
COLUMN_NAMES = config.COLUMN_NAMES
TRAIN_COLUMN = config.TRAIN_COLUMN
TEST_COLUMN = config.TEST_COLUMN
DATA = config.DATA
def preprocess_text(tokenizer, ques1, ques2):
    input_ids = []
    segment_ids = []
    attention_masks = []
    for (q1,q2) in zip(ques1, ques2):
        q1 = '[CLS] ' + q1 + ' [SEP] '
        q2 = q2 + ' [SEP] '

        token_q1 = tokenizer.tokenize(q1)
        token_q2 = tokenizer.tokenize(q2)

        token = token_q1 + token_q2
        segment_id = [0] * len(token_q1) + [1] * len(token_q2)
        attention_mask = [1]*len(segment_id)

        input_id = tokenizer.convert_tokens_to_ids(token)

        input_ids.append(input_id)
        segment_ids.append(segment_id)
        attention_masks.append(attention_mask)

    input_ids = np.array(pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post"))  
    segment_ids = np.array(pad_sequences(segment_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post"))
    attention_masks = np.array(pad_sequences(attention_masks, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")) 

    return input_ids, segment_ids, attention_masks

def get_unique_index(url):
    if url.endswith('.mp3'):
        return url.split('/')[-1].split('.')[0]
    r = requests.get(url)
    
    soup = BeautifulSoup(r.content,'html.parser')
    links = soup.findAll('a')
    mp3_links = [link['href'] for link in links if link['href'].endswith('mp3')]
    link=mp3_links[0]
    return link.split('/')[-1].split('.')[0]

def split_rule(length):
    
    if length == 1:
        return 0
    if length <= 4:
        return 1
    if length <= 6:
        return 2
    if length <= 8:
        return 3
    if length <= 12:
        return 4
    
    return length//2

if __name__ == '__main__':

    train_q1 = []
    train_q2 = []
    train_labels = []
    
    for file_index, excel_file in enumerate(config.EXCEL_FILE_PATH):
        df = pd.read_excel(excel_file, sheet_name=0)
        
        for column_entity, column_name in COLUMN_NAMES.items():
            if column_name not in df.columns:
                raise Exception(f'No Column present with the name {column_name} for {column_entity}')
        
        df = df.dropna(subset=[COLUMN_NAMES['RELEVANT_TOPIC']]).fillna('').reset_index()
        df = df[[column_name for _, column_name in COLUMN_NAMES.items()]]
        
        for index,row in df.iterrows():
            if ' '.join(row[COLUMN_NAMES['BROAD_THEME']].split()) == '':
                row[COLUMN_NAMES['BROAD_THEME']] = df.loc[index-1, COLUMN_NAMES['BROAD_THEME']]
                row[COLUMN_NAMES['DETAIL']] = df.loc[index-1, COLUMN_NAMES['DETAIL']]
                row[COLUMN_NAMES['ANSWER_URL']] = df.loc[index-1, COLUMN_NAMES['ANSWER_URL']]
                row[COLUMN_NAMES['A_TRANSCRIPTION']] = df.loc[index-1, COLUMN_NAMES['A_TRANSCRIPTION']]
                row[COLUMN_NAMES['RELEVANT_POINT']] = df.loc[index-1, COLUMN_NAMES['RELEVANT_POINT']]
            if row[COLUMN_NAMES['STT']] == '':
                row[COLUMN_NAMES['STT']] = row[COLUMN_NAMES['Q_TRANSCRIPTION']]
            for column in list(df.columns)[1:]:
                df.loc[index, column] = ' '.join(df.loc[index, column].split())
        
        print(len(df))
        df = df.replace('', np.nan).dropna().fillna('').reset_index(drop = True)
        print(len(df))
        df['Answer Index'] = df[COLUMN_NAMES['ANSWER_URL']].transform(lambda x :get_unique_index(x))
        df['Question Index'] = df[COLUMN_NAMES['QUESTION_URL']].transform(lambda x :get_unique_index(x))
        
        train_indices = []
        test_indices = []
        
        unique_answer_groups = df.groupby(['Answer Index'])
        num_unique_answer_questions = 0
        num_broad_themes = len(list(df[COLUMN_NAMES['BROAD_THEME']].unique()))
        print("Theme Names: ", list(df[COLUMN_NAMES['BROAD_THEME']].unique()))
        for answer_index in df['Answer Index'].unique():
            group = unique_answer_groups.get_group(answer_index)
            group_length = len(group)
            test_length = split_rule(group_length)
            
            if test_length == 0:
                num_unique_answer_questions += 1
            
            test_indices_for_group = random.sample(list(group.index), k=test_length)
            train_indices_for_group = [x for x in list(group.index) if x not in test_indices_for_group]
            
            train_indices.extend(train_indices_for_group)
            test_indices.extend(test_indices_for_group)    
        
        
        df_train = df.iloc[train_indices]
        unique_answer_groups = df_train.groupby(['Answer Index'])
        
        for answer_index in df_train['Answer Index'].unique():
            group = unique_answer_groups.get_group(answer_index)
            group_length = len(group)
            train_indices_for_group = list(group.index)
            other_indices = [x for x in df_train.index if x not in group]
            
            for i in range(len(train_indices_for_group)):
                for j in range(i+1 , len(train_indices_for_group)):
                    train_q1.append(df_train.loc[train_indices_for_group[i], TRAIN_COLUMN])
                    train_q2.append(df_train.loc[train_indices_for_group[j], TRAIN_COLUMN])
                    train_labels.append(1)
                    
            num_negative_samples = int(1.75 * group_length)
            
            for i in range(num_negative_samples):
                
                random_index_1 = random.choice(train_indices_for_group)
                random_index_2 = random.choice(other_indices)
                train_q1.append(df_train.loc[random_index_1, TRAIN_COLUMN])
                train_q2.append(df_train.loc[random_index_2, TRAIN_COLUMN])
                train_labels.append(0)
                
        test_q1 = []
        test_q2 = []
        
        for test_index in test_indices:
            for train_index in train_indices:
                test_q1.append(df.loc[test_index, TEST_COLUMN])
                test_q2.append(df.loc[train_index, TRAIN_COLUMN])
                
        if file_index == 0:
            pd.DataFrame({'q1' : test_q1, 'q2' : test_q2}).to_csv('data/' + DATA + '_test.csv', index = False)
            df.to_csv('data/' + DATA + '_all_data.csv', index = False)
            print(f'Number of Broad Themes in Data: {num_broad_themes}')
            print(f'Number of Questions in Data: {len(df)}')
            print(f'Number of Questions with Unique Answers: {num_unique_answer_questions}')
            print(f'Number of Questions in Test Data: {len(test_indices)}')
            print(f'Length of Test Data is {len(test_q1)}')
        
    pd.DataFrame({'q1' : train_q1, 'q2' : train_q2, 'label' : train_labels}).to_csv('data/' + DATA + '_train.csv', index = False)
    
    
    print(f'Number of Questions in Train Data (includes questions from all given excel sheets): {len(train_indices)}')
    print(f'Number of Question-Question pairs in Train Data (includes questions from all given excel sheets) is {len(train_q1)}')
    