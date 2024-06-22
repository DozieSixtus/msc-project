import os
import pandas as pd

pos_data = r'.\data\IMDB\train\pos'
neg_data = r'.\data\IMDB\train\neg'
path = r'.\data'

def load_imdb(path, sentiment_category):
    files = os.listdir(path)
    output = pd.DataFrame()

    for file in files:
        
        #file_name = os.path.splitext(file)[0]

        with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
            info = f.read()
            info = ''.join([x for x in info])
            
        #print(info)
                
        df = pd.DataFrame([info], columns=['Reviews'])

        output = pd.concat([output, df], axis=0)

    if sentiment_category=='pos':
        output['Label'] = 'positive'
    elif sentiment_category=='neg':
        output['Label'] = 'negative'
    return output

pos_train = load_imdb(pos_data, 'pos')
neg_train = load_imdb(neg_data, 'neg')

train = pd.concat([pos_train, neg_train], axis=0)

train.to_csv(path + '\IMDB.csv', sep='\t', index=False)