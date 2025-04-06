import re
import pandas as pd
import ast

def clean_text(text):
    cleaned_text = re.sub(r'[\xa0\xad]+', ' ', text)
    cleaned_text = re.sub(r'\.{2,}', '.', cleaned_text)
    cleaned_text = re.sub(r'\…+', '', cleaned_text)
    cleaned_text = re.sub(r'[@#%^&*]+', '', cleaned_text)
    cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)
    cleaned_text = re.sub(r'-{2,}', '', cleaned_text)
    cleaned_text = re.sub(r'_{4,}', '', cleaned_text)
    cleaned_text = re.sub(r':\.*', '', cleaned_text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text


corpus = pd.read_csv('corpus.csv')
texts = corpus['text'].values.tolist()
texts = [clean_text(text.lower()) for text in texts]
texts[131503] = 'các cơ quan thuộc, trực thuộc các cơ quan đảng 000.00.00.g12 là mã của bộ tài chính; 000.00.18.g12 là mã của tổng cục thuế; ví dụ đối với các cơ quan, đơn vị cấp 3, cấp 4 thuộc, trực thuộc cơ quan trung ương ở các tỉnh, thành phố trực thuộc trung ương được đánh mã trùng với mã tỉnh được ban hành theo quyết định số 124/2004/qđ-ttg ngày 08 tháng 7 năm 2004 của thủ tướng chính phủ như sau  phụ lục d (tham khảo) đơn vị cấp 2, cấp 3 và cấp 4 d.1. đơn vị cấp 2 d.1.1. các cơ quan thuộc, trực thuộc các cơ quan đảng'
cleaned_corpus = pd.DataFrame({
    'cid': corpus['cid'].values,
    'text': texts
    }
)
cleaned_corpus.to_csv('cleaned_corpus.csv', index=False)
cid2corpus = {}
for id, row in cleaned_corpus.iterrows():
    cid2corpus[int(row['cid'])] = row['text']

hard_neg = pd.read_csv('hard_neg_pos_aware.csv')
from tqdm import tqdm
df_train = {'text':[], 'positive' : [], 'negative' : []}
for idx, data in tqdm(hard_neg.iterrows(), total=hard_neg.shape[0], desc="Processing rows"):
    query = data['text']
    poss = ast.literal_eval(data['pos'])
    negs = ast.literal_eval(data['neg'])
    if len(poss) == 0 or len(negs) == 0:
        continue
    else:
        for neg_id in negs:
            neg = cid2corpus[int(neg_id)]
            for pos_id in poss:
                pos = cid2corpus[int(pos_id)]
                df_train['text'].append("query: " + clean_text(query.lower()))
                df_train['positive'].append("passage: " + pos)
                df_train['negative'].append("passage: " + neg)

df_train = pd.DataFrame(df_train)
df_train.to_csv('full_hard_neg.csv', index=False)



