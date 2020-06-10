from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1
from google.oauth2 import service_account
from langdetect import detect
from fastai.text import *
import pathlib
import multifit
from multifit.datasets import ULMFiTDataset, Dataset

# to import data from GCP:
credentials = service_account.Credentials.from_service_account_file(
    '/mnt/c/Users/aleja/Documents/llaves/tlac-vision/tlac-vision-c0786b53c370.json')

project_id = "tlac-vision"

bqclient = bigquery.Client(
    credentials=credentials,
    project=project_id,
)

bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient(
    credentials=credentials
)

query_string = """SELECT * FROM `tlac-vision.book_backend.train_categories`"""

df = (
    bqclient.query(query_string)
    .result()
    .to_dataframe(bqstorage_client=bqstorageclient)
)

# sorting dataframe by book category, from A to Z
df = df.sort_values(['category'], ascending=True)

# dropping duplicate rows
df = df.drop_duplicates(['title']).reset_index(drop=True)

# deleting results that are in other languages different to spanish
df['language'] = ""
for index, row in df.iterrows():
    row['language'] = detect(row['description'])
df = df[df.language == 'es']

cat_count = df.iloc[:, 0:2].groupby(
    'category').count().rename(columns={'title': 'count'})  # total of registers per category

# 75% of data is for training
cat_count['training'] = round(0.75 * cat_count['count'], 0)
# organizing indexes to split data
cat_count['acum'] = cat_count['count'].cumsum()
cat_count['init_idx'] = cat_count['acum'] - cat_count['count']
cat_count['train_idx'] = cat_count['init_idx'] + cat_count['training']

# creating training set with 75% of data per category
train_set = pd.DataFrame(data=None, columns=df.columns)
for i in range(len(cat_count)):
    i_idx = int(cat_count.iloc[i, 3])
    f_idx = int(cat_count.iloc[i, 4])
    train_set = train_set.append(df.iloc[i_idx:f_idx, :])

train_set_f = train_set.loc[:, ['category', 'description']]

# creating validation set with 25% of data per category
val_set = pd.DataFrame(data=None, columns=df.columns)
for i in range(len(cat_count)):
    i_idx = int(cat_count.iloc[i, 4])
    f_idx = int(cat_count.iloc[i, 2])
    val_set = val_set.append(df.iloc[i_idx:f_idx, :])

val_set_f = val_set.loc[:, ['category', 'description']]

path = pathlib.Path().absolute()

# train_set.to_csv('train.csv')  changes to structure?
# val_set.to_csv('dev.csv')

lang = 'es'
exp = multifit.from_pretrained(f'{lang}_multifit_paper_version')
exp.replace_(name='multifit_paper_version20')
exp.arch

mldoc_dataset = exp.arch.dataset(path, exp.pretrain_lm.tokenizer)

mldoc_dataset.databunch_from_df(
    bunch_class=TextLMDataBunch, train_df=train_set_f, valid_df=val_set_f)

# mldoc_dataset.show_batch()

# list(train_set.columns.values)[1:]

# tokenization------------
#tok = Tokenizer(tok_func=SpacyTokenizer,lang='es')
# data = TextLMDataBunch.from_df(
#    path=path, train_df=train_set, valid_df=val_set, tokenizer=tok, text_cols='description', label_cols='category')
# data.show_batch()
