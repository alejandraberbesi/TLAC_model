import os
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1
from google.oauth2 import service_account

credentials= service_account.Credentials.from_service_account_file(
    '/mnt/c/Users/aleja/Documents/llaves/tlac-vision/tlac-vision-c0786b53c370.json')

project_id="tlac-vision"

bqclient = bigquery.Client(
    credentials=credentials,
    project=project_id,
)

bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient(
    credentials=credentials
)

query_string = """
SELECT *
FROM `tlac-vision.book_backend.train_categories`
"""

dataframe = (
    bqclient.query(query_string)
    .result()
    .to_dataframe(bqstorage_client=bqstorageclient)
)

print(dataframe.groupby('category').count())
