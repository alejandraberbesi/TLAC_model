import os
import google.auth
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/mnt/c/Users/aleja/Documents/llaves/tlac-vision/tlac-vision-c0786b53c370.json"

credentials, your_project_id = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

bqclient = bigquery.Client(
    credentials=credentials,
    project=your_project_id,
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
print(dataframe.head())

#print(type(dataframe))