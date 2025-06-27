import os
from google.cloud import bigquery
import pyarrow.parquet as pq

data_path = '/home/vsiewolod/datasets/flight_data'
project_id = 'flight-cancellation-prediction'
dataset_id = 'flight_data'

def main():
    client = bigquery.Client(project=project_id)
    dataset_ref = client.dataset(dataset_id)

    for year in range(2018, 2023):
        file_path = os.path.join(data_path, f'Combined_Flights_{year}.parquet')
        table_id = f'combined_flights_{year}'
        print(f"Uploading {file_path} to BigQuery table {table_id}...")

        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            autodetect=True
        )

        with open(file_path, "rb") as f:
            load_job = client.load_table_from_file(
                f,
                dataset_ref.table(table_id),
                job_config=job_config
            )
            load_job.result()
            print(f"Uploaded {year} data successfully to table {table_id}.")

    print("All files uploaded to separate tables.")

if __name__ == "__main__":
    main()
