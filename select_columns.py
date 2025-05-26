import os
from google.cloud import bigquery

project_id = os.environ.get('GCP_PROJECT_ID')


def load_all_columns():
    client = bigquery.Client(project=project_id)
    query = f"""
                SELECT column_name
                FROM `{project_id}.flights.INFORMATION_SCHEMA.COLUMNS`
                WHERE table_name = 'flights_all';
            """
    df = client.query(query).to_dataframe(create_bqstorage_client=True)
    lst = [name[0] for name in df.values]
    return lst

def load_drop(file_pathes):
    lst = []
    for file_name in file_pathes:
        with open(file_name, 'r') as file:
            cur_lst = file.readlines()
        cur_lst = [word.strip('\n') for word in cur_lst]
        lst = cur_lst + lst
    lst = list(set(lst))
    lst.sort()
    return lst

def main():   
    drop_lst = load_drop(['columns_to_drop_high_correlation.txt',
                   'columns_to_drop_leakage.txt',
                   'columns_to_drop_low_importance.txt',
                   'columns_to_drop_id.txt'])
    all_column_lst = load_all_columns()
    print(list(set(all_column_lst) - set(drop_lst)))

if __name__ == "__main__":
    main()
