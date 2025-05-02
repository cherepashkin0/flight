import pandas as pd
import clickhouse_connect
import os

def map_dtype_to_clickhouse(dtype: str, nullable: bool) -> str:
    """ Преобразует pandas dtype в ClickHouse тип, включая Nullable """
    if dtype.startswith('int'):
        ch_type = 'Int64'
    elif dtype.startswith('float'):
        ch_type = 'Float64'
    elif dtype == 'bool':
        ch_type = 'UInt8'
    elif dtype.startswith('datetime'):
        ch_type = 'DateTime'
    elif dtype == 'object' or dtype == 'string':
        ch_type = 'String'
    elif dtype == 'category':
        ch_type = 'String'
    else:
        ch_type = 'String'

    return f'Nullable({ch_type})' if nullable else ch_type

# Название новой базы данных
db_name = 'flights_data'

# Подключение к ClickHouse
client = clickhouse_connect.get_client(
    host=os.getenv('CLICKHOUSE_HOST'),
    port=os.getenv('CLICKHOUSE_PORT'),
    username=os.getenv('CLICKHOUSE_USER'),
    password=os.getenv('CLICKHOUSE_PASSWORD')
)

# Создание базы данных, если не существует
client.command(f'CREATE DATABASE IF NOT EXISTS {db_name}')

# Загрузка и обработка данных по годам
for year in range(2018, 2023):
    parquet_path = f'flight_delay/Combined_Flights_{year}.parquet'
    print(f'📦 Загрузка файла: {parquet_path}')
    
    df = pd.read_parquet(parquet_path)
    table_name = f'{db_name}.flight_{year}'

    # Определяем ClickHouse-тип для каждого столбца с Nullable
    column_definitions = []
    for col in df.columns:
        sanitized_col = col.replace(' ', '_')
        dtype = str(df[col].dtype)
        nullable = df[col].isnull().any()
        ch_type = map_dtype_to_clickhouse(dtype, nullable)
        column_definitions.append(f"`{sanitized_col}` {ch_type}")
        df.rename(columns={col: sanitized_col}, inplace=True)

    # Удаление таблицы, если она существует
    client.command(f'DROP TABLE IF EXISTS {table_name}')

    # Создание таблицы
    create_query = f'''
    CREATE TABLE {table_name} (
        {', '.join(column_definitions)}
    ) ENGINE = MergeTree()
    ORDER BY tuple()
    '''
    client.command(create_query)

    # Заменяем NaN на None, чтобы ClickHouse правильно обработал Nullable
    df = df.where(pd.notnull(df), None)

    # Загрузка данных
    client.insert_df(table_name, df)

    print(f'✅ Год {year}: загружено {len(df)} строк в таблицу {table_name}')
