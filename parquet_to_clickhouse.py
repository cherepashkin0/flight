# parquet_to_clickhouse.py
import pandas as pd
import clickhouse_connect
import os

def map_dtype_to_clickhouse(dtype: str, nullable: bool, min_val=None, max_val=None) -> str:
    """ Преобразует pandas dtype в ClickHouse тип, включая Nullable, с учетом диапазона значений """
    ch_type = "String"  # fallback по умолчанию

    if dtype.startswith('int'):
        if min_val is not None and max_val is not None:
            if 0 <= min_val and max_val <= 255:
                ch_type = 'UInt8'
            elif -128 <= min_val <= max_val <= 127:
                ch_type = 'Int8'
            elif -32768 <= min_val <= max_val <= 32767:
                ch_type = 'Int16'
            elif -2**31 <= min_val <= max_val <= 2**31 - 1:
                ch_type = 'Int32'
            else:
                ch_type = 'Int64'
        else:
            ch_type = 'Int64'

    elif dtype.startswith('float'):
        ch_type = 'Float32' if max_val is not None and abs(max_val) < 1e38 else 'Float64'

    elif dtype == 'bool':
        ch_type = 'UInt8'

    elif dtype.startswith('datetime'):
        ch_type = 'DateTime'

    elif dtype == 'object' or dtype == 'string' or dtype == 'category':
        ch_type = 'String'

    return f'Nullable({ch_type})' if nullable else ch_type


# Название базы данных
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

    # Определяем ClickHouse-тип для каждого столбца
    column_definitions = []
    for col in df.columns:
        sanitized_col = col.replace(' ', '_')
        df.rename(columns={col: sanitized_col}, inplace=True)

        series = df[sanitized_col]
        dtype = str(series.dtype)
        nullable = series.isnull().any()

        min_val = series.min(skipna=True) if pd.api.types.is_numeric_dtype(series) else None
        max_val = series.max(skipna=True) if pd.api.types.is_numeric_dtype(series) else None

        ch_type = map_dtype_to_clickhouse(dtype, nullable, min_val, max_val)
        column_definitions.append(f"`{sanitized_col}` {ch_type}")

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
