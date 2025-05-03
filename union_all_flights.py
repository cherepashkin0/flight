import os
import clickhouse_connect

DB_NAME = 'flights_data'
TABLE_YEARS = range(2018, 2023)
MERGED_TABLE = f'{DB_NAME}.flight_all'

# Подключение к ClickHouse
client = clickhouse_connect.get_client(
    host=os.getenv('CLICKHOUSE_HOST'),
    port=os.getenv('CLICKHOUSE_PORT'),
    username=os.getenv('CLICKHOUSE_USER'),
    password=os.getenv('CLICKHOUSE_PASSWORD')
)

# Удаление старой таблицы, если была
client.command(f'DROP TABLE IF EXISTS {MERGED_TABLE}')

# Получение колонок исходной таблицы
sample_table = f'{DB_NAME}.flight_2018'
columns_info = client.query(f'DESCRIBE TABLE {sample_table}')
columns = [row[0] for row in columns_info.result_rows]

# Добавляем колонку year
columns.append('year')
column_definitions = []

for row in columns_info.result_rows:
    col_name, col_type = row[0], row[1]
    column_definitions.append(f"`{col_name}` {col_type}")
# Добавляем явную колонку year
column_definitions.append("`year` UInt16")

# Создание итоговой таблицы
create_query = f'''
CREATE TABLE {MERGED_TABLE} (
    {', '.join(column_definitions)}
) ENGINE = MergeTree()
ORDER BY tuple()
'''
client.command(create_query)
print(f"✅ Создана таблица: {MERGED_TABLE}")

# Вставка данных из всех таблиц
column_list = ', '.join(f"`{col}`" for col in columns if col != 'year')  # без year

for year in TABLE_YEARS:
    source_table = f'{DB_NAME}.flight_{year}'
    insert_query = f'''
        INSERT INTO {MERGED_TABLE}
        SELECT {column_list}, {year} AS year FROM {source_table}
    '''
    client.command(insert_query)
    print(f"📥 Добавлены данные из {source_table}")

print(f"🎉 Таблица '{MERGED_TABLE}' со всеми годами успешно создана.")
