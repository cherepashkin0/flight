time_cols = ["CRSDepTime", "DepTime", "WheelsOff", "WheelsOn", "CRSArrTime", "ArrTime"]

for col in time_cols:
    print(f"""{col},
  EXTRACT(HOUR FROM PARSE_TIME('%H:%M', FORMAT('%02d:%02d', 
    CAST(FLOOR(CAST({col} AS INT64) / 100) AS INT64), 
    CAST(MOD(CAST({col} AS INT64), 100) AS INT64)
  ))) * 60 +
  EXTRACT(MINUTE FROM PARSE_TIME('%H:%M', FORMAT('%02d:%02d', 
    CAST(FLOOR(CAST({col} AS INT64) / 100) AS INT64), 
    CAST(MOD(CAST({col} AS INT64), 100) AS INT64)
  ))) AS {col}_minutes,\n""")
