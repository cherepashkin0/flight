import pandas as pd

def main():
    df = pd.read_csv('results/flights_all_analysis.csv')

    role_dict = {
        'tgt': ['Cancelled'],
        'dat': ['FlightDate'],
        'num': [
            'OriginAirportID', 'TaxiIn', 'DepartureDelayGroups', 'AirTime', 'TaxiOut',
            'ActualElapsedTime', 'DayOfWeek', 'Month', 'DestCityMarketID', 'DestStateFips',
            'DepDelayMinutes', 'ArrDelay', 'DOT_ID_Marketing_Airline', 'DivAirportLandings',
            'OriginAirportSeqID', 'Quarter', 'OriginWac', 'DestAirportID', 'CRSElapsedTime',
            '__index_level_0__', 'DOT_ID_Operating_Airline', 'DistanceGroup', 'Distance',
            'DayofMonth', 'DepDelay', 'OriginCityMarketID', 'DestAirportSeqID', 'Year',
            'ArrivalDelayGroups', 'ArrDelayMinutes', 'OriginStateFips',
            'Flight_Number_Operating_Airline', 'Flight_Number_Marketing_Airline', 'Diverted'
        ],
        'hot': [
            'DayOfWeek', 'Month', 'Quarter', 'DistanceGroup', 'Year',
            'DepDel15', 'ArrDel15'
        ],
        'cat': [
            'IATA_Code_Operating_Airline', 'Marketing_Airline_Network', 'DestState',
            'OriginCityName', 'DestStateName', 'Operated_or_Branded_Code_Share_Partners',
            'Dest', 'Operating_Airline', 'IATA_Code_Marketing_Airline', 'OriginState',
            'OriginStateName', 'Airline', 'Tail_Number'
        ]
    }

    skip_reason_phik = {
        'id': ['Flight_Number_Operating_Airline', 'Flight_Number_Marketing_Airline'],
        'high_cardinality': ['Tail_Number'],
        'time_stamp': ['FlightDate']
    }

    data_leakage = {
        'true': ['ActualElapsedTime', 'DepDelayMinutes', 'ArrDelayMinutes', 'TaxiOut', 'TaxiIn', 'Diverted']
    }

    # Reverse mapping to assign roles
    col_to_role = dct_revert(role_dict)
    col_to_skip_reason_phik = dct_revert(skip_reason_phik)
    col_data_leakage = dct_revert(data_leakage)
    df['Role'] = df['Column_Name'].map(col_to_role).fillna('unk')
    df['Skip_reason_phik'] = df['Column_Name'].map(col_to_skip_reason_phik).fillna('unk')
    df['Data_leakage'] = df['Column_Name'].map(col_data_leakage).fillna('unk')
    df.to_csv('results/flights_all_analysis_with_roles.csv', index=False)

def dct_revert(input_dict):
    output_dict = {col: role for role, cols in input_dict.items() for col in cols}
    return output_dict

if __name__ == "__main__":
    main()
