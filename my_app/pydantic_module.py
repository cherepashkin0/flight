from pydantic import BaseModel, Field
from typing import Optional

class FlightFeatures(BaseModel):
    OriginAirportID: int
    ArrDel15: float
    IATA_Code_Operating_Airline: str
    Marketing_Airline_Network: str
    DestState: str
    OriginCityName: str
    DepDel15: float
    DestStateName: str
    DayOfWeek: int
    Month: int
    DestCityMarketID: int
    DestStateFips: int
    DOT_ID_Marketing_Airline: int
    Operated_or_Branded_Code_Share_Partners: str
    OriginAirportSeqID: int
    Quarter: int
    Dest: str
    Operating_Airline: str
    Cancelled: bool
    IATA_Code_Marketing_Airline: str
    OriginWac: int
    DestAirportID: int
    OriginState: str
    CRSElapsedTime: float
    DOT_ID_Operating_Airline: int
    DistanceGroup: int
    Distance: float
    DayofMonth: int
    OriginCityMarketID: int
    DestAirportSeqID: int
    Year: int
    OriginStateName: str
    OriginStateFips: int
    Airline: str