from pydantic import BaseModel, Field
from typing import Optional

class FlightFeatures(BaseModel):
    OriginAirportID: int
    DestCityMarketID: int
    DestStateFips: int
    DOT_ID_Marketing_Airline: int
    OriginAirportSeqID: int
    OriginWac: int
    DestAirportID: int
    CRSElapsedTime: float
    DOT_ID_Operating_Airline: int
    Distance: float
    DayofMonth: int
    OriginCityMarketID: int
    DestAirportSeqID: int
    OriginStateFips: int
    Flight_Number_Operating_Airline: int
    Flight_Number_Marketing_Airline: int