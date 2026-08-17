DATA_START, DATA_END = "2006Q1", "2011Q4"
PRE_END = "2008Q1"
TRANSITION = ("2008Q2", "2009Q4")
POST_START = "2010Q2"
ES_REFERENCE = "2008Q1"
BASE_YEAR = 2006

TRIM_MODE = "fixed"
FARE_MIN, FARE_MAX = 25, 2500
FARE_PCT_MIN, FARE_PCT_MAX = 0.01, 0.99
MIN_PAX_ROUTE_QTR = 20
OVERLAP_SHARE = 0.05
MIN_PRE_QUARTERS = 6
ID_MULTIPLIER = 100000
MAX_COUPONS = 5

TREAT_PAIR = ("DL", "NW")
CONFOUND_PAIRS = [("UA", "CO"), ("WN", "FL")]

from vocab import AIRPORTS, CARRIERS
from pandas import CategoricalDtype
AIRPORT_DTYPE = CategoricalDtype(categories=AIRPORTS)
CARRIER_DTYPE = CategoricalDtype(categories=CARRIERS)

DATA_COLS = ['Year', 'Quarter', 'Origin', 'Dest', 'TkCarrier', 
             'Passengers', 'MktFare', 'MktDistance', 'MktCoupons', 
             'BulkFare', 'TkCarrierChange', 'MktGeoType','OriginCityMarketID', 
             'DestCityMarketID']

DATA_TYPES = {
    'Origin': AIRPORT_DTYPE, 'Dest': AIRPORT_DTYPE, 'TkCarrier': CARRIER_DTYPE,
    'MktFare': 'float32', 'MktDistance': "float32",
}

INT_COLS = {
    "MktCoupons": "int8", "BulkFare": "int8",
    "TkCarrierChange": "int8", "MktGeoType": "int8",
    "Year": "int16", "Quarter": "int8",
    "Passengers": "int32",
    "OriginCityMarketID": "int32", "DestCityMarketID": "int32",
}

def q_to_t(yearq):
    year = int(yearq[:4])
    quarter = int(yearq[-1])
    return (year - BASE_YEAR) * 4 + quarter