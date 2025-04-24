import datetime
import pprint
import numpy as np

''''

Aux Functions

'''


def datetime_to_yf_format(date: datetime.datetime) -> str:
    """Converts date in datetime format to string in YYYY-MM-DD format"""
    return f"{date.year:04d}-{date.month:02d}-{date.day:02d}"

def print_pretty_dic(dic_to_print):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(dic_to_print)



    