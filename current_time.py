import datetime
import pytz

def get_time():
    current_time = datetime.datetime.now()
    date = current_time.date()
    time = current_time.strftime("%H:%M:%S")
    
    return str(date) + " " + str(time)
