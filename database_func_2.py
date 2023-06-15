import psycopg2
import pandas as pd

conn_data = psycopg2.connect(host="localhost",database="UserElectricityData", user="postgres",password="Oppark12#g")

def get_csv(userid):

    cur = conn_data.cursor()
    cur.execute("SELECT * FROM userdata where userid = (%s)", (userid,))
    row = cur.fetchone()

    df = pd.DataFrame([row], columns=[i[0] for i in cur.description])

    num_cols = df.shape[1]
    start_date = pd.to_datetime('2014-01-01 00:00:00')

    for i in range(num_cols):
        orignal_col = 'column_{}'.format(i)
        df.rename(columns = {orignal_col:start_date}, inplace = True)
        start_date = start_date + pd.Timedelta(days=1)
    
    df.drop(['userid'],inplace=True, axis=1)
    df = df.T

    df_modified = create_columns(df)

    return df_modified

def create_columns(df):
    df['date'] = pd.to_datetime(df.index)

    df['month'] = df['date'].dt.month.astype(int)
    df['day_of_month'] = df['date'].dt.day.astype(int)

    # day_of_week=0 corresponds to Monday
    df['day_of_week'] = df['date'].dt.dayofweek.astype(int)
    df['hour_of_day'] = df['date'].dt.hour.astype(int)

    one_user_data = ['date', 'day_of_week', 'hour_of_day', 0]
    df = df[one_user_data]

    df['Energy_consumption']=df[0]
    df.drop(0,axis=1,inplace=True)

    return df

