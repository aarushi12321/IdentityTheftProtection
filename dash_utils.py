import psycopg2
from datetime import datetime
import pandas as pd
import numpy as np

def get_dataframe(query):

    conn = psycopg2.connect(
    host="localhost",
    database="UserElectricityData",
    user="postgres",
    password="Oppark12#g"
)

    cur = conn.cursor()

    cur.execute(query)
    rows = cur.fetchall()
    col_names = list(pd.read_csv('/Users/aarushisethi/Desktop/PredOnly/column_list.csv')['column_list'])

    col_list = ['userid']
    for col in col_names:
        if col not in ['userid', 'FLAG']:
            col_list.append(datetime.strptime(col, '%Y-%m-%d %H:%M:%S'))
    
    col_list.append('FLAG')        
    df = pd.DataFrame(rows, columns=col_list)

    cur.close()
    conn.close()

    return df

def get_users(df):
    users = df.shape[0]

    return str(users)

def total_consumption(df):
    
    sum = 0
    
    for col in df.columns:
        if col not in ['userid', 'FLAG']:
            sum += float(df.loc[:,col].values)

    return np.round(sum,5) 


    