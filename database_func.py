import psycopg2
import csv
from flask import render_template

conn_data = psycopg2.connect(host="localhost",database="UserElectricityData", user="postgres",password="Oppark12#g")

def add_user(data):
    userid = data["userid"]
    data.pop("userid", None)
    data = list(data.values())

    cur = conn_data.cursor()

    flag = 0

    cur.execute("SELECT * FROM userdata where userid = (%s)", (userid,))
    row = cur.fetchone()

    if row:
        flag = 1

    else:
        cur.execute("INSERT INTO userdata (userid) VALUES (%s)", (userid,))
        query = "UPDATE userdata SET "
        for i in range(1034):
            col_name = "column_" + str(i)
            if i == 1033:
                query += "{} = {} WHERE userid = %s".format(col_name, data[i])
            else:
                query += "{} = {},".format(col_name, data[i])
        cur.execute(query, (userid,))

    
    conn_data.commit()
    cur.close()

    return flag

def add_flag(userid, flag):

    cur = conn_data.cursor()
    cur.execute("INSERT INTO userflag (userid, flag_output) VALUES (%s, %s)", (userid, flag))

    conn_data.commit()
    cur.close()

    return

def get_result(userid):
    cur = conn_data.cursor()
    cur.execute("SELECT flag_output FROM userflag WHERE userid = '{}'".format(userid))
    row = cur.fetchone()

    conn_data.commit()
    cur.close()

    return row


