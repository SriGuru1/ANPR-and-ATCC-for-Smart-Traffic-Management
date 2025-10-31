# dbconnection.py
import pymysql

def get_connection():
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='BABLUinfosys@12345',
        database='mydb'
    )
    return conn