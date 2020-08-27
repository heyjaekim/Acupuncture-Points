import datetime
def insert_value(id_, Searching_keyword):

    conn = sqlite3.connect('Collect_Data.db')
    cur = conn.cursor()
    cur.execute("SELECT Patient_id FROM Paitient WHERE ID == '" + id_ + "'")
    Patient_id = cur.fetchall()[0][0]
    cur.executescript('''
    INSERT INTO Searching_log (timestamp, Patient_id, Searching_keyword) 
    VALUES('''+ "'" + str(datetime.datetime.now()) + "','" + str(Patient_id) + "', '"+ str(Searching_keyword) + "'"''');
    ''')
    conn.commit()

insert_value('sungj8770', '배가 아파요')
