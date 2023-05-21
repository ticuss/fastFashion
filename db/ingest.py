import sqlite3
import csv

conn = sqlite3.connect("db/raw_data.sqlite3")

c = conn.cursor()

with open("data.csv", "r") as f:
    reader = csv.reader(f)

    next(reader)

    for row in reader:
        c.execute(
            """
            INSERT INTO model_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            row,
        )

conn.commit()

conn.close()
