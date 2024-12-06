import os
import sqlite3
import kagglehub
import pandas as pd

csv_folder = kagglehub.dataset_download("rohanrao/formula-1-world-championship-1950-2020")

db_name = 'f1_database.db'
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

for file in os.listdir(csv_folder):
    if file.endswith(".csv"):
        table_name = os.path.splitext(file)[0]
        file_path = os.path.join(csv_folder, file)

    df = pd.read_csv(file_path)

    df.to_sql(table_name,conn,if_exists='replace', index=False)
    print(f"Database '{table_name}' created from '{file}'.")

print(f"Database '{db_name}' created succesfully")

