import sqlite3
import pandas as pd

db_name = 'f1_database.db'
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

try:
    cursor.execute("""
        DROP TABLE IF EXISTS final_table;
    """)
    print("Table 'final_table' successfully dropped")

    cursor.execute("""               
        CREATE TABLE final_table AS 
        SELECT res.raceId, 
               rac.name AS circuit, 
               rac.round, 
               rac.year, 
               rac.date, 
               c.name AS constructor, 
               (d.forename || ' ' || d.surname) AS driver, 
               res.grid AS starting_position, 
               res.positionOrder AS finishing_position, 
               CAST(res.points AS INTEGER) AS points,
               rac.url
        FROM results res 
        LEFT JOIN races rac ON res.raceId = rac.raceId
        LEFT JOIN constructors c ON res.constructorId = c.constructorId
        LEFT JOIN drivers d ON res.driverId = d.driverId
        WHERE rac.year >= 2020
         and rac.year <= 2024;
    """)
    print("Table 'final_table' created successfully.")

    # Check for NULL values in columns
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN raceId IS NULL THEN 1 ELSE 0 END) AS raceId_nulls,
            SUM(CASE WHEN circuit IS NULL THEN 1 ELSE 0 END) AS race_name_nulls,
            SUM(CASE WHEN round IS NULL THEN 1 ELSE 0 END) AS round_nulls,
            SUM(CASE WHEN year IS NULL THEN 1 ELSE 0 END) AS year_nulls,
            SUM(CASE WHEN date IS NULL THEN 1 ELSE 0 END) AS date_nulls,
            SUM(CASE WHEN driver IS NULL THEN 1 ELSE 0 END) AS driver_nulls,
            SUM(CASE WHEN constructor IS NULL THEN 1 ELSE 0 END) AS constructor_nulls,
            SUM(CASE WHEN starting_position IS NULL THEN 1 ELSE 0 END) AS starting_position_nulls,
            SUM(CASE WHEN finishing_position IS NULL THEN 1 ELSE 0 END) AS finishing_position_nulls,
            SUM(CASE WHEN points IS NULL THEN 1 ELSE 0 END) AS points_nulls
        FROM final_table;
    """)

    null_counts = cursor.fetchone()
    print("NULL value counts by column:", null_counts)

except sqlite3.Error as e:
    print(f"An error occurred: {e}")

conn.commit()
conn.close()

conn = sqlite3.connect(db_name)
df_final = pd.read_sql_query("SELECT * FROM final_table", conn)
print(df_final)

csv_path = "final_table.csv"
df_final.to_csv(csv_path, index=False)
print(f"Data saved to CSV at: {csv_path}")

conn.close()
