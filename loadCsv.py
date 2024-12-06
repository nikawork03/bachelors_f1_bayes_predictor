import glob
import pandas as pd
import kagglehub

# import f1 data from kaggle datasets

path = kagglehub.dataset_download("rohanrao/formula-1-world-championship-1950-2020")
all_files = glob.glob(path + "/*.csv")
df_list = []
for filename in all_files:
    df = pd.read_csv(filename)
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)

# test
print(df_list[0].head())

# check for null valuesa
for i in df_list:
    print(i.isnull().sum())


# Filter data for years 2010-2024
df = df[df['year'] >= 2010]

# Check for relevant columns in the dataset
print(df.columns)

distinct_years = df['year'].unique()
print("Distinct years in the dataset (2010-2024):", sorted(distinct_years))


# Save the DataFrame to a CSV file
df.to_csv("f1_data.csv", index=False)
