import pandas as pd
df = pd.read_csv(r"G:\OPT\JSCS-WFC20261005\data\tooth24_clean.csv")
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print(df.head(3))
