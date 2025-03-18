#%%

import pandas as pd

df = pd.read_csv("data/train.csv")

print(df.head())

print("Avarage age is", df["Age"].mean())
df["colonne_sans_sens"] = df["Age"] / df["Pclass"]

#%%

arr=df.to_numpy
print(arr)

#%%
import pandas as pd

df = pd.read_csv("data/train.csv")

print(df.head())

from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="Profiling Report")
profile.to_file("your_report.html")

# %%
