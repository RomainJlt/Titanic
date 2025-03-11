import pandas as pd
import numpy as np
import matplotlib as mpl


df = pd.read_csv("data/gender_submission.csv")

print(df.head())

from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="Profile Report")
profile.to_file("your_report.html")