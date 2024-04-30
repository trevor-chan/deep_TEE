import pandas as pd

#%% load data
def load_excel (fn):
  return pd.read_excel(fn, sheet_name=None, engine='openpyxl')

#%% aggregate data into one DF
def aggregate_data (data):
  agg_data = pd.DataFrame()
  subjects = list(data.keys())[1:]
  for subject in subjects:
    subject_data = data[subject]
    subject_data['Subject'] = f'{subject}'
    agg_data = pd.concat([agg_data, subject_data], ignore_index=True)
  return agg_data
