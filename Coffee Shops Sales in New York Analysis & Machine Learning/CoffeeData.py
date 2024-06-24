import pandas as pd

data = pd.read_csv('data/dataWithDupeQty.csv')
data['transaction_qty'] = 1

data['pair_id'] = -1
purchases = pd.DataFrame()
inside_purch = False
current_purch = []

pair_id = 0
prev_transaction_time = None
prev_pair_id = None

for index, row in data.iterrows():
    if prev_transaction_time is not None and row['transaction_time'] == prev_transaction_time:
        if prev_pair_id is None:
            pair_id += 1
            prev_pair_id = pair_id
        data.at[index - 1, 'pair_id'] = prev_pair_id
        data.at[index, 'pair_id'] = prev_pair_id
    else:
        prev_pair_id = None
    prev_transaction_time = row['transaction_time']

data.to_csv('data/PairIDdata.csv', index=False)