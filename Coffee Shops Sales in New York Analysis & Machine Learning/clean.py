import pandas as pd

data = pd.read_csv('data/subdata.csv')


for index, row in data.iterrows():
    # We want to add a duplicate row for each purchase that has more than one product
    quantity = row['transaction_qty']
    if quantity > 1:
        row['transaction_qty'] = 1
        for i in range(quantity - 1):
            data = data._append(row)

data.sort_index(inplace=True)
data.to_csv('data/dataWithDupeQty.csv', index=False)