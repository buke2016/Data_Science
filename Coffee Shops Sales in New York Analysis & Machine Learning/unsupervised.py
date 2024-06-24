
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf, suppress=True)



def update_matrix(matrix, order):
    # remove dupes from order (So we don't count purchases of the same product twice as a pair)
    temp_set = set(order)
    order = list(temp_set)

    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            matrix[order[i]][order[j]] += 1
            matrix[order[j]][order[i]] += 1
    return matrix


def main():
        
    data = pd.read_csv('data/PairIDdata.csv')

    # Construct A matrix that has a row and column for each product id
    # How large is the matrix?
    largest_index = data['product_id'].unique().max()
    ref_matrix = np.zeros((largest_index + 1, largest_index + 1))
    id_to_name = data.set_index('product_id')['product_detail'].to_dict()


    curr_pair_id = -1
    curr_order = []


    for index, row in data.iterrows():
        pair_id = row['pair_id']

        # Skip if the row is not a pair
        if pair_id > 0:
            
            # New pair
            if pair_id != curr_pair_id:
                # We also know we recently finished a pair
                # so we will do all the adding to the matrix here
                ref_matrix = update_matrix(ref_matrix, curr_order)

                # Reset the variables for this new order
                curr_pair_id = pair_id
                curr_order = []
                curr_order.append(row['product_id'])

            # Second or later item in a pair
            if pair_id == curr_pair_id:
                # add the product to the order
                curr_order.append(row['product_id'])

    # update for last pair in the dataset
    ref_matrix = update_matrix(ref_matrix, curr_order)

    # Okay so now we have a matrix. We want to look at an item and see what other items were most commonly paired with it.
    # One question arises: what if there is never an item that was never paired with another item? What do we recommend?
    
    print(ref_matrix[50].argsort()[-5:][::-1])
    print(ref_matrix[50])
    print(id_to_name[50])

    # THis is good enough for now. Adding a subsetting feature for isolating bakery vs. drinks would be a good next step. If you get a scone, offer a drink. If you get a drink, offer a scone or a cookie.

if __name__ == '__main__':
    main()