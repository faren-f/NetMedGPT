def prepare_txgnn_data(nodes, train_txgnn, val_txgnn, test_txgnn, type2order):

    train_txgnn = correct_order(train_txgnn, type2order)
    val_txgnn = correct_order(val_txgnn, type2order)
    test_txgnn = correct_order(test_txgnn, type2order)

    ###prepare train data
    nodes['id_type'] = [f"{i}|{j}" for i, j in zip(nodes['node_id'], nodes['node_type'])]
    id_type2index = dict(zip(nodes['id_type'], nodes.index))
    train_txgnn['x_id'] = train_txgnn['x_id'].apply(normalize_id)
    train_txgnn['y_id'] = train_txgnn['y_id'].apply(normalize_id)
    train_txgnn['x_id_type'] = [f"{i}|{j}" for i, j in zip(train_txgnn['x_id'], train_txgnn['x_type'])]
    train_txgnn['y_id_type'] = [f"{i}|{j}" for i, j in zip(train_txgnn['y_id'], train_txgnn['y_type'])]
    train_txgnn['x_index'] = train_txgnn['x_id_type'].map(id_type2index).astype(int)
    train_txgnn['y_index'] = train_txgnn['y_id_type'].map(id_type2index).astype(int)
    train_txgnn = train_txgnn[['x_index', 'y_index', 'relation']]
    train_txgnn["xy"] = (
        train_txgnn["x_index"].astype(str) + "|" +
        train_txgnn["y_index"].astype(str)
    )
    
    ###prepare val data
    nodes['id_type'] = [f"{i}|{j}" for i, j in zip(nodes['node_id'], nodes['node_type'])]
    id_type2index = dict(zip(nodes['id_type'], nodes.index))
    val_txgnn['x_id'] = val_txgnn['x_id'].apply(normalize_id)
    val_txgnn['y_id'] = val_txgnn['y_id'].apply(normalize_id)
    val_txgnn['x_id_type'] = [f"{i}|{j}" for i, j in zip(val_txgnn['x_id'], val_txgnn['x_type'])]
    val_txgnn['y_id_type'] = [f"{i}|{j}" for i, j in zip(val_txgnn['y_id'], val_txgnn['y_type'])]
    val_txgnn['x_index'] = val_txgnn['x_id_type'].map(id_type2index).astype(int)
    val_txgnn['y_index'] = val_txgnn['y_id_type'].map(id_type2index).astype(int)
    val_txgnn = val_txgnn[['x_index', 'y_index', 'relation']]
    val_txgnn["xy"] = (
        val_txgnn["x_index"].astype(str) + "|" +
        val_txgnn["y_index"].astype(str)
    )
    
    ###prepare test data
    nodes['id_type'] = [f"{i}|{j}" for i, j in zip(nodes['node_id'], nodes['node_type'])]
    id_type2index = dict(zip(nodes['id_type'], nodes.index))
    test_txgnn['x_id'] = test_txgnn['x_id'].apply(normalize_id)
    test_txgnn['y_id'] = test_txgnn['y_id'].apply(normalize_id)
    test_txgnn['x_id_type'] = [f"{i}|{j}" for i, j in zip(test_txgnn['x_id'], test_txgnn['x_type'])]
    test_txgnn['y_id_type'] = [f"{i}|{j}" for i, j in zip(test_txgnn['y_id'], test_txgnn['y_type'])]
    test_txgnn['x_index'] = test_txgnn['x_id_type'].map(id_type2index).astype(int)
    test_txgnn['y_index'] = test_txgnn['y_id_type'].map(id_type2index).astype(int)
    test_txgnn = test_txgnn[['x_index', 'y_index', 'relation']]
    test_txgnn["xy"] = (
        test_txgnn["x_index"].astype(str) + "|" +
        test_txgnn["y_index"].astype(str)
    )
    return train_txgnn, val_txgnn, test_txgnn

######################################################################

def correct_order(df, type2order):

    # optional: avoid KeyError/except in a loop
    target = df['relation'].map(type2order).fillna('keep_it_as_it_is')
    
    order = df['x_type'].astype(str) + '|' + df['y_type'].astype(str)
    keep = (target == 'keep_it_as_it_is') | (order == target)
    
    out = df.copy()
    
    # rows to swap
    sw = ~keep
    
    # swap x/y columns for the rows that need it
    x_cols = ['x_type', 'x_id', 'x_idx']
    y_cols = ['y_type', 'y_id', 'y_idx']
    
    out.loc[sw, x_cols], out.loc[sw, y_cols] = (
        out.loc[sw, y_cols].to_numpy(),
        out.loc[sw, x_cols].to_numpy()
    )
    
    return out

#prepare TxGNN data
def normalize_id(x):
    x = str(x)

    # Only convert things like "123.0", "45.000"
    if "." in x and "_" not in x:
        try:
            f = float(x)
            if f.is_integer():
                return str(int(f))
        except ValueError:
            pass

    return x