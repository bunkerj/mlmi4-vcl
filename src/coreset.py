################################################################################
# Coreset Heuristics ###########################################################
################################################################################

# Random Selection
def coreset_rand(x_coreset, y_coreset, x_train, y_train, coreset_size):
    # randomly permute the indices
    idx = torch.randperm(x_train.shape[0])[:coreset_size]
    # at first timestep, x_coreset and y_coreset should be set to "torch.FloatTensor()"
    x_coreset = torch.cat([x_coreset, x_train[idx[:coreset_size],:]], dim=0)
    y_coreset = torch.cat([y_coreset, y_train[idx[:coreset_size]]], dim=0)

    # remaining indices form the training data
    x_train = x_train[idx[coreset_size:],:]
    y_train = y_train[idx[coreset_size:]]

    return x_coreset, y_coreset, x_train, y_train
