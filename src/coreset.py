
import torch

# Random Selection
def coreset_rand(x_train, y_train, coreset_size):
    # randomly permute the indices
    idx = torch.randperm(x_train.shape[0])

    x_coreset = x_train[idx[:coreset_size]]
    y_coreset = y_train[idx[:coreset_size]]

    # remaining indices form the training data
    x_train = x_train[idx[coreset_size:],:]
    y_train = y_train[idx[coreset_size:]]

    return x_coreset, y_coreset, x_train, y_train

# K-center Selection
def coreset_k(x_train, y_train, coreset_size):
    # distance tensor contains the distance to the furthest center, for each data point
    distance = torch.ones(x_train.shape[0])*float('inf').type(FloatTensor)
    # randomly select the first center
    cur_idx = torch.randint(x_train.shape[0], (1,)).item()
    # list idx will contain the indices of the coreset

    idx = [cur_idx]
    for i in range(1, coreset_size):
        # subtract the image of current center from all other images
        # x_diff[s,:] is the difference between s'th image and c'th image (c = cur_idx)
        x_diff = x_train - x_train[cur_idx,:].expand(x_train.shape)
        # torch.norm(x_diff, dim=1) obtains the norm of each images
        # update the distance by selecting the shorter distance for each image
        distance = torch.min(distance, torch.norm(x_diff, dim=1))
        # the image with the highest distance is selected as the next center
        new_idx = torch.max(distance, 0)[1].item()
        idx.append(new_idx)
        # now, new_idx is the current center
        cur_idx = new_idx

    # at first timestep, x_coreset and y_coreset should be set to "torch.FloatTensor()"
    x_coreset = x_train[idx,:]
    y_coreset = y_train[idx]

    # idx_train: all the indices not in idx, used to update training data
    idx_train = [i for i in range(x_train.shape[0]) if i not in idx]

    # remaining indices form the training data
    x_train = x_train[idx_train,:]
    y_train = y_train[idx_train]

    return x_coreset, y_coreset, x_train, y_train
