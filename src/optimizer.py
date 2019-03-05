def minimizeLoss(maxIter, optimizer, lossFunc, lossFuncArgs):
    for i in range(maxIter):
        optimizer.zero_grad()
        loss = lossFunc(*lossFuncArgs)
        if maxIter > 1:
            loss.backward(retain_graph = True)
        else:
            loss.backward()
        optimizer.step()
    return loss
