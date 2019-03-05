def minimizeLoss(maxIter, optimizer, lossFunc, lossFuncArgs):
    for i in range(maxIter):
        optimizer.zero_grad()
        loss = lossFunc(*lossFuncArgs)
        loss.backward(retain_graph = True)
        optimizer.step()
        return loss 
