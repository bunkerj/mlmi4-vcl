def minimizeLoss(maxIter, lossFunc, optimizer, lossFuncArgs):
    for i in range(maxIter):
        optimizer.zero_grad()
        loss = lossFunc(*lossFuncArgs)
        loss.backward(retain_graph = True)
        optimizer.step()
        if i % 100 == 0:
            print(loss)
