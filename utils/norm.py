def binary_auc(labels, probs):
    ###initialize
    P = 0
    N = 0
    for i in labels:
        if (i == 1):
            P += 1
        else:
            N += 1

    TP = 0
    FP = 0
    TPR_last = 0
    FPR_last = 0
    AUC = 0
    pair = zip(probs, labels)
    pair = sorted(pair, key=lambda x: x[0], reverse=True)
    i = 0
    while i < len(pair):
        if (pair[i][1] == 1):
            TP += 1
        else:
            FP += 1
        ### maybe have the same probs
        while (i + 1 < len(pair) and pair[i][0] == pair[i + 1][0]):
            i += 1
            if (pair[i][1] == 1):
                TP += 1
            else:
                FP += 1
        TPR = TP / P
        FPR = FP / N
        AUC += 0.5 * (TPR + TPR_last) * (FPR - FPR_last)
        TPR_last = TPR
        FPR_last = FPR
        i += 1
    return AUC