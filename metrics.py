def confusion_matrix(pred, real):
    """
        Input: List of Prediction and real data.
        Output: confusion_matrix as dataframe, precision, recall, f1-score
        |gt \ pred | P | N |
        -----------|---|---|
        |    P     |TP |FN |
        |----------|---|---|
        |    N     |FP |TN |
        |----------|---|---|

        TP: Model predicted class 1 for class 1.
        FP: Model predicted class 1 for class 0.
        FN: Model predicted class 0 for class 1.
        TN: Model predicted class 0 for class 0.

        precision: How many did we predicted correct?
                    TP / (TP + FP)
        recall: How many of predicted correct were classified correctly?
                    TP / (TP + FN)
        f1: Harmonic Mean of precision and recall
    """

    x = pred
    y = real
    
    tp = sum(((x == 1) == True) * ((y == 1)== True))
    fp = sum(((x == 1) == True) * ((y == 1)== False))

    fn = sum(((x == 1) == False) * ((y == 1)== True))
    tn = sum(((x == 1) == False) * ((y == 1)== False))
    
#     print("|gt \ pred | P | N |")
#     print("-----------|---|---|")
#     print(f"|    P     | {tp} | {fn} |")
#     print("|----------|---|---|")
#     print(f"|    N     | {fp} | {tn} |")
#     print("|----------|---|---|")
    

    lbl = ['P', 'N']
    index = ['P', 'N']

    df = pd.DataFrame(data = [[tp, fn], [fp, tn]], index=index, columns=lbl)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    
    return df, round(precision, 5), round(recall, 5), round(f1, 5)
    
