def accuracy(output, labels):
    correct_false = 0
    correct_true = 0
    total_false = 0
    total_true = 0
    for o, l in zip(output, labels):
        if l == 0:
            total_false += 1
            if o == 0:
                correct_false += 1
        elif l == 1:
            total_true += 1
            if o == 1:
                correct_true += 1
    if total_true == 0:
        total_true = 1
    if total_false == 0:
        total_false = 1
    return (correct_true + correct_false)/(total_true + total_false), correct_true, total_true, correct_false, total_false
