def kai_abam(train):
    drop_index = train.index[((train["often_laugh"]==1) & (train["joy"] <= 2)) | ((train["often_laugh"]==0) & (train["joy"] <= 3))]
    train = train.drop(drop_index)
    return train

def egao_abam(probs, x_test, a_level, b_level):
    tmp = [0] * len(probs)
    for i in range(len(probs)):
        # クラスが1の時
        if (x_test[i][2] == 1):
            if (probs[i][1] >= a_level):
                tmp[i] = 1
            else:
                tmp[i] = 0
        else:
            if (probs[i][1] >= b_level):
                tmp[i] = 1
            else:
                tmp[i] = 0
    return tmp
