#必要なツールのインポート
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from kai import pleasant_estimation

# joy 1~5
# interested 1, not 0
# often_laugh 1, not 0
# knowledge 1, ない   0
# time,joy,interested,often_laugh,knowledge

# ------------abam用---------------
a_level = 0.398
b_level = 0.712
# --------------------------------

level = 2

# train = pd.read_csv("csv/level1.csv", sep=",") #(1)
train = pleasant_estimation()
# print(type(train))
# print(train)

#快状態推定のabam
if level == 2:
    drop_index = train.index[((train["often_laugh"]==1) & (train["joy"] <= 2)) | ((train["often_laugh"]==0) & (train["joy"] <= 3))]
    train = train.drop(drop_index)
# print(type(train))
#データセットをテスト用と訓練用に分ける
x_train, x_test, y_train, y_test = train_test_split(
    train.loc[:, ['time', 'knowledge','often_laugh']].values,
    train['interested'].values,
    test_size = 0.3,
)

#データを標準化
# print(y_train)
y_train=y_train.astype('int')
scl = StandardScaler()
scl.fit(x_train) #学習用データで標準化
x_train_std = scl.transform(x_train)
x_test_std = scl.transform(x_test)

# ロジスティック回帰分析
clf = LogisticRegression()

classifier = clf.fit(x_train_std, y_train)#訓練データから学習を行う

# print("0である確率, 1である確率")
probs = classifier.predict_proba(x_test_std)
predict = clf.predict(x_test_std)
# print(predict)

# ---------------abam-----------------
if (level == 2):
    predict = [0] * len(probs)
    for i in range(len(probs)):
        # クラスが1の時
        if (x_test[i][2] == 1):
            if (probs[i][1] >= a_level):
                predict[i] = 1
            else:
                predict[i] = 0
        else:
            if (probs[i][1] >= b_level):
                predict[i] = 1
            else:
                predict[i] = 0
# 変化がないのになぜ、入れなきゃエラーが出るの？
y_test=y_test.astype('int')
print( "正解率:{:.2f}%".format(accuracy_score(y_test, predict) * 100 ))