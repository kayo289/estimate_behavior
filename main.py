from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import smile
import kai

# ----memo----
# joy 1~5
# interested 1, not 0
# often_laugh 1, not 0
# knowledge 1, ない   0
# time,joy,interested,often_laugh,knowledge

def main(a_level,b_level,filename,often_laugh,knowledge,cnt):
    interested = kai.readfiles(filename)
    train = kai.pleasant_estimation(filename, often_laugh, knowledge, cnt, interested)

    if level == 2:
        train = smile.kai_abam(train)

    x_train, x_test, y_train, y_test = train_test_split(
        train.loc[:, ['time', 'knowledge','often_laugh']].values,
        train['interested'].values,
        test_size = 0.3,
    )

    y_train=y_train.astype('int')
    scl = StandardScaler()
    scl.fit(x_train) #学習用データで標準化
    x_train_std = scl.transform(x_train)
    x_test_std = scl.transform(x_test)

    # ロジスティック回帰分析
    clf = LogisticRegression()
    classifier = clf.fit(x_train_std, y_train) #訓練データから学習を行う
    # print("0である確率, 1である確率")
    probs = classifier.predict_proba(x_test_std)
    predict = clf.predict(x_test_std)

    if (level == 2):
        predict = smile.egao_abam(probs, x_test, a_level, b_level)

    y_test=y_test.astype('int')
    print( "正解率:{:.2f}%".format(accuracy_score(y_test, predict) * 100 ))


if __name__ == '__main__':
    # ------------データ---------------
    a_level = 0.398
    b_level = 0.712
    filename = ["fukushima","houjin","neishi","oikawa","ueda"]
    often_laugh = [0,1,1,1,0]
    knowledge = [0,1,1,1,1]
    cnt = 20
    # --------------------------------
    level = 2
    main(a_level,b_level,filename,often_laugh,knowledge,cnt)