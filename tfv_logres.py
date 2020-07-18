import pandas as pd

from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection

if __name__ == '__main__':
    df = pd.read_csv('imdb.csv')

    df['kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    y = df.label.values

    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X=df,y=y)):
        df.loc[v_, 'kfold'] = f

    for fold_ in range(5):

        test_df = df[df.kfold == fold_].reset_index(drop=True)
        train_df = df[df.kfold != fold_].reset_index(drop=True)

        tfidf_vec = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None, ngram_range=(2,2))
        tfidf_vec.fit(train_df.text)

        Xtrain = tfidf_vec.transform(train_df.text)
        Xtest = tfidf_vec.transform(test_df.text)

        model = LogisticRegression()
        model.fit(Xtrain, train_df.label)

        preds = model.predict(Xtest)
        accuracy = metrics.accuracy_score(test_df.label, preds)

        print(f"kfold : {fold_}")
        print(f"accuracy : {accuracy}")
        print("")


