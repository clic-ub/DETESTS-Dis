import argparse
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


def parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create baselines for both task 1 and task 2",
    )
    parser.add_argument(
        "-train",
        type=str,
        default="data/train.csv",
        help="Train file",
    )
    parser.add_argument(
        "-test",
        type=str,
        default="data/test.csv",
        help="Test file",
    )
    parser.add_argument(
        "-folder",
        type=str,
        default="baselines",
        help="Folder for saving the predictions",
    )
    parser.add_argument(
        "-model",
        type=str,
        default="all",
        choices=["all", "zeros", "ones", "random", "tfidf", "fast"],
        help="Model to run",
    )

    args = parser.parse_args()

    return args


# ------------ Models ---------------------------------


def random_classifier(X_train, y_train, X_test):
    """Weighted random classifier"""
    np.random.seed(seed=42)
    vals, prob = np.unique(y_train, return_counts=True)
    prob = prob / y_train.shape[0]
    pred = np.random.choice(vals, size=(X_test.shape[0],), p=prob)
    return pred


def svm(X_train, y_train, X_test, kernel="linear", C=1):
    """Fit and predict SVC"""
    cls = SVC(kernel=kernel, C=C, random_state=42)
    cls.fit(X_train, y_train)
    pred = cls.predict(X_test)
    return pred


# ------------ Vectorizations ---------------------------------


def tfidf(X_train, X_test, ngrams=(1, 1), max_features=10000):
    """TF-IDF + SVC"""
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        ngram_range=ngrams,
        max_features=max_features,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec


def fast_text(X_trn, X_tst):
    """Get text embedding"""
    import spacy

    # Disable all components except tok2vec
    nlp = spacy.load("es_core_news_md", enable=["tok2vec"])

    def get_emb(text):
        sentences = list(nlp.pipe(text.tolist()))
        text_vec = np.array([sent.vector for sent in sentences])
        return text_vec

    X_train_vec = get_emb(X_trn)
    X_test_vec = get_emb(X_tst)

    return X_train_vec, X_test_vec


# ------------ Main ---------------------------------


def main(args):
    model = args.model

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    X_trn = train["text"]
    X_tst = test["text"]
    X_train = X_trn.to_numpy()
    X_test = X_tst.to_numpy()

    y_train = train["stereotype"].to_numpy()

    results = test[["id"]].copy()
    results2 = test[["id"]].copy()

    # Train set for task 2
    train2 = train[train["stereotype"] == 1]
    X_train2 = train2["text"].to_numpy()

    if model in ("zeros", "all"):
        results["stereotype"] = 0
        results.to_csv(os.path.join(args.folder, "all_zeros_t1_hard.csv"), index=False)

        results2["stereotype"] = 0
        results2["implicit"] = 0
        results2.to_csv(os.path.join(args.folder, "all_zeros_t2_hard.csv"), index=False)

    if model in ("ones", "all"):
        results["stereotype"] = 1
        results.to_csv(os.path.join(args.folder, "all_ones_t1_hard.csv"), index=False)

        results2["stereotype"] = 1
        results2["implicit"] = 1
        results2.to_csv(os.path.join(args.folder, "all_ones_t2_hard.csv"), index=False)

    if model in ("random", "all"):
        pred = random_classifier(X_train, y_train, X_test)
        results["stereotype"] = pred
        results.to_csv(os.path.join(args.folder, "random_classifier_t1_hard.csv"), index=False)

        # mask
        test_mask = pred > 0.5
        test2 = test[test_mask]
        X_test2 = test2["text"].to_numpy()

        # implicit
        results2["stereotype"] = pred
        results2["implicit"] = 0
        results2.loc[test_mask, "implicit"] = random_classifier(
            X_train2, train2["implicit"].to_numpy(), X_test2
        )
        results2.to_csv(os.path.join(args.folder, "random_classifier_t2_hard.csv"), index=False)

    if model in ("tfidf", "all"):
        # stereotype
        X_train_vec, X_test_vec = tfidf(X_train, X_test, ngrams=(1, 3))
        pred = svm(X_train_vec, y_train, X_test_vec)
        results.to_csv(os.path.join(args.folder, "tfidf_svc_t1_hard.csv"), index=False)

        # mask
        test_mask = pred > 0.5
        test2 = test[test_mask]
        X_test2 = test2["text"].to_numpy()
        X_train_vec2, X_test_vec2 = tfidf(X_train2, X_test2, ngrams=(1, 3))

        # implicit
        results2["stereotype"] = pred
        results2["implicit"] = 0
        results2.loc[test_mask, "implicit"] = svm(
            X_train_vec2, train2["implicit"].to_numpy(), X_test_vec2
        )
        results2.to_csv(os.path.join(args.folder, "tfidf_svc_t2_hard.csv"), index=False)

    if model in ("fast", "all"):
        # stereotype
        X_train_vec, X_test_vec = fast_text(X_trn, X_tst)
        pred = svm(X_train_vec, y_train, X_test_vec)
        results.to_csv(os.path.join(args.folder, "fast_text_svc_t1_hard.csv"), index=False)

        # mask
        test_mask = pred > 0.5
        test2 = test[test_mask]
        X_test2 = test2["text"].to_numpy()
        X_train_vec2, X_test_vec2 = fast_text(X_train2, X_test2)

        # implicit
        results2["stereotype"] = pred
        results2["implicit"] = 0
        results2.loc[test_mask, "implicit"] = svm(
            X_train_vec2, train2["implicit"].to_numpy(), X_test_vec2
        )
        results2.to_csv(os.path.join(args.folder, "fast_text_svc_t2_hard.csv"), index=False)


if __name__ == "__main__":
    args = parsing()
    main(args)
