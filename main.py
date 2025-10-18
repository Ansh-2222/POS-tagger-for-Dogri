import re
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

# ---------- Step 1: Load dataset ----------
def load_data(file_path):
    sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            sentence = []
            for tok in tokens:
                if "/" not in tok:
                    continue
                parts = tok.rsplit("/", 1)
                if len(parts) != 2 or not parts[0].strip():
                    continue
                word, tag = parts
                sentence.append((word, tag))
            if sentence:
                sentences.append(sentence)
    return sentences


# ---------- Step 2: Feature extraction ----------
def extract_features(word, idx, sentence):
    if not word:
        word = "<EMPTY>"

    features = {
        "word": word,
        "lowercase": word.lower(),
        "isCapitalized": word[0].isupper() if len(word) > 0 else False,
        "prefix-1": word[:1],
        "prefix-2": word[:2],
        "suffix-1": word[-1:],
        "suffix-2": word[-2:],
        "isDigit": word.isdigit(),
    }

    if idx > 0:
        features["prev_word"] = sentence[idx-1][0].lower()
    else:
        features["prev_word"] = "<START>"

    if idx < len(sentence)-1:
        features["next_word"] = sentence[idx+1][0].lower()
    else:
        features["next_word"] = "<END>"

    return features


# ---------- Step 3: Prepare dataset ----------
def prepare_dataset(sentences):
    X, y = [], []
    for sentence in sentences:
        for idx, (word, tag) in enumerate(sentence):
            feats = extract_features(word, idx, sentence)
            X.append(feats)
            y.append(tag)
    return X, y


# ---------- Step 4: Train SVM ----------
def train_svm(X, y):
    vec = DictVectorizer(sparse=True)
    X_vec = vec.fit_transform(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y_enc, test_size=0.2, random_state=42
    )

    clf = LinearSVC(max_iter=5000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n--- Evaluation Report ---")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=range(len(le.classes_)),
            target_names=le.classes_,
            zero_division=0
        )
    )

    return clf, le, vec


# ---------- Step 5: Save model ----------
if __name__ == "__main__":
    data_path = "DogriDataset.txt"
    sentences = load_data(data_path)
    print(f"Loaded {len(sentences)} sentences")

    X, y = prepare_dataset(sentences)
    clf, le, vec = train_svm(X, y)

    joblib.dump(clf, "model_files/model.pkl")
    joblib.dump(le, "model_files/label_encoder.pkl")
    joblib.dump(vec, "model_files/vectorizer.pkl")

    print("✅ Model trained and saved successfully!")
