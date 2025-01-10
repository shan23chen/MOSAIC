import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
import argparse
import os

# Define constant variables
LAYERS = [9, 20, 31]
LANGUAGES = ["arabic", "english", "french", "german", "hindi", "italian", "portuguese", "spanish"]
DATADIR = '/PHShome/sc1051/desktop/sae_llava/src/output_llm_both/'

TRAINING_DICT = {
    0: '9b_original',
    1: '9b_translated',
    2: '9b_it_translated',
    3: '9b_it_original'
}

def clip_features(data, threshold=1):
    """Clip feature values: set values less than threshold to 0, others to 1."""
    return np.where(data['features'] < threshold, 0, 1)

def load_data(language, layer, path_index, width=16, top_n=0):
    """
    Load NPZ data for a given language, layer, and file variant index.
    Returns: loaded data and corresponding file path
    """
    paths = [
        f"{DATADIR}google_gemma-2-9b/cardiffnlp_tweet_sentiment_multilingual/{language}/layer_{layer}/{width}k/{layer}_{top_n}_llm_features.npz",
        f"{DATADIR}google_gemma-2-9b/AIM-Harvard_tweet_sentiment_multilingual/{language}/layer_{layer}/{width}k/{layer}_{top_n}_llm_features.npz",
        f"{DATADIR}google_gemma-2-9b-it/AIM-Harvard_tweet_sentiment_multilingual/{language}/layer_{layer}/{width}k/{layer}_{top_n}_llm_features.npz",
        f"{DATADIR}google_gemma-2-9b-it/cardiffnlp_tweet_sentiment_multilingual/{language}/layer_{layer}/{width}k/{layer}_{top_n}_llm_features.npz"
    ]
    try:
        path = paths[path_index]
        data = np.load(path)
        return data, path
    except IndexError:
        raise ValueError(f"Invalid path index: {path_index}. Must be between 0 and {len(paths)-1}.")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {path}.")

def train_and_test(train, test, clip_value, features_choice):
    """
    Train on training data and evaluate on test data using logistic regression.
    """
    if features_choice == 'features':
        X_train = train['features']
        X_test = test['features']
    else:
        X_train = train['hidden_states']
        X_test = test['hidden_states']

    if clip_value:
        X_train = clip_features(train, threshold=clip_value)
        X_test = clip_features(test, threshold=clip_value)

    y_train = train['label']
    y_test = test['label']

    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)

def cross_validate(data, clip_value, n_splits=5):
    """Perform K-fold cross-validation on the given dataset."""
    features = clip_features(data, threshold=clip_value) if clip_value else data['features']
    labels = data['label']

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, f1_scores = [], []

    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

    return np.mean(accuracies), np.mean(f1_scores)

def main():
    parser = argparse.ArgumentParser(description="Multilingual Sentiment Analysis using Logistic Regression.")
    parser.add_argument("--path", type=int, required=True, help="Index for selecting data path variant.")
    parser.add_argument("--features", type=str, default='features', help="Choose feature type: 'features' or other for hidden_states.")
    parser.add_argument("--top_n", type=int, default=0, help="Top_n parameter for file selection.")
    parser.add_argument("--width", type=int, default=16, help="Width parameter for file path.")
    args = parser.parse_args()

    results = []
    if args.path not in TRAINING_DICT:
        raise ValueError(f"Invalid --path value: {args.path}. Must be one of {list(TRAINING_DICT.keys())}.")

    for layer in LAYERS:
        for clip_value in [1]:
            print(f"\nProcessing Layer {layer}, with Clipping" if clip_value else f"\nProcessing Layer {layer}, without Clipping")
            for train_lang in LANGUAGES:
                try:
                    train, _ = load_data(train_lang, layer, args.path, args.width, args.top_n)
                except Exception as e:
                    print(e)
                    continue

                for test_lang in LANGUAGES:
                    print(f"Training on {train_lang.upper()}, Testing on {test_lang.upper()}")
                    if train_lang == test_lang:
                        accuracy, f1 = cross_validate(train, clip_value=clip_value, n_splits=5)
                    else:
                        try:
                            test, _ = load_data(test_lang, layer, args.path, args.width, args.top_n)
                        except Exception as e:
                            print(e)
                            continue
                        accuracy, f1 = train_and_test(train, test, clip_value=clip_value, features_choice=args.features)

                    print(f"Accuracy: {accuracy:.4f}, F1 score: {f1:.4f}")
                    results.append({
                        'Layer': layer,
                        'Clipping': 'Clipped' if clip_value else 'Original',
                        'Training Language': train_lang.upper(),
                        'Testing Language': test_lang.upper(),
                        'Accuracy': accuracy,
                        'F1 Score': f1
                    })

    df_results = pd.DataFrame(results)

    # Determine model task string from last used path
    try:
        model_task = _.split(DATADIR)[1].split("_dataset/")[0]
    except Exception:
        model_task = "unknown_model_task"

    output_csv_path = f"{TRAINING_DICT[args.path]}{model_task}_cross_language_results.csv"
    df_results.to_csv(output_csv_path, index=False)
    print(f"\nResults saved to {output_csv_path}")

    # Generate heatmaps
    for layer in LAYERS:
        df_layer_clip = df_results[(df_results['Layer'] == layer) & (df_results['Clipping'] == 'Clipped')]
        if df_layer_clip.empty:
            continue

        accuracy_matrix = df_layer_clip.pivot(index='Training Language', columns='Testing Language', values='Accuracy')
        f1_matrix = df_layer_clip.pivot(index='Training Language', columns='Testing Language', values='F1 Score')

        for metric_matrix, metric_name, cmap in [(accuracy_matrix, "Accuracy", "Blues"), (f1_matrix, "F1 Score", "Greens")]:
            plt.figure(figsize=(8, 6))
            sns.heatmap(metric_matrix, annot=True, fmt=".4f", cmap=cmap)
            plt.title(f"Cross-Language {metric_name} Heatmap (Layer {layer}, Clipped)")
            plt.xlabel("Testing Language")
            plt.ylabel("Training Language")
            plt.tight_layout()
            filename = f"{TRAINING_DICT[args.path]}{model_task}_{metric_name.lower().replace(' ', '_')}_heatmap_layer_{layer}_clipped.png"
            plt.savefig(filename)
            plt.close()

if __name__ == "__main__":
    main()