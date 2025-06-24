---
layout: single
title:  "Movie Recommendation based on Plot (Supervised)"
subtitle: "Recommendation"
categories: [python, Machine Learning]
tag: [RandomForest, XGBoost, Logistic Regression, KNN, Confusion Matrix, Ensemble, Roc Curve, Precision-Recall Curve, Calibration Curve]
toc: true
---


## Movie Lens/IMDB/OMDB Movie Recommendation based on Plot

# supervised_pipeline.py

# Import necessary libraries
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix, save_npz, load_npz
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve

import warnings
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# Load CSV and clean specific numeric columns
def load_and_clean_data(path):
    df = pd.read_csv(path, encoding="ISO-8859-1")
    df[['runtimeMinutes', 'startYear']] = df[['runtimeMinutes', 'startYear']].replace('\\N', np.nan)
    df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'], errors='coerce')
    df['startYear'] = pd.to_numeric(df['startYear'], errors='coerce')
    df['runtimeMinutes'] = df['runtimeMinutes'].fillna(df['runtimeMinutes'].median())
    df['startYear'] = df['startYear'].fillna(df['startYear'].median())
    return df

# Generate binary labels based on avg_rating column
def generate_labels(df):
    df = df[df['avg_rating'].notna()]
    df['avg_rating'] = df['avg_rating'].astype(int)
    df['label'] = df['avg_rating'].apply(lambda x: 0 if x <= 2 else 1)
    return df

# Load or create BERT embeddings from the plot column
def load_or_generate_bert_embeddings(df, path="bert_mpnet_embeddings.npy"):
    if os.path.exists(path):
        print(f"Loading BERT embeddings from {path}...")
        return np.load(path)
    else:
        print("Generating BERT embeddings...")
        model = SentenceTransformer('all-mpnet-base-v2')
        plots = df['plot'].fillna('').tolist()
        embeddings = model.encode(plots, show_progress_bar=True).astype(np.float32)
        np.save(path, embeddings)
        print(f"Embeddings saved to {path}.")
        return embeddings

# Convert categorical column into multi-hot encoded format
# Collapse rare values into 'Others' if top_k is specified
def multi_hot_encode_with_others(df, column, top_k=None):
    df[column + '_list'] = df[column].fillna('').apply(lambda x: [s.strip() for s in x.split(',') if s.strip()])
    all_items = df[column + '_list'].explode()
    if top_k:
        top_items = set(all_items.value_counts().nlargest(top_k).index)
        df[column + '_list'] = df[column + '_list'].apply(lambda lst: [x if x in top_items else 'Others' for x in lst])
    mlb = MultiLabelBinarizer()
    return pd.DataFrame(mlb.fit_transform(df[column + '_list']), columns=[f"{column}_{c}" for c in mlb.classes_])

# Combine text embeddings, categorical multi-hot features, and numerical features
def assemble_features(df, X_text):
    genres = multi_hot_encode_with_others(df, 'genres')
    actors = multi_hot_encode_with_others(df, 'actors', top_k=100)
    writers = multi_hot_encode_with_others(df, 'writer', top_k=50)
    directors = multi_hot_encode_with_others(df, 'director', top_k=20)
    countries = multi_hot_encode_with_others(df, 'country', top_k=20)
    languages = multi_hot_encode_with_others(df, 'language', top_k=10)

    X_cat_df = pd.concat([genres, actors, writers, directors, countries, languages], axis=1)
    X_cat = csr_matrix(X_cat_df.values)
    X_num_df = df[['runtimeMinutes', 'startYear', 'num_rating']]
    X_num = StandardScaler().fit_transform(X_num_df)

    feature_names = (
        [f"text_{i}" for i in range(X_text.shape[1])] +
        list(X_cat_df.columns) +
        list(X_num_df.columns)
    )

    return hstack([csr_matrix(X_text), X_cat, X_num]), feature_names

# Apply SMOTE for class balancing (or load from cache if available)
def apply_smote(X_train, y_train, X_file="X_resampled.npz", y_file="y_resampled.npy"):
    if os.path.exists(X_file) and os.path.exists(y_file):
        print("Loading SMOTE-resampled data...")
        return load_npz(X_file), np.load(y_file)
    else:
        print("Applying SMOTE...")
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        save_npz(X_file, X_res)
        np.save(y_file, y_res)
        return X_res, y_res

# Compute ensemble results by averaging predicted probabilities
def ensemble_and_evaluate(y_test, probas):
    ensemble_proba = np.mean(probas, axis=0)
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, ensemble_pred)
    f1 = f1_score(y_test, ensemble_pred, average='macro')
    plot_confusion_matrix(y_test, ensemble_pred, labels=[0, 1], title="Ensemble Confusion Matrix")
    print(f"\n[Ensemble] Accuracy: {acc:.4f}")
    print(f"[Ensemble] Macro F1-score: {f1:.4f}")
    return acc, f1

# Plot a confusion matrix as a heatmap
def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Plot top-n feature importances from tree-based models
def plot_feature_importance(model, feature_names, top_n=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.show()

# Plot ROC curves for multiple models
def plot_roc_curve_multi(y_test, probas_dict):
    plt.figure(figsize=(7, 6))
    for model_name, proba in probas_dict.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (All Models)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot bar chart comparing accuracy and F1 score of models
def plot_model_performance(results):
    df_perf = pd.DataFrame([
        {"Model": name, "Accuracy": res['accuracy'], "F1-score": res['f1_score']}
        for name, res in results.items()
    ])
    df_perf.set_index("Model")[["Accuracy", "F1-score"]].plot(kind="bar", figsize=(8, 5), ylim=(0, 1))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.show()

# Plot Precision-Recall curves for multiple models
def plot_pr_curve_multi(y_test, probas_dict):
    plt.figure(figsize=(7, 6))
    for model_name, proba in probas_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, proba)
        plt.plot(recall, precision, lw=2, label=model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (All Models)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot Calibration Curves to show how well predicted probabilities reflect true likelihood
def plot_calibration_curve_multi(y_test, probas_dict):
    plt.figure(figsize=(7, 6))
    for model_name, proba in probas_dict.items():
        prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve (All Models)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
```
:::

::: {#96fa1155-5d0c-416e-97bb-a54467cdf518 .cell .code execution_count="2"}
``` python
from supervised_pipeline import *
```
:::

::: {#94c66a5b-c848-4295-9452-222719cef3bc .cell .code execution_count="3"}
``` python
def train_and_predict_models(X_resampled, y_resampled, X_test, y_test, model_dir="saved_models_3"):
    os.makedirs(model_dir, exist_ok=True)
    results, probas = {}, []
    models = {}

    model_defs = {
        "RandomForest": (RandomForestClassifier(random_state=42, n_jobs=-1), {
            'n_estimators': [100, 200],
            'max_depth': [None, 10]
        }),
        "XGBoost": (XGBClassifier(random_state=36, use_label_encoder=False, eval_metric='logloss', n_jobs=-1), {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }),
        "Logistic": (LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42), {
            'C': [0.1, 1.0, 10.0]
        }),
        "KNN": (KNeighborsClassifier(n_jobs=-1), {
            'n_neighbors': [3, 5, 7]
        })
    }

    for name, (model, params) in model_defs.items():
        path = os.path.join(model_dir, f"{name}_best_model_3.pkl")

        if os.path.exists(path):
            print(f"\n Loading {name} model from: {path}")
            model_info = joblib.load(path)
            best_model = model_info['model']
            mean_f1 = model_info.get('mean_f1')
            std_f1 = model_info.get('std_f1')

            if mean_f1 is not None and std_f1 is not None:
                print(f"[{name}] (Loaded) Cross-Validation F1-score: Mean = {mean_f1:.4f}, Std = {std_f1:.4f}")

        else:
            print(f"\n Training {name} using GridSearchCV...")
            grid = GridSearchCV(model, params, scoring='f1_macro', cv=5, n_jobs=-1, verbose=1)
            grid.fit(X_resampled, y_resampled)
            best_model = grid.best_estimator_

            mean_f1 = grid.cv_results_['mean_test_score'].mean()
            std_f1 = grid.cv_results_['std_test_score'].mean()
            print(f"[{name}] Cross-Validation F1-score: Mean = {mean_f1:.4f}, Std = {std_f1:.4f}")

            joblib.dump({
                'model': best_model,
                'mean_f1': mean_f1,
                'std_f1': std_f1
            }, path)
            print(f" Saved {name} model to: {path}")

        # Predict & Evaluate
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"\n [{name}] Best Model: {best_model}")
        print(f" [{name}] Test Accuracy: {acc:.4f}")
        print(f" [{name}] Test Macro F1-score: {f1:.4f}")

        results[name] = {
            'model': best_model,
            'accuracy': acc,
            'f1_score': f1,
            'mean_cv_f1': mean_f1,
            'std_cv_f1': std_f1
        }
        models[name] = best_model
        probas.append(y_proba)

        plot_confusion_matrix(y_test, y_pred, labels=[0, 1], title=f"{name} Confusion Matrix")

    return results, probas, models
```
:::

::: {#feb6c4f4-ad2d-4424-9b0b-8e9cd21cffef .cell .code execution_count="5"}
``` python
def main():
    # Load and clean data
    df = load_and_clean_data("../1. Data_preparation/df_final_frozen_62188.csv")
    df = generate_labels(df)

    # Load or generate text embeddings
    X_text = load_or_generate_bert_embeddings(df)

    # Assemble features
    X_all, feature_names = assemble_features(df, X_text)
    y = df['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42, stratify=y)


    # Train models and get predictions
    results, probas, models = train_and_predict_models(X_train, y_train, X_test, y_test)
    ensemble_and_evaluate(y_test, probas)

    # Convert list of probas into a dictionary keyed by model name
    model_names = list(results.keys())
    probas_dict = {model_name: probas[i] for i, model_name in enumerate(model_names)}
    

    # Visualizations for all models
    plot_roc_curve_multi(y_test, probas_dict)
    plot_pr_curve_multi(y_test, probas_dict)
    plot_calibration_curve_multi(y_test, probas_dict)
    plot_model_performance(results)

    tree_models = ["XGBoost", "RandomForest"]
    for name in tree_models:
        model = models.get(name)
        if hasattr(model, "feature_importances_"):
            print(f"\n Showing feature importances for: {name}")
            plot_feature_importance(model, feature_names, top_n=20)


    # XGBoost Misclassification
    model_xgb = models.get("XGBoost")
    if model_xgb:
        y_pred = model_xgb.predict(X_test)
    
        # Reconstruct test data based on original index
        df_test = df.iloc[y_test.index].copy()
        df_test["true"] = y_test.values
        df_test["pred"] = y_pred
    
        # Extracting misclassified samples
        false_negatives = df_test[(df_test["true"] == 1) & (df_test["pred"] == 0)].head(5)
        false_positives = df_test[(df_test["true"] == 0) & (df_test["pred"] == 1)].head(5)
    
        print("\nFalse Negatives (Positive but predicted as negative):")
        print(false_negatives[["primaryTitle", "plot", "genres", "country", "language", "true", "pred"]])
    
        print("\nFalse Positives (Negative but predicted positive):")
        print(false_positives[["primaryTitle", "plot", "genres", "country", "language", "true", "pred"]])






if __name__ == "__main__":
    main()
```

::: {.output .stream .stdout}
    Loading BERT embeddings from bert_mpnet_embeddings.npy...

     Loading RandomForest model from: saved_models_3\RandomForest_best_model_3.pkl
    [RandomForest] (Loaded) Cross-Validation F1-score: Mean = 0.5878, Std = 0.0031

     [RandomForest] Best Model: RandomForestClassifier(n_jobs=-1, random_state=42)
     [RandomForest] Test Accuracy: 0.6577
     [RandomForest] Test Macro F1-score: 0.6022
:::

::: {.output .display_data}
![](vertopal_a43c76182a72468ea4b8bf98809c9515/3881904ff7867d21327241b95c231007d5bc3689.png)
:::

::: {.output .stream .stdout}

     Loading XGBoost model from: saved_models_3\XGBoost_best_model_3.pkl
    [XGBoost] (Loaded) Cross-Validation F1-score: Mean = 0.6577, Std = 0.0042

     [XGBoost] Best Model: XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
                  colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
                  early_stopping_rounds=None, enable_categorical=False,
                  eval_metric='logloss', gamma=0, gpu_id=-1,
                  grow_policy='depthwise', importance_type=None,
                  interaction_constraints='', learning_rate=0.1, max_bin=256,
                  max_cat_to_onehot=4, max_delta_step=0, max_depth=7, max_leaves=0,
                  min_child_weight=1, missing=nan, monotone_constraints='()',
                  n_estimators=100, n_jobs=-1, num_parallel_tree=1,
                  predictor='auto', random_state=36, reg_alpha=0, reg_lambda=1, ...)
     [XGBoost] Test Accuracy: 0.6940
     [XGBoost] Test Macro F1-score: 0.6721
:::

::: {.output .display_data}
![](vertopal_a43c76182a72468ea4b8bf98809c9515/51db271da17c535593e27f3ff903e521e0ff915f.png)
:::

::: {.output .stream .stdout}

     Loading Logistic model from: saved_models_3\Logistic_best_model_3.pkl
    [Logistic] (Loaded) Cross-Validation F1-score: Mean = 0.6571, Std = 0.0039

     [Logistic] Best Model: LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
     [Logistic] Test Accuracy: 0.6804
     [Logistic] Test Macro F1-score: 0.6536
:::

::: {.output .display_data}
![](vertopal_a43c76182a72468ea4b8bf98809c9515/87d2405d85f8d06c78ac91bb781852097ba29b2a.png)
:::

::: {.output .stream .stdout}

     Loading KNN model from: saved_models_3\KNN_best_model_3.pkl
    [KNN] (Loaded) Cross-Validation F1-score: Mean = nan, Std = nan

     [KNN] Best Model: KNeighborsClassifier(n_jobs=-1, n_neighbors=3)
     [KNN] Test Accuracy: 0.6366
     [KNN] Test Macro F1-score: 0.6211
:::

::: {.output .display_data}
![](vertopal_a43c76182a72468ea4b8bf98809c9515/256f438098f736be1e724650a7cb89f16cb0d8d8.png)
:::

::: {.output .display_data}
![](vertopal_a43c76182a72468ea4b8bf98809c9515/700ce31b4782c25ca475295b25c8d582533f4459.png)
:::

::: {.output .stream .stdout}

    [Ensemble] Accuracy: 0.6897
    [Ensemble] Macro F1-score: 0.6619
:::

::: {.output .display_data}
![](vertopal_a43c76182a72468ea4b8bf98809c9515/de6c0dd1d6745a3848876171805ae042fe05c1bb.png)
:::

::: {.output .display_data}
![](vertopal_a43c76182a72468ea4b8bf98809c9515/f3011b42e4a60dcf8933478fac4f17c8c0ca4099.png)
:::

::: {.output .display_data}
![](vertopal_a43c76182a72468ea4b8bf98809c9515/e36b8e1e26987c08ea0b012a47ed8a1bc69add34.png)
:::

::: {.output .display_data}
![](vertopal_a43c76182a72468ea4b8bf98809c9515/01acf589646146467060df692da5da232cda1667.png)
:::

::: {.output .stream .stdout}

     Showing feature importances for: XGBoost
:::

::: {.output .display_data}
![](vertopal_a43c76182a72468ea4b8bf98809c9515/d57c154cb617b0b9d54b3bba546a2e7fe754dbef.png)
:::

::: {.output .stream .stdout}

     Showing feature importances for: RandomForest
:::

::: {.output .display_data}
![](vertopal_a43c76182a72468ea4b8bf98809c9515/b4abc8f3051fa497bcbc532fce8a3956ce87bc00.png)
:::

::: {.output .stream .stdout}

    False Negatives (Positive but predicted as negative):
                  primaryTitle                                               plot  \
    58574             Alcatraz  America's most infamous maximum security priso...   
    55405     Husband Material  Rumi and Vicky are love birds where Rumi convi...   
    19857      My Summer Story  In this second sequel to A Christmas Story (19...   
    5585    Back from Eternity  A South American plane loaded with an assortme...   
    30473  Ju-Rei: The Uncanny  Japanese school girls die violently after seei...   

                           genres                        country         language  \
    58574  Action,Adventure,Crime  United Kingdom, United States          English   
    55405    Comedy,Drama,Romance                          India            Hindi   
    19857           Comedy,Family                  United States          English   
    5585          Adventure,Drama                  United States  Gaelic, English   
    30473                  Horror                          Japan         Japanese   

           true  pred  
    58574     1     0  
    55405     1     0  
    19857     1     0  
    5585      1     0  
    30473     1     0  

    False Positives (Negative but predicted positive):
                                                primaryTitle  \
    29882                                         Opal Dream   
    17780                        The Bonfire of the Vanities   
    36650  I Propose We Never See Each Other Again After ...   
    35847                                    Jennifer's Body   
    5141                              Emil und die Detektive   

                                                        plot  \
    29882  A young Australian girl living in a small rura...   
    17780  Financial "Master of the Universe" Sherman McC...   
    36650  Girl meets Boy. Girl loses Boy. Girl tries out...   
    35847  Nerdy, reserved bookworm Needy Lesnicki, and a...   
    5141   Emil goes to Berlin to see his grandmother wit...   

                         genres                    country          language  \
    29882          Drama,Family  Australia, United Kingdom           English   
    17780  Comedy,Drama,Romance              United States           English   
    36650                Comedy                     Canada           English   
    35847         Comedy,Horror      United States, Canada  English, Spanish   
    5141       Adventure,Family               West Germany            German   

           true  pred  
    29882     0     1  
    17780     0     1  
    36650     0     1  
    35847     0     1  
    5141      0     1  
:::
:::

::: {#ef21b83f-285a-4a66-9b87-521f60960bfa .cell .code}
``` python
```
:::

::: {.output .stream .stdout}

    **① False Negative in Emotion-Focused Dramas**
    Characteristics: These movies, despite receiving positive reviews, were classified negatively
    Analysis: The model seemed insufficient in capturing nuanced emotional content, particularly when relying solely on BERT embeddings
    Cases: Husband Material (India, Hindi) and The Bonfire of the Vanities were misclassified in this manner
    Improvements: Enhance rating distribution representation and adjust genre-based precision
    
    **② False Positive in Commercial Films**
    Characteristics: Commercial films rated negatively were predicted positively
    Analysis: Similarities in metadata like genre and runtime with positively-reviewed films led to misclassification
    Cases: Jennifer’s Body (USA, Horror/Comedy) and The Bonfire of the Vanities were misclassified in this manner
    Improvements: Enhance rating distribution representation and adjust genre-based precision
    
    **③ Underestimation of Non-English Language Films (False Negatives):**
    Characteristics: Non-English or multinational films with positive feedback were classified negatively
    Analysis: Insufficient training data for languages and countries, and cultural nuances were not captured
    Cases: Ju-Rei: The Uncanny (Japan, Japanese) and Back from Eternity had misclassification
    Improvements: Consider oversampling language/country categories and strengthening relevant metadata embeddings
:::

![image](https://github.com/user-attachments/assets/2bf135f5-86fc-4d88-b20c-9daf2760f6b6)
