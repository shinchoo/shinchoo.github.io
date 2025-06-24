---
layout: single
title:  "Movie Recommendation based on Plot (Unsupervised)"
subtitle: "Recommendation"
categories: [python, Machine Learning]
tag: [Silhouette Scores by K, UMAP + KMeans Cluster, WordCloud]
toc: true
---


## 2020 Forecasting Daily New COVID-19 Cases

::: {#aad356cd-d8df-4c39-a00b-90c48b2cfe59 .cell .code execution_count="1" scrolled="true"}
``` python
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from scipy.sparse import hstack, csr_matrix, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import joblib
import umap

# Suppress warnings
warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden to 1 by setting random_state.*")
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")



def load_data(filepath):
    return pd.read_csv(filepath, encoding="ISO-8859-1")


def multi_hot_encode_with_others(df, column, top_k=None):
    df[column + '_list'] = df[column].fillna('').apply(lambda x: [s.strip() for s in x.split(',') if s.strip()])
    all_items = df[column + '_list'].explode()
    top_items = all_items.value_counts().nlargest(top_k).index if top_k else all_items.unique()
    df[column + '_filtered'] = df[column + '_list'].apply(
        lambda items: [item if item in top_items else 'others' for item in items]
    )
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(df[column + '_filtered'])
    return csr_matrix(encoded), mlb.classes_


def get_tfidf_embeddings(df, max_features=5000,
                         tfidf_embed_file="plot_tfidf_embeddings.npz",
                         tfidf_vectorizer_file="plot_tfidf_vectorizer.pkl"):
    print("Checking TF-IDF embedding...")

    if os.path.exists(tfidf_embed_file) and os.path.exists(tfidf_vectorizer_file):
        plot_embeddings = load_npz(tfidf_embed_file)
        tfidf_vectorizer = joblib.load(tfidf_vectorizer_file)
        print("Loaded cached TF-IDF embeddings.")
    else:
        tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        plot_texts = df["plot"].fillna("")
        plot_embeddings = tfidf_vectorizer.fit_transform(plot_texts)
        save_npz(tfidf_embed_file, plot_embeddings)
        joblib.dump(tfidf_vectorizer, tfidf_vectorizer_file)
        print("TF-IDF embeddings computed and saved.")

    return plot_embeddings, tfidf_vectorizer


def build_feature_matrix(df, plot_embeddings, actor_encoded, director_encoded):
    numeric_features = df[['avg_rating', 'num_rating']].fillna(0)
    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(numeric_features)
    return hstack([plot_embeddings, actor_encoded, director_encoded, csr_matrix(scaled_numeric)])


def reduce_dimensions(X, n_neighbors=15, min_dist=0.1, random_state=42, umap_file="X_umap_3d.npy"):
    if os.path.exists(umap_file):
        print("Loading cached UMAP result...")
        X_umap = np.load(umap_file)
        print("UMAP result loaded.")
    else:
        print("Performing UMAP dimensionality reduction...")
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        X_umap = reducer.fit_transform(X_dense)
        np.save(umap_file, X_umap)
        print(f"UMAP result saved to: {umap_file}")
    return X_umap


def find_best_k(X, k_range=range(3, 9),
                kmeans_cache_file="kmeans_labels_3d.npy",
                silhouette_cache_file="kmeans_silhouette_scores_3d.pkl"):
    if os.path.exists(kmeans_cache_file) and os.path.exists(silhouette_cache_file):
        print("Loading cached KMeans results...")
        best_labels = np.load(kmeans_cache_file)
        k_list, score_list = joblib.load(silhouette_cache_file)
        best_k = k_list[np.argmax(score_list)]
        best_score = max(score_list)
    else:
        print("Evaluating silhouette scores:")
        best_k = None
        best_score = -1
        k_list = []
        score_list = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            score = silhouette_score(X, cluster_labels)

            k_list.append(k)
            score_list.append(score)

            print(f"K={k}: Silhouette Score = {score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = cluster_labels

        np.save(kmeans_cache_file, best_labels)
        joblib.dump((k_list, score_list), silhouette_cache_file)

    return best_k, best_labels, k_list, score_list


def plot_silhouette_scores(k_list, score_list):
    plt.figure(figsize=(8, 5))
    plt.plot(k_list, score_list, marker='o', linestyle='-')
    plt.title("Silhouette Scores by K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.show()



def visualize_clusters(X_umap, labels, k):
    df_vis = pd.DataFrame(X_umap, columns=["UMAP 1", "UMAP 2", "UMAP 3"])
    df_vis["Cluster"] = labels

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        df_vis["UMAP 1"],
        df_vis["UMAP 2"],
        df_vis["UMAP 3"],
        c=df_vis["Cluster"],
        cmap='tab10',
        s=50,
        alpha=0.8
    )

    ax.set_title(f"UMAP + KMeans Best Cluster (K={k})", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")

    norm = mcolors.Normalize(vmin=min(labels), vmax=max(labels))
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap='tab10', norm=norm),
        ax=ax,
        pad=0.15,
        ticks=np.arange(k),
        label='Cluster'
    )

    plt.tight_layout()
    plt.show()


def summarize_clusters(df):
    print("\nCluster Summary: Average Rating / Total Number of Ratings")
    cluster_stats = df.groupby("cluster").agg({
        "avg_rating": "mean",
        "num_rating": "sum"
    }).round(2)
    print(cluster_stats)

    print("\nTop 5 Movies by Number of Ratings per Cluster:")
    for cluster_id in sorted(df["cluster"].unique()):
        print(f"\nCluster {cluster_id}")
        cluster_movies = df[df["cluster"] == cluster_id]
        top5 = cluster_movies.sort_values(by="num_rating", ascending=False).head(5)
        for idx, row in top5.iterrows():
            print(f"- {row['primaryTitle']} (Rating: {row['avg_rating']:.2f}, Num Ratings: {int(row['num_rating'])}, Genres: {row['genres']})")

    print("\nTop 3 Genres per Cluster:")
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_genres = df[df["cluster"] == cluster_id]["genres"].dropna().tolist()
        all_genres = [g.strip() for genres in cluster_genres for g in genres.split(",")]
        top_genres = Counter(all_genres).most_common(3)
        genre_str = ", ".join([f"{g}({c})" for g, c in top_genres]) if top_genres else "None"
        print(f"Cluster {cluster_id} Top Genres: {genre_str}")

    print("\nMost Representative Genre (excluding Drama and Comedy):")
    excluded = {"Drama", "Comedy"}
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_genres = df[df["cluster"] == cluster_id]["genres"].dropna().tolist()
        all_genres = [g.strip() for genres in cluster_genres for g in genres.split(",") if g.strip() not in excluded]
        top_genre = Counter(all_genres).most_common(1)
        genre_str = top_genre[0][0] if top_genre else "None"
        print(f"Cluster {cluster_id} Representative Genre: {genre_str}")

    print("\nTop Directors and Actors per Cluster:")
    for cluster_id in sorted(df["cluster"].unique()):
        print(f"\nCluster {cluster_id}")
        cluster_df = df[df["cluster"] == cluster_id]
        directors = [d.strip() for dir_list in cluster_df["director"].dropna() for d in dir_list.split(",")]
        top_directors = Counter(directors).most_common(3)
        director_str = ", ".join([f"{name}({cnt})" for name, cnt in top_directors]) if top_directors else "No info"
        print(f"Top 3 Directors: {director_str}")
        actors = [a.strip() for actor_list in cluster_df["actors"].dropna() for a in actor_list.split(",")]
        top_actors = Counter(actors).most_common(5)
        actor_str = ", ".join([f"{name}({cnt})" for name, cnt in top_actors]) if top_actors else "No info"
        print(f"Top 5 Actors: {actor_str}")

    print("\nTop 3 Release Years per Cluster:")
    df[['runtimeMinutes', 'startYear']] = df[['runtimeMinutes', 'startYear']].replace('\\N', np.nan)
    df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'], errors='coerce').fillna(df['runtimeMinutes'].median())
    df['startYear'] = pd.to_numeric(df['startYear'], errors='coerce').fillna(df['startYear'].median())
    for cluster_id in sorted(df["cluster"].unique()):
        years = df[df["cluster"] == cluster_id]["startYear"].dropna().astype(int)
        top_years = Counter(years).most_common(3)
        year_str = ", ".join([f"{year} ({count} films)" for year, count in top_years]) if top_years else "No info"
        print(f"Cluster {cluster_id}: {year_str}")

    print("\nTop 3 Languages and Countries per Cluster:")
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == cluster_id]
        languages = [l.strip() for lang in cluster_df["language"].dropna() for l in lang.split(",")]
        top_langs = Counter(languages).most_common(3)
        lang_str = ", ".join([f"{l}({c})" for l, c in top_langs]) if top_langs else "No info"
        countries = [c.strip() for country in cluster_df["country"].dropna() for c in country.split(",")]
        top_countries = Counter(countries).most_common(3)
        country_str = ", ".join([f"{c}({cnt})" for c, cnt in top_countries]) if top_countries else "No info"
        print(f"Cluster {cluster_id} - Languages: {lang_str} | Countries: {country_str}")

    print("\nPlot WordCloud per Cluster:")
    stopwords = set(STOPWORDS)
    stopwords.update(["find", "one", "will", "life", "take", "family", "love", "man", "two", "father", "new", "live", "world", "friend", "become",
                 "young", "help", "meet", "story", "time", "make", "film", "mother", "finds", "girl", "way", "son"])
    for cluster_id in sorted(df["cluster"].unique()):
        texts = df[df["cluster"] == cluster_id]["plot"].fillna("").str.cat(sep=" ")
        wordcloud = WordCloud(width=800, height=400, background_color="white", stopwords=stopwords).generate(texts)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Cluster {cluster_id} Plot WordCloud")
        plt.tight_layout()
        plt.show()

def plot_cluster_feature_distributions(df):
    features = ['avg_rating', 'num_rating']
    for feature in features:
        plt.figure(figsize=(8, 4))
        sns.barplot(data=df, x='cluster', y=feature, estimator=np.mean)
        plt.title(f'Cluster-wise Mean of {feature}')
        plt.ylabel(f'Avg {feature}')
        plt.xlabel('Cluster')
        plt.tight_layout()
        plt.show()


def plot_genre_heatmap(df):
    genre_counts = {}
    for cluster_id in sorted(df["cluster"].unique()):
        genres = df[df["cluster"] == cluster_id]["genres"].dropna()
        genre_list = [g.strip() for gl in genres for g in gl.split(",")]
        count = Counter(genre_list)
        genre_counts[cluster_id] = count

    genre_df = pd.DataFrame(genre_counts).fillna(0).astype(int)
    plt.figure(figsize=(12, 8))
    sns.heatmap(genre_df, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Genre Frequency by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.show()



def main():
    df = load_data("../1. Data_preparation/df_final_frozen_62188.csv")
    # df = load_data("df_final.csv")
    plot_embeddings, _ = get_tfidf_embeddings(df)
    actor_encoded, _ = multi_hot_encode_with_others(df, 'actors', top_k=50)
    director_encoded, _ = multi_hot_encode_with_others(df, 'director', top_k=20)
    X = build_feature_matrix(df, plot_embeddings, actor_encoded, director_encoded)
    X_umap = reduce_dimensions(X)
    best_k, best_labels, k_list, score_list = find_best_k(X_umap)
    plot_silhouette_scores(k_list, score_list)
    df['cluster'] = best_labels
    visualize_clusters(X_umap, best_labels, best_k)
    plot_cluster_feature_distributions(df)    
    plot_genre_heatmap(df)

    summarize_clusters(df)


if __name__ == "__main__":
    main()
```

::: {.output .stream .stdout}
    Checking TF-IDF embedding...
    Loaded cached TF-IDF embeddings.
    Loading cached UMAP result...
    UMAP result loaded.
    Loading cached KMeans results...
:::

::: {.output .display_data}
![](vertopal_859baca405274de1955ea5535e81fcc2/02879ae77b8c1dbdf23869d69471aa46cddb2d09.png)
:::

::: {.output .display_data}
![](vertopal_859baca405274de1955ea5535e81fcc2/788ca995672d65755d8a8b24a560fe5e586ef844.png)
:::

::: {.output .display_data}
![](vertopal_859baca405274de1955ea5535e81fcc2/a3c2edcf4c3204e6c5c0e365304b83800123bcce.png)
:::

::: {.output .display_data}
![](vertopal_859baca405274de1955ea5535e81fcc2/3edd958de028aee63c1c604375475770e9d764f1.png)
:::

::: {.output .display_data}
![](vertopal_859baca405274de1955ea5535e81fcc2/2e802d2c044540a1342e6720cccb432b808747b1.png)
:::

::: {.output .stream .stdout}

    Cluster Summary: Average Rating / Total Number of Ratings
             avg_rating  num_rating
    cluster                        
    0              2.91   3130631.0
    1              4.21   9168236.0
    2              3.66   3997948.0
    3              1.74    383390.0
    4              3.34  14240554.0

    Top 5 Movies by Number of Ratings per Cluster:

    Cluster 0
    - The Devil's Own (Rating: 3.03, Num Ratings: 3786, Genres: Action,Crime,Drama)
    - The Brothers Grimm (Rating: 2.92, Num Ratings: 3744, Genres: Action,Adventure,Comedy)
    - National Lampoon's European Vacation (Rating: 2.98, Num Ratings: 3742, Genres: Adventure,Comedy)
    - Paranormal Activity (Rating: 3.00, Num Ratings: 3735, Genres: Horror,Mystery)
    - Not Another Teen Movie (Rating: 2.66, Num Ratings: 3726, Genres: Comedy)

    Cluster 1
    - The Shawshank Redemption (Rating: 4.40, Num Ratings: 102929, Genres: Drama)
    - Forrest Gump (Rating: 4.05, Num Ratings: 100296, Genres: Drama,Romance)
    - Pulp Fiction (Rating: 4.20, Num Ratings: 98409, Genres: Crime,Drama)
    - The Matrix (Rating: 4.16, Num Ratings: 93808, Genres: Action,Sci-Fi)
    - The Silence of the Lambs (Rating: 4.15, Num Ratings: 90330, Genres: Crime,Drama,Horror)

    Cluster 2
    - Dial M for Murder (Rating: 4.03, Num Ratings: 7252, Genres: Crime,Drama,Mystery)
    - Strangers on a Train (Rating: 4.10, Num Ratings: 6568, Genres: Crime,Drama,Film-Noir)
    - Notorious (Rating: 4.15, Num Ratings: 5962, Genres: Drama,Film-Noir,Mystery)
    - Rebecca (Rating: 4.06, Num Ratings: 5879, Genres: Drama,Mystery,Romance)
    - To Catch a Thief (Rating: 3.97, Num Ratings: 5728, Genres: Drama,Mystery,Romance)

    Cluster 3
    - The Muppet Christmas Carol (Rating: 3.62, Num Ratings: 4208, Genres: Comedy,Drama,Family)
    - The Man Who Would Be King (Rating: 4.00, Num Ratings: 3918, Genres: Adventure,Drama,War)
    - Women on the Verge of a Nervous Breakdown (Rating: 3.89, Num Ratings: 3157, Genres: Comedy,Drama)
    - Police Academy 3: Back in Training (Rating: 2.24, Num Ratings: 3109, Genres: Comedy)
    - Secondhand Lions (Rating: 3.74, Num Ratings: 3077, Genres: Comedy,Drama,Family)

    Cluster 4
    - The Game (Rating: 3.87, Num Ratings: 23310, Genres: Drama,Mystery,Thriller)
    - Deadpool (Rating: 3.85, Num Ratings: 23147, Genres: Action,Comedy,Sci-Fi)
    - The Good, the Bad and the Ugly (Rating: 4.13, Num Ratings: 22922, Genres: Adventure,Western)
    - Citizen Kane (Rating: 4.07, Num Ratings: 22920, Genres: Drama,Mystery)
    - Philadelphia (Rating: 3.80, Num Ratings: 22912, Genres: Drama)

    Top 3 Genres per Cluster:
    Cluster 0 Top Genres: Drama(16280), Comedy(10183), Romance(4852)
    Cluster 1 Top Genres: Drama(963), Comedy(468), Action(279)
    Cluster 2 Top Genres: Drama(11611), Comedy(5557), Romance(3016)
    Cluster 3 Top Genres: Drama(4365), Comedy(3388), Horror(2445)
    Cluster 4 Top Genres: Drama(1277), Comedy(873), Action(621)

    Most Representative Genre (excluding Drama and Comedy):
    Cluster 0 Representative Genre: Romance
    Cluster 1 Representative Genre: Action
    Cluster 2 Representative Genre: Romance
    Cluster 3 Representative Genre: Horror
    Cluster 4 Representative Genre: Action

    Top Directors and Actors per Cluster:

    Cluster 0
    Top 3 Directors: Takashi Miike(42), IshirÃÂ´ Honda(28), Roy Del Ruth(26)
    Top 5 Actors: Amitabh Bachchan(51), Ajay Devgn(38), Henry Fonda(36), Harvey Keitel(33), George Sanders(32)

    Cluster 1
    Top 3 Directors: Raoul Walsh(26), Gordon Douglas(23), Steven Spielberg(10)
    Top 5 Actors: Nicolas Cage(64), James Mason(43), Samuel L. Jackson(41), Bette Davis(40), Akshay Kumar(39)

    Cluster 2
    Top 3 Directors: Jean-Luc Godard(41), Dino Risi(32), Werner Herzog(32)
    Top 5 Actors: John Wayne(82), Bruce Willis(51), Boris Karloff(40), Kirk Douglas(32), Barbara Stanwyck(30)

    Cluster 3
    Top 3 Directors: Richard Thorpe(40), JesÃÂºs Franco(36), Claude Chabrol(33)
    Top 5 Actors: Michael Caine(63), Robert Taylor(39), Robert Mitchum(26), Antonio Banderas(24), Tom Sizemore(24)

    Cluster 4
    Top 3 Directors: Michael Curtiz(63), John Ford(44), Woody Allen(43)
    Top 5 Actors: GÃÂ©rard Depardieu(91), Robert De Niro(71), Jackie Chan(67), Catherine Deneuve(42), Isabelle Huppert(42)

    Top 3 Release Years per Cluster:
    Cluster 0: 2016 (1171 films), 2018 (1160 films), 2017 (1136 films)
    Cluster 1: 2014 (72 films), 2012 (69 films), 2015 (67 films)
    Cluster 2: 2015 (718 films), 2017 (714 films), 2016 (694 films)
    Cluster 3: 2019 (513 films), 2020 (485 films), 2018 (483 films)
    Cluster 4: 1995 (80 films), 1996 (73 films), 1999 (73 films)

    Top 3 Languages and Countries per Cluster:
    Cluster 0 - Languages: English(19526), French(3146), Spanish(2498) | Countries: United States(14360), United Kingdom(3225), France(3080)
    Cluster 1 - Languages: English(1321), French(144), Italian(126) | Countries: United States(1122), United Kingdom(163), France(112)
    Cluster 2 - Languages: English(10037), French(2617), Spanish(1498) | Countries: United States(6488), France(2599), United Kingdom(1900)
    Cluster 3 - Languages: English(7388), Spanish(654), French(633) | Countries: United States(5548), United Kingdom(932), Italy(723)
    Cluster 4 - Languages: English(2148), French(451), Spanish(292) | Countries: United States(1973), United Kingdom(384), France(313)

    Plot WordCloud per Cluster:
:::

::: {.output .display_data}
![](vertopal_859baca405274de1955ea5535e81fcc2/8273ebf541b68f4056f8d3b32cfdb6281d320bff.png)
:::

::: {.output .display_data}
![](vertopal_859baca405274de1955ea5535e81fcc2/3e1b18631ff209b58aeef522b5ed8962f0d15bbc.png)
:::

::: {.output .display_data}
![](vertopal_859baca405274de1955ea5535e81fcc2/d39ab8acf8f4ae4b2f680ffc04996dec26574d7f.png)
:::

::: {.output .display_data}
![](vertopal_859baca405274de1955ea5535e81fcc2/3e7b5e23d979ed03c757d2d745c7e06eec77f6ff.png)
:::

::: {.output .display_data}
![](vertopal_859baca405274de1955ea5535e81fcc2/0d877a439e38b84ec232b61545f75e052c204ed8.png)
:::
:::

::: {#3156a423-fc7a-4b77-bf2c-530a24f00054 .cell .code}
``` python
```
:::

::: {#d5403572-8db1-4f9c-be42-cdaec846cd78 .cell .code}
``` python
```
:::

::: {#df242fc6-330c-45b8-a6f7-d67b9e9510b1 .cell .code}
``` python
```
:::

::: {.output .stream .stdout}

    **Cluster 0: Diverse and Intense Genre Mix**
    Average Rating: 2.91
    Number of Reviews: 3,130,631
    Top Genres: Action, Comedy, Drama
    Representative Movies: The Devil’s Own, National Lampoon’s European Vacation, Paranormal Activity
    Most Frequent Keywords: murder, run, death, escape
    Characteristics: This cluster encapsulates films with a blend of various genres, often characterized by intense plot elements. These movies leverage diverse genre components to captivate audiences
    
    **Cluster 1: Impactful Dramas and Stories**
    Average Rating: 4.21
    Number of Reviews: 9,168,236
    Top Genres: Drama, Crime, Romance
    Representative Movies: The Shawshank Redemption, Forrest Gump, Pulp Fiction
    Most Frequent Keywords: life, love, young, family
    Characteristics: This cluster focuses on deeply moving dramas and storytelling, emphasizing emotional connections and life narratives, which often result in higher ratings.
    
    **Cluster 2: Classic Style Creations**
    Average Rating: 3.66
    Number of Reviews: 3,997,948
    Top Genres: Drama, Mystery, Romance
    Representative Movies: Dial M for Murder, Notorious, Rebecca
    Most Frequent Keywords: life, young, love, old
    Characteristics: Dominated by classic and traditional-style films, this cluster primarily involves mystery and romance, appealing to fans of traditional cinema.
    
    **Cluster 3: Family and Drama Comedies**
    Average Rating: 1.74
    Number of Reviews: 383,390
    Top Genres: Comedy, Drama, Family
    Representative Movies: The Muppet Christmas Carol, Police Academy 3, Secondhand Lions
    Most Frequent Keywords: life, young, family, love
    Characteristics: This cluster centers on family and comedy dramas, emphasizing warm atmospheres and family relationships, with a substantial focus on casualness and humor.
    
    **Cluster 4: Modern Action and Suspense**
    Average Rating: 3.34
    Number of Reviews: 14,240,554
    Top Genres: Action, Comedy, Thriller
    Representative Movies: The Game, Deadpool, The Good, the Bad and the Ugly
    Most Frequent Keywords: life, new, work, young
    Characteristics: Featuring modern and dynamic action and suspense films, this cluster includes popular movies that offer fresh experiences and continuous tension to their audiences.
    Summary of Differences Between Clusters: Cluster 1 vs. Cluster 3: Cluster 1 focuses on storytelling-centric, impactful dramas achieving high ratings frequently, while Cluster 3 encompasses lighter content, including comedy and family dramas.
:::

![image](https://github.com/user-attachments/assets/34e4e022-0ba7-4ccf-af5f-f1194e18bbb1)
