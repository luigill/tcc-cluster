import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Coleta de dados
data = {
    "words": [
        "cachorro",
        "gaxorro",
        "cachoro",
        "gachorro",
        "relógio",
        "relogio",
        "relógio",
        "relógia",
    ]
}

# Criação do DataFrame
df = pd.DataFrame(data)

# Pré-processamento
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
X = vectorizer.fit_transform(df["words"])

# Clusterização
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)

# Avaliação
labels = kmeans.labels_
silhouette_avg = silhouette_score(X, labels)

print(f"Silhouette Score: {silhouette_avg}")
print(f"Labels: {labels}")

# Adicionando os labels ao DataFrame
df["cluster"] = labels

print(df)
