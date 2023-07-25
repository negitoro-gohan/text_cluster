import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# データの読み込み
data = pd.read_csv('data.csv')  # データはCSVファイルとして提供されていると仮定

# テキストデータの前処理と特徴量の抽出
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])  # 'text'は文章の列名

# クラスタリングの実行
num_clusters = 2  # クラスタの数（適宜変更してください）
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# クラスタリング結果の確認
clusters = kmeans.labels_

# クラスタリング結果の表示
for i, text in enumerate(data['text']):
    print(f"Text: {text} \t Cluster: {clusters[i]}")
   