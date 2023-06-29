import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from utils import trainingFile_preprocess,combine,remove_tabs_newlines
import fasttext
import numpy as np
from tqdm import tqdm
import faiss
import json
from sklearn.cluster import DBSCAN
import chardet
# with open('Data_Models/cri_head.csv', 'rb') as f:
#     result = chardet.detect(f.read())
# print(result)
def draw():
    svd = TruncatedSVD(n_components=3)
    reduced_features = svd.fit_transform(tfidf_matrix)
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2], c=labels)
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    ax.set_title('Cluster Visualization')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.view_init(elev=30, azim=135)

    plt.show()

k=6
df=pd.read_csv('Data_Models/cri.csv',encoding='utf-8',nrows=5000)
df=df[['IncidentId','CreateDate','Title','Mitigation','Summary']]
df.dropna(subset=['Mitigation'],inplace=True)
df['label']=''
corpus = df['Mitigation'].tolist()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

kmeans = KMeans(n_clusters=k)
kmeans.fit(tfidf_matrix)
labels=kmeans.labels_
# dbscan = DBSCAN(eps=0.6, min_samples=50)
# dbscan.fit(tfidf_matrix)
# labels = dbscan.labels_
df['label'] = labels
# draw()
df['CreateDate']=pd.to_datetime(df['CreateDate'])
df_sorted = df.sort_values('CreateDate')
df_train=df_sorted[:4853]
df_test=df_sorted[4853:]
df_train=df_train.reset_index(drop=True)
df_test=df_test.reset_index(drop=True)
target='label'
column_list=['Title','Summary']
df_train['metadata']=df_train.to_dict(orient='records')
df_test['metadata']=df_test.to_dict(orient='records')
file = trainingFile_preprocess(df_train,column_list,target)
np.savetxt('train_tem.txt', file, fmt='%s',encoding='utf-8')
model = fasttext.train_supervised(input='train_tem.txt', epoch=100, lr=0.9, thread=7, wordNgrams=3,seed=41)
dict={}
df_train_text=df_train['metadata'].apply(lambda x: combine(x,column_list))
df_train_text= df_train_text.apply(remove_tabs_newlines)
vectors = np.array([model.get_sentence_vector(text) for text in df_train_text])
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)
alpha=0.001
num=3
for i in tqdm(range(len(df_test))):
    def decay_factor(alpha, time_diff):
        return np.exp(-alpha * time_diff)

    def similarity_with_time_decay(x, length, created_time, alpha, num):
        # distance
        distances, indices = index.search(x, k=length)
        distances = distances.flatten()
        sorted_indices = np.argsort(indices)
        sorted_distances = distances[sorted_indices]
        # time_diff
        time_diffs = (created_time - df_train['CreateDate']).values / np.timedelta64(1, 'D')
        # decay
        decay_factors = decay_factor(alpha, time_diffs)
        # similarity
        similarities = 1 / (1 + sorted_distances) * decay_factors
        # find k similar
        sorted_indices = np.argsort(similarities[0])[::-1]
        similar_scores = similarities[0][sorted_indices[:num]]
        return df_train.iloc[sorted_indices[:num]],similar_scores
    created_time=df_test['CreateDate'][i]
    query = combine(df_test['metadata'][i], column_list)
    query = remove_tabs_newlines(query)
    query_vector = np.array([model.get_sentence_vector(query)])
    train_samples,similar_scores = similarity_with_time_decay(query_vector, len(df_train), created_time, alpha, num)
    to_add=[]
    j=0
    for _, train_sample in train_samples.iterrows():
        tmp={}
        tmp['SimilarScore']=similar_scores[j]
        tmp['SimilarID']=train_sample['IncidentId']
        to_add.append(tmp)
        j+=1
    dict[df_test['IncidentId'][i]]=to_add
    row_to_add = df_test.iloc[i]
    df_train = pd.concat([df_train, row_to_add.to_frame().T], axis=0, ignore_index=True)
    index.add(query_vector)
print(dict)

with open('output.json', 'w') as f:
    json.dump(dict, f)

