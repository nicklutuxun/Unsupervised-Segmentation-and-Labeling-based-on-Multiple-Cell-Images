from sklearn.cluster import KMeans

def cluster(feature_list, n_clusters=8):
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', max_iter=1000)
    kmeans.fit(feature_list)
    
    return kmeans