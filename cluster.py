from sklearn.cluster import KMeans, SpectralClustering

def kmeans(feature_list, n_clusters=11):
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', max_iter=1000)
    kmeans.fit(feature_list)
    
    return kmeans

def sectral(feature_list, n_clusters=11):
    model = SpectralClustering(n_clusters=n_clusters)
    model.fit(feature_list)

    return model
      