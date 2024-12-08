import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# EX 1

def MinHash_function(movieIds,number_hash_function,p):
    signature=[]
    #set a seed to generate random parameters reproducibles
    random.seed(42)
    #generate a random parameter for each hash function
    list_random_a=random.sample(range(1,1001),number_hash_function)
    #calculate min hash value for each hash function
    for a in list_random_a:
        min=float('inf')
        for ID in movieIds:
            hash_output=(a*ID)%p
            if hash_output<min:
                min=hash_output
        signature.append(min)
    return signature

def create_buckets(user_signatures_vector, number_bands):
    import hashlib
    from collections import defaultdict

    # Set number of bands and calculate number of elements per bands
    user_1=list(user_signatures_vector.keys())[0]
    signature_1=user_signatures_vector[user_1]
    number_element_per_band=len(signature_1)//number_bands

    # Create buckets
    buckets={}
    for userId,signature in user_signatures_vector.items():
        for i in range(number_bands):
            start=i*number_element_per_band  # Calculate each times start and end positions to hashing in bands
            end=start+number_element_per_band
            band=signature[start:end]
            bucket_id=hashlib.md5(str(band).encode()).hexdigest()
            # If bucket doesn't exist inizialized it
            if bucket_id not in buckets:
                buckets[bucket_id]=[]
            buckets[bucket_id].append(userId)

    # Show results for first 5 buckets
    for bucket_id,userId in list(buckets.items())[:5]:
        print(f"Bucket ID: {bucket_id}, Users: {userId}")

def two_most_similar_users(given_user,buckets):
    #save in user_buckets all buckets's ID that contain the given_user
    buckets_ID_shared=[]
    for bucket_id,userId in buckets.items():
      if given_user in userId:
        buckets_ID_shared.append(bucket_id)
    #save in similar_users all users present in the same buckets where the given_user is present
    similar_users=[]
    for bucket_id in buckets_ID_shared:
      for userid in buckets[bucket_id]:
        if userid!=given_user:
            similar_users.append(userid)
    #count how many times each user appears in the buckets shared with the given_user
    user_counts=Counter(similar_users)
    #sort users frequencies in descending order and take the two top similar users to given user
    most_similar=user_counts.most_common(2)
    #maintain in most_similar_users only userIds of the two top similar users
    most_similar_users=[]
    for userId,frequencies in most_similar:
      most_similar_users.append(userId)
    return most_similar_users

def movie_recommendation(user_id,similar_users,ratings,top_rated=5):
    #save in user1 and user2 the Ids of two most similar users
    user1,user2=similar_users[0],similar_users[1]
    #save the list of movieIds and their ratings for user1 and user2
    user1_ratings=ratings[ratings['userId']==user1][['movieId','rating']]
    user2_ratings=ratings[ratings['userId']==user2][['movieId','rating']]
    #inner join between user1_ratings and user2_ratings
    commonly_rated_movies=pd.merge(user1_ratings,user2_ratings,on='movieId')
    if not commonly_rated_movies.empty:
        #if there are observations in common, create 'avg_rating' variable with average rating and mantain only movieId and avg_rating in output dataset sorted in descending order by 'avg_rating'
        commonly_rated_movies['avg_rating']=(commonly_rated_movies['rating_x']+commonly_rated_movies['rating_y'])/2
        recommend_movies=commonly_rated_movies[['movieId', 'avg_rating']].sort_values(by='avg_rating', ascending=False).head(top_rated)
    else:#if there aren't observations in common, maintain movies and rating of user1 sorted in descending order by rating variable
        recommend_movies=user1_ratings.sort_values(by='rating', ascending=False).head(top_rated)
    return recommend_movies

def final_recommendation(user_id, similar_users, ratings, movies, top_n=5):
    #save in user1 and user2 the Ids of two most similar users
    user1, user2=similar_users[0],similar_users[1]

    #save the list of movieIds and their ratings for user1 and user2
    user1_ratings=ratings[ratings['userId'] == user1][['movieId','rating']]
    user2_ratings=ratings[ratings['userId'] == user2][['movieId','rating']]
    #inner join between user1_ratings and user2_ratings
    commonly_rated_movies=pd.merge(user1_ratings,user2_ratings,on='movieId')
    #create 'avg_rating' variable with average rating and sorted dataset in descending order by 'avg_rating'
    commonly_rated_movies['avg_rating']=(commonly_rated_movies['rating_x']+commonly_rated_movies['rating_y'])/2
    commonly_rated_movies=commonly_rated_movies.sort_values(by='avg_rating',ascending=False).head(top_n)
    #select top rated movies for user1 and user2
    top_rated_user_1=user1_ratings.sort_values(by='rating',ascending=False).head(top_n)
    top_rated_user_2=user2_ratings.sort_values(by='rating', ascending=False).head(top_n)

    #start to combine movies in commons and top-rated movies for each users, avoiding duplicates
    recommend_movies=commonly_rated_movies[['movieId','avg_rating']].copy()
    recommend_movies=recommend_movies.rename(columns={'avg_rating':'rating'})
    for top_rated in [top_rated_user_1,top_rated_user_2]:
    #remaining_movies contain only movies_id in top_rated not present in recommend_movies
      remaining_movies=top_rated[~top_rated['movieId'].isin(recommend_movies['movieId'])]
    #calculate the number of movies needed to reach top_n recommendations
      length_remaining=top_n-len(recommend_movies)
    #select the first length_remaining of movies from remaining_movies
      remaining_movies=remaining_movies.head(length_remaining)
    #Concatenate remaining_movies to recommend_movies dataframe
      recommend_movies=pd.concat([recommend_movies,remaining_movies[['movieId','rating']]])
    #Done left join to add movie titles from the movies dataframe to recommend_movies
      recommend_movies=recommend_movies.merge(movies[['movieId','title']],on='movieId',how='left')


    return recommend_movies.head(top_n)




# EX 2 - 3

def kmeans(df, k, max_iter=100, threshold=1e-4, plus = False, seed = 42):

    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    data = df.to_numpy()

    # If plus==True, initialize centroids with KMeans++
    if plus:
        # Centroid initialization with KMeans++

        centroids = np.zeros((k, df.shape[1]))
        # Select randomly the first centroid
        centroids[0] = data[random.randint(0, data.shape[0]-1)]

        for i in  range(1, k):
            # Compute distances from the nearest centroid
            distances = np.array([min(np.sum((point-centroid)**2) for centroid in centroids[:i]) for point in data])

            # Convert distances in probabilities
            probs = distances/np.sum(distances)
            cumulative_probs = np.cumsum(probs)
            rand_val = random.random()


            # If the cumulative probability of a data point is bigger than a certain threshold, the data point becomes a centroid
            for j in range(len(cumulative_probs)):
                if cumulative_probs[j] > rand_val:
                    centroids[i] = data[j]
                    break

    else:
        # Choose k random centroids in the data
        centroids = data[random.sample(range(data.shape[0]), k)]

    # List to store the centroids for each iteration for animation
    centroids_history = [centroids.copy()]
    clusters_history = []

    # Map Phase
    for iter in range(max_iter):
        # Compute the distance from each point to each centroid
        distances = np.array([[np.sum((point - centroid) ** 2) for centroid in centroids] for point in data])

        # Find the closest centroid to each point
        closest_centroids = np.argmin(distances, axis=1)

        # Reduce Phase
        new_centroids = np.zeros_like(centroids)
        for cluster in range(k):
            # Find which data point are in the cluster
            points_in_cluster = data[closest_centroids==cluster]
            if len(points_in_cluster)>0:
                # Compute new centroid
                new_centroids[cluster] = np.mean(points_in_cluster, axis=0)
            else:
                # If there isn't any point assigned to a given centroid, find another random centroid
                new_centroids[cluster] = data[random.randint(0, data.shape[0]-1)]

        # If the centroids remain the same stop the algorithm
        err = np.linalg.norm(new_centroids-centroids, axis=1)
        if np.max(err)<threshold:
            break

        centroids = new_centroids
        centroids_history.append(centroids.copy())
        clusters_history.append(closest_centroids)

    # Calculate distances and re-assign data points to clusters
    distances = np.array([[np.sum((point-centroid)**2) for centroid in centroids] for point in data])
    clusters = np.argmin(distances, axis = 1)

    final_centroids = pd.DataFrame(centroids, columns=df.columns)
    final_centroids['Cluster'] = range(k)
    final_series = pd.Series(clusters, index=df.index, name = 'Cluster')

    # Add cluster column to the initial data frame in place
    df['Cluster'] = final_series.astype(int)

    return final_centroids, df, centroids_history, clusters_history



def elbow(df, max_k = 10, plus=False, seed=42):

    w_sums_squares = []
    for k in range(2, max_k + 1):
        # Run k-means for the current value of k
        centroids, clusters, _, __ = kmeans(df.copy(), k, plus=plus, seed=seed)

        # Compute Within Sum of Squares
        wss = sum(
            np.sum((clusters[clusters['Cluster'] == cluster].drop(columns='Cluster').to_numpy() - centroids.iloc[cluster, :-1].to_numpy())**2)
            for cluster in range(k)
        )
        w_sums_squares.append(wss)

    # Plot the inertia values
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_k + 1), w_sums_squares, marker='o', linestyle='--', color='skyblue')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within Sum of Squares (WSS)')
    plt.title('Elbow Method to Determine Optimal k')
    plt.grid()
    plt.show()


def silhouette(df, max_k=10, plus=False, seed=42):
    # Initialize list to store silhouette scores for different values of k
    silhouette_scores = []

    for k in range(2, max_k + 1):
        # Perform k-means for current k
        centroids, clusters, _, __ = kmeans(df.copy(), k, plus=plus, seed=seed)

        # Get cluster labels
        cluster_labels = clusters['Cluster'].to_numpy()

        # Obtain points
        data = clusters.drop(columns=['Cluster']).to_numpy()
        n_points = data.shape[0]

        # Initialize silhouette score for all points
        s_values = np.zeros(n_points)

        for i in range(n_points):
            # Current data point cluster
            current_cluster = cluster_labels[i]

            # Get points from the same cluster
            same_cluster_mask = cluster_labels == current_cluster
            same_cluster_points = data[same_cluster_mask]

            # Get points from the other clusters
            other_clusters_points = data[~same_cluster_mask]
            other_cluster_labels = cluster_labels[~same_cluster_mask]

            # Compute mean within-cluster distance
            if len(same_cluster_points) > 1:
                a_i = np.mean(cdist([data[i]], same_cluster_points)[0][1:])
            else:
                a_i = 0

            # Compute mean distance from nearest cluster points
            b_i = np.inf
            for cluster in np.unique(cluster_labels):
                if cluster != current_cluster:
                    cluster_points = data[cluster_labels == cluster]
                    b_i = min(b_i, np.mean(cdist([data[i]], cluster_points)))

            # Compute Silhouette for point i
            s_values[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0

        # Mean Silhouette for k clusters
        silhouette_scores.append(np.mean(s_values))

    # Plot Silhouette Score
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o', linestyle='--', color='orange')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method to Determine Optimal k')
    plt.grid()
    plt.show()


def plot_kmeans_animation(df, k, centroids_history, clusters_history):
    # Create Frame
    fig = make_subplots(rows=1, cols=1)

    # Colors for clusters
    colors = ['#57C4E5', '#F97068', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Add initial data points (in grey before assignment)
    scatter_data = go.Scatter(
        x=df['PC1'], y=df['PC2'], mode='markers',
        marker=dict(color='gray', size=8),
        name="Data points"
    )
    fig.add_trace(scatter_data)
    # Create centroids' traces
    centroid_traces = []
    for j in range(k):
        centroid_traces.append(go.Scatter(
            x=[], y=[], mode='markers+text', text=[f"Centroid {j+1}"], textposition='top center',
            marker=dict(color='black', symbol='x', size=12, opacity=1),
            name=f"Centroid {j+1}",
            showlegend=True
        ))
        fig.add_trace(centroid_traces[-1])

    # Prepare frames
    frames = []

    # Iterate through centroids positions
    for i in range(len(centroids_history)):
        frame_data = []

        # Add points to cluster
        scatter_data = go.Scatter(
            x=df['PC1'], y=df['PC2'], mode='markers',
            marker=dict(color=[colors[c] for c in clusters_history[i]], size=8),
            name="Data points"
        )
        frame_data.append(scatter_data)

        # Update centroids positions
        for j in range(k):
            centroid_traces[j].x = [centroids_history[i][j][0]] 
            centroid_traces[j].y = [centroids_history[i][j][1]] 
            centroid_traces[j].text = [f"Centroid {j+1} (Iter {i+1})"]

        # Add frame for current iteration
        frames.append(go.Frame(data=frame_data, name=f"Iteration {i+1}"))

    fig.frames = frames
    # Layout
    fig.update_layout(
        title="K-Means Clustering Animation",
        xaxis_title="PC1",
        yaxis_title="PC2",
        updatemenus=[dict(
            type="buttons",
            x=0.1, y=-0.1,
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, dict(frame=dict(duration=1000, redraw=True), fromcurrent=True)]
            )]
        )]
    )

    fig.show()