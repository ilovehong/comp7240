
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
from surprise import SVD, Dataset, Reader
from sklearn.metrics.pairwise import cosine_similarity



class SVDRecommender:
    
        
    def __init__(self):
        
        # Load the datasets separately
        pass

 
    def find_k_nearest_neighbors(self, user_id, svd_bias, k=20000):
        # Convert raw user ID to inner user ID
        user_inner_id = svd_bias.trainset.to_inner_uid(user_id)
        
        # Get the latent features for the user
        user_latent = svd_bias.pu[user_inner_id]
        
        # Compute cosine similarity between this user and all other users
        similarities = cosine_similarity([user_latent], svd_bias.pu)[0]
        
        # Get the top k most similar user indices
        nearest_neighbors_ids = similarities.argsort()[-k:][::-1]

        # Convert inner user ids to raw user ids
        nearest_neighbors_raw_ids = [svd_bias.trainset.to_raw_uid(inner_id) for inner_id in nearest_neighbors_ids]


        return nearest_neighbors_raw_ids


    def get_neighbors_ratings(self, user_id, svd_bias, review_cache):
            # Assuming the existence of a method to find the k nearest neighbors
            neighbors_ids = self.find_k_nearest_neighbors(user_id, svd_bias)

            # Filter the reviews for the neighbors' ratings for the specific item
            neighbors_reviews = review_cache[review_cache['user_id'].isin(neighbors_ids)]
            
            return neighbors_reviews

    
    def recommend(self, user_id=None, review_cache=None):

        unique_reviews = review_cache.drop_duplicates(['user_id', 'business_id'], keep='first').reset_index(drop=True)

        reader = Reader(rating_scale=(1, 5))
        # load the entire dataset into Surprise
        data = Dataset.load_from_df(unique_reviews[['user_id','business_id','stars']], reader)
        svd_bias = SVD(n_factors=10, n_epochs = 20, lr_all=0.005, reg_all=0.05, biased=True) # initiate a SVD algorithm object with the bias terms
        svd_bias.fit(data.build_full_trainset())

        # Extract unique business IDs from the DataFrame
        all_items = unique_reviews['business_id'].unique()

        # Create the testset with the specified user ID and all unique business IDs
        testset = [(user_id, item_id, 0) for item_id in all_items]

        # Predict using the testset
        predictions = svd_bias.test(testset)

        # Converting predictions to a DataFrame
        df_predictions = pd.DataFrame({
            'business_id': [pred.iid for pred in predictions],
            'score_svd': [pred.est for pred in predictions]
        })

        neighbors_reviews = self.get_neighbors_ratings(user_id, svd_bias, review_cache)

        return df_predictions, neighbors_reviews