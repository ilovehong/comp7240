
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
from surprise import SVD, Dataset, Reader
from sklearn.metrics.pairwise import cosine_similarity
from surprise import KNNWithMeans


class KNNRecommender:
    
        
    def __init__(self):
        
        # Load the datasets separately
        pass

 
    def get_neighbors_ratings(self, knn_model, business):
        all_neighbors_ratings_list = []
        for item_id in business:
            try:
                inner_item_id = knn_model.trainset.to_inner_iid(item_id)
            except ValueError:
                self.logger.warning(f"Item ID {item_id} not found in trainset.")
                continue

            # Attempt to retrieve exactly k neighbors
            neighbors = knn_model.get_neighbors(inner_item_id, k=knn_model.k)
            if not neighbors:  # Fallback if no neighbors found
                # Try finding any available neighbors
                available_neighbors = len(knn_model.trainset.ir[inner_item_id])
                neighbors = knn_model.get_neighbors(inner_item_id, k=available_neighbors)

            for neighbor in neighbors:
                neighbor_raw_id = knn_model.trainset.to_raw_iid(neighbor)
                ratings = knn_model.trainset.ur[neighbor]
                for rating in ratings:
                    if rating[0] == inner_item_id:
                        neighbor_rating_df = pd.DataFrame({
                            'user_id': [neighbor_raw_id],
                            'business_id': [item_id],
                            'rating': [rating[1]]
                        })
                        all_neighbors_ratings_list.append(neighbor_rating_df)

        return pd.concat(all_neighbors_ratings_list, ignore_index=True) if all_neighbors_ratings_list else pd.DataFrame()


    def recommend(self, user_id=None, review_cache=None):

        unique_reviews = review_cache.drop_duplicates(['user_id', 'business_id'], keep='first').reset_index(drop=True)

        reader = Reader(rating_scale=(1, 5))
        # load the entire dataset into Surprise
        data = Dataset.load_from_df(unique_reviews[['user_id','business_id','stars']], reader)

        knn = KNNWithMeans(k=80, sim_options={
            "name": "cosine",
            "user_based": False,  
            })
        knn.fit(data.build_full_trainset())
       
        # Extract unique business IDs from the DataFrame
        all_items = unique_reviews['business_id'].unique()

        # Create the testset with the specified user ID and all unique business IDs
        testset = [(user_id, item_id, 0) for item_id in all_items]

        # Predict using the testset
        predictions = knn.test(testset)

        # Converting predictions to a DataFrame
        predictions_df = pd.DataFrame({
            'business_id': [pred.iid for pred in predictions],
            'score_knn': [pred.est for pred in predictions]
        })

        neighbors_ratings_df = self.get_neighbors_ratings(knn, all_items)

        return predictions_df, neighbors_ratings_df