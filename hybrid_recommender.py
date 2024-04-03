
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from content_based_recommender import ContentBasedRecommender
from collaborative_svd_recommender import SVDRecommender
from collaborative_nn_recommender import NNRecommender

class HybridRecommender:


        
    def __init__(self):
        
        # Load the datasets separately
        self.business = self.load_dataset('business.pkl')
        self.user = self.load_dataset('user.pkl')
        self.review = self.load_dataset('review.pkl')
        self.photo = self.load_dataset('photo.pkl')
        self.user_list = self.generate_user_list()
        self.category_list = sorted(list(self.business['categories'].apply(lambda x: x.split(', ')).explode().unique()))
        self.first_photo = self.photo.groupby('business_id', as_index=False).first()[['business_id', 'photo_id']]
        self.content_based_recommender = ContentBasedRecommender()
        self.svd_recommender = SVDRecommender()
        self.nn_recommender = NNRecommender()

    def load_dataset(self, file_name):
        """Load a single dataset from a pickle file."""
        try:
            with open(file_name, 'rb') as file:
                dataset = pickle.load(file)
            print(f"{file_name} loaded OK.")
            return dataset
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
        return None

    def get_user_list(self):
        return self.user_list

    def generate_user_list(self):

        # Calculate review counts, sort in descending order, and take the top 10
        sorted_users = (self.review.groupby('user_id')
                                .size()  # Group by user_id and count
                                .reset_index(name='review_count')  # Convert to DataFrame and name the count column
                                .sort_values(by='review_count', ascending=False)  # Sort by counts in descending order
                                .head(10))  # Take the top 10

        # Generate the current date in yyyymmdd format for the new user's ID
        current_date = datetime.now().strftime("%Y%m%d")

        # Create a DataFrame for the new user
        new_user_df = pd.DataFrame({
            'user_id': [f"{current_date}_new_user_demo"],
            'review_count': [0]  # Assuming 0 reviews for the new user demonstration
        })

        # Concatenate the new user DataFrame with the sorted_users DataFrame
        # Placing the new_user_df at the beginning (top) of the sorted_users DataFrame
        return pd.concat([new_user_df, sorted_users], ignore_index=True)


    def get_category_list(self):
        return self.category_list
    

    def recommend(self, user_id=None, categories=None, new_rating=None, algos=None):
        # Implement your recommendation logic here
        recommend_business = self.business.copy()
        my_review = None
        cb_explanation = None
        svd_explanation = None
        nn_explanation = None
        item_latent_df = None

        # Mapping of algorithms to score keys
        algo_to_score_key = {
            'Content Based': 'score_cb',
            'Collaborative SVD': 'score_svd',
            'Collaborative NN': 'score_nn'
        }

        if user_id is not None:
            review_cache = self.load_dataset('review_cache.pkl')
            new_rating_size = len(new_rating)
            if new_rating_size > 0:
                review_cache = pd.concat([review_cache, new_rating], ignore_index=True)
                review_cache.to_pickle('review_cache.pkl')

            my_review = review_cache[review_cache['user_id'] == user_id].sort_values(by='date', ascending=False)[['user_id','business_id','stars','date']]
            my_review_size = len(my_review)

            if my_review_size > 0:
                is_new_user = my_review_size == new_rating_size
                is_new_data = new_rating_size > 0
                my_review = pd.merge(my_review, recommend_business[['business_id','photo_id','name']], on="business_id", how="left")
                my_review = my_review.rename(columns={'name': 'item_name'})
                my_review['photo_id'] = my_review['photo_id'].fillna('no-image')

                cb_score, cb_explanation = self.content_based_recommender.recommend(user_id=user_id, review_cache=review_cache)
                svd_score, svd_explanation, item_latent_df = self.svd_recommender.recommend(user_id=user_id, review_cache=review_cache)
                nn_score, nn_explanation = self.nn_recommender.recommend(user_id=user_id, review_cache=review_cache, model_rebuild=is_new_user, model_refit=is_new_data)
                # Example of specifying columns to avoid duplicates
                all_score = pd.merge(cb_score[['business_id', 'score_cb']], svd_score[['business_id', 'score_svd']], on="business_id", how="inner")
                hybrid_score = pd.merge(all_score, nn_score[['business_id', 'score_nn']], on="business_id", how="inner")

                # Create a new list based on whether the corresponding algorithm is included
                score_keys = [algo_to_score_key[algo] for algo in algos if algo in algo_to_score_key]
                hybrid_score['weighted_score'] = hybrid_score[score_keys].mean(axis=1)

                reviewed_business_ids = set(my_review.business_id.unique())
                recommend_business = pd.merge(recommend_business, hybrid_score, on="business_id", how="left")
                recommend_business = recommend_business[~recommend_business['business_id'].isin(reviewed_business_ids)]
                if not score_keys:
                    recommend_business['weighted_score'] = recommend_business['adjusted_score']
                recommend_business = recommend_business.sort_values(by="weighted_score", ascending=False)

            else:
                recommend_business['score_cb'] = 0
                recommend_business['score_svd'] = 0
                recommend_business['score_nn'] = 0
                recommend_business['weighted_score'] = recommend_business['adjusted_score']

        if categories:
            recommend_business = recommend_business[recommend_business['categories'].apply(lambda x: any(category in x.split(', ') for category in categories))]

        recommend_business = recommend_business.head(10)
        my_review = my_review.head(10)
        top_10_business_ids = set(recommend_business.business_id.unique())

        if svd_explanation is not None:
            svd_explanation = svd_explanation[svd_explanation['business_id'].isin(top_10_business_ids)]

        if cb_explanation is not None:
            cb_explanation = cb_explanation[cb_explanation['business_id'].isin(top_10_business_ids)]

        if nn_explanation is not None:
            nn_explanation = nn_explanation[nn_explanation['business_id'].isin(top_10_business_ids)]

        if item_latent_df is not None:
            item_latent_df['Rank'] = item_latent_df['business_id'].apply(lambda x: 'High' if x in top_10_business_ids else 'Low')

        return recommend_business, my_review, svd_explanation, cb_explanation, nn_explanation, item_latent_df

