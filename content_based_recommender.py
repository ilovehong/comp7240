
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle

from sklearn.metrics.pairwise import linear_kernel


class ContentBasedRecommender:
    
        
    def __init__(self):
        
        # Load the datasets separately
        self.rest_pcafeature_all = self.load_pickle('rest_pcafeature_all.pkl')
        self.original_feature_names = self.load_pickle('original_feature_names.pkl')

    def load_pickle(self, file_name):
        """Load a single dataset from a pickle file."""
        try:
            with open(file_name, 'rb') as file:
                dataset = pickle.load(file)
            print(f"{file_name} loaded OK.")
            return dataset
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
        return None
        
 
    #------------------------------------------------------------
    # personalized content-based filtering recommender module
    def recommend(self, user_id=None, review_cache=None):
        """Passing of user_id is required if personalized recommendation is desired.
        """
        
        # Merge user ratings with restaurant PCA features
        user_pcafeature = pd.merge(
            review_cache[['user_id', 'business_id', 'stars']],
            self.rest_pcafeature_all,
            how='inner',
            left_on='business_id',
            right_index=True
        ).drop('business_id', axis=1)

        


        # Scale PCA components by user ratings
        for col in user_pcafeature.columns[2:]:  # Skip user_id and stars columns
            user_pcafeature[col] = user_pcafeature[col] * user_pcafeature['stars']

        # Aggregate PCA components by user
        user_pcafeature_all = user_pcafeature.groupby('user_id').sum().drop(columns='stars')

        # Normalize user feature vectors
        user_pcafeature_all = user_pcafeature_all.div(np.linalg.norm(user_pcafeature_all, axis=1), axis=0)


        # predict personalized cosine similarity scores for the user_id of interest
        sim_matrix = linear_kernel(user_pcafeature_all.loc[user_id].values.reshape(1, -1), self.rest_pcafeature_all)
        sim_matrix = sim_matrix.flatten()
        sim_matrix = pd.Series(sim_matrix, index = self.rest_pcafeature_all.index)
        sim_matrix.name = 'similarity_score'
        similarity_score = sim_matrix.reset_index()
        similarity_score.columns = ['business_id', 'score_cb']
        similarity_score['score_cb'] = ((similarity_score['score_cb'] - similarity_score['score_cb'].min()) * (5 - 1) / (similarity_score['score_cb'].max() - similarity_score['score_cb'].min())) + 1

        # Generate explanations DataFrame
        explanations = []
        for idx, row in similarity_score.iterrows():
            business_id = row['business_id']
            score = row['score_cb']
            user_pca_components = user_pcafeature_all.loc[user_id].values
            rest_pca_components = self.rest_pcafeature_all.loc[business_id].values
            
            # Calculate keyword_preference for all features
            keyword_preference = user_pca_components * rest_pca_components
            
            # Sort by keyword_preference to get indices of top preferences
            top_indices = np.argsort(np.abs(keyword_preference))[::-1][:3]
            
            top_keywords = [self.original_feature_names[i][:5] for i in top_indices]
            keyword_preference_sorted = [keyword_preference[i] for i in top_indices]
            keyword_relevance = [user_pca_components[i] for i in top_indices]
            
            feature_list = [kw for kws in top_keywords for kw in kws]
            num_features = len(feature_list)
            
            explanation_data = pd.DataFrame({'business_id': [business_id] * num_features,
                                            'feature': feature_list,
                                            'match': [pref for pref, num_keywords in zip(keyword_preference_sorted, [len(kws) for kws in top_keywords]) for _ in range(num_keywords)],
                                            'strength': [rel for rel, num_keywords in zip(keyword_relevance, [len(kws) for kws in top_keywords]) for _ in range(num_keywords)]})
            explanations.append(explanation_data)

        explanation_df = pd.concat(explanations, axis=0, ignore_index=True)

        return similarity_score, explanation_df