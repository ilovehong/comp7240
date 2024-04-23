
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

    def find_max_positive_contributor(self, contributions, labels):
        # Find the index of the maximum positive contribution
        max_index = np.argmax(contributions)
        max_contribution = contributions[max_index] if contributions[max_index] > 0 else None
        if max_contribution:
            return labels[max_index], max_contribution
        else:
            return None, None

    def explain_predictions_batch(self, user_id, item_ids, svd_model):
        try:
            inner_user_id = svd_model.trainset.to_inner_uid(user_id)
            user_factors = svd_model.pu[inner_user_id]
        except KeyError:
            return pd.DataFrame(), "User ID not found in the dataset"

        all_contributions = []
        item_labels = [svd_model.trainset.to_raw_iid(iid) for iid in range(len(svd_model.qi))]

        for item_id in item_ids:
            try:
                inner_item_id = svd_model.trainset.to_inner_iid(item_id)
                item_factors = svd_model.qi[inner_item_id]
                contributions = user_factors * item_factors
                total_base_prediction = np.dot(user_factors, item_factors)
                total_prediction = total_base_prediction + svd_model.bi[inner_item_id] + svd_model.bu[inner_user_id] + svd_model.trainset.global_mean

                max_label, max_contribution = self.find_max_positive_contributor(contributions, item_labels)
                if max_contribution:
                    all_contributions.append({
                        'business_id': item_id,
                        'factor_id': max_label,
                        'contribution': max_contribution,
                        'total': total_prediction
                    })
                else:
                    all_contributions.append({
                        'business_id': item_id,
                        'factor_id': 'No positive contribution',
                        'contribution': 0,
                        'total': total_prediction
                    })
            except KeyError:
                all_contributions.append({
                    'business_id': item_id,
                    'factor_id': 'N/A',
                    'contribution': 'N/A',
                    'total': 'Item ID not found in the dataset'
                })

        return pd.DataFrame(all_contributions)

    def business_to_latent_mapping(self, user_id, svd_bias):
        # Convert user and item matrices to NumPy arrays if they aren't already
        user_latent_matrix = np.array(svd_bias.pu[svd_bias.trainset.to_inner_uid(user_id)])

        print("user_latent_matrix")
        print(user_latent_matrix)
        
        items_latent_matrices = np.array([svd_bias.qi[svd_bias.trainset.to_inner_iid(iid)] for iid in svd_bias.trainset._raw2inner_id_items.keys()])

        print("items_latent_matrices")
        print(items_latent_matrices)

        # Element-wise multiplication and reshaping
        combined_features_array = np.multiply(user_latent_matrix, items_latent_matrices)
    
        print("combined_features_array")
        print(combined_features_array)

        # Create a DataFrame directly from the numpy array
        df_combined_features = pd.DataFrame(combined_features_array, columns=[f'Factor_{i+1}' for i in range(combined_features_array.shape[1])])
        df_combined_features['business_id'] = list(svd_bias.trainset._raw2inner_id_items.keys())

        print("df_combined_features")
        print(df_combined_features)

        return df_combined_features



    def recommend(self, user_id=None, review_cache=None):

        unique_reviews = review_cache.drop_duplicates(['user_id', 'business_id'], keep='first').reset_index(drop=True)

        reader = Reader(rating_scale=(1, 5))
        # load the entire dataset into Surprise
        data = Dataset.load_from_df(unique_reviews[['user_id','business_id','stars']], reader)
        svd_bias = SVD(n_factors=3, n_epochs=40, lr_all=0.005, reg_all=0.1, biased=True) # initiate a SVD algorithm object with the bias terms
        svd_bias.fit(data.build_full_trainset())

        # Extract unique business IDs from the DataFrame
        all_items = unique_reviews['business_id'].unique()

        # Create the testset with the specified user ID and all unique business IDs
        testset = [(user_id, item_id, 0) for item_id in all_items]

        # Predict using the testset
        predictions = svd_bias.test(testset)

        # Converting predictions to a DataFrame
        predictions_df = pd.DataFrame({
            'business_id': [pred.iid for pred in predictions],
            'score_svd': [pred.est for pred in predictions]
        })

        contributions_df = self.explain_predictions_batch(user_id, all_items, svd_bias)

        item_latent_df = self.business_to_latent_mapping(user_id, svd_bias)

        return predictions_df, contributions_df, item_latent_df