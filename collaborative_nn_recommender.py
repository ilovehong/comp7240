
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
import shap

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Flatten, Input, Dot, Concatenate, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



class NNRecommender:
            
    def __init__(self):
        
        # Load the datasets separately
        pass

    def create_collab_model(self, num_businesses, num_users):
        embedding_dim=32

        user_input = Input(shape=(1,), name='user_input')
        business_input = Input(shape=(1,), name='business_input')

        user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, embeddings_regularizer=l2(1e-6))(user_input)
        business_embedding = Embedding(input_dim=num_businesses, output_dim=embedding_dim, embeddings_regularizer=l2(1e-6))(business_input)

        user_flatten = Flatten()(user_embedding)
        business_flatten = Flatten()(business_embedding)

        merged = Concatenate()([user_flatten, business_flatten])
        merged = BatchNormalization()(merged)

        dense_layer = Dense(128, activation='relu')(merged)
        dropout = Dropout(0.4)(dense_layer)
        output_layer = Dense(1, activation='linear')(dropout)

        model = Model(inputs=[user_input, business_input], outputs=output_layer)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

        return model

    def load_encoders(self):
        with open('model/business_encoder.pickle', 'rb') as b:
            business_encoder = pickle.load(b)

        with open('model/user_encoder.pickle', 'rb') as b:
            user_encoder = pickle.load(b)
        
        return business_encoder, user_encoder, len(business_encoder.classes_), len(user_encoder.classes_)


    def create_dict(self, business_encoder, user_encoder):
        business2idx = dict(zip(business_encoder.classes_, business_encoder.transform(business_encoder.classes_)))
        idx2business = dict(zip(business_encoder.transform(business_encoder.classes_), business_encoder.classes_))

        user2idx = dict(zip(user_encoder.classes_, user_encoder.transform(user_encoder.classes_)))
        idx2user = dict(zip(user_encoder.transform(user_encoder.classes_), user_encoder.classes_))

        return business2idx, idx2business, user2idx, idx2user


    def load_data(self, user_id=None, review_cache=None, model_rebuild=False, model_refit=False):

        batch_size = 256
        epochs = 20

        if model_rebuild:
            user_encoder = LabelEncoder()
            business_encoder = LabelEncoder()
            review_cache['user_id_encoded'] = user_encoder.fit_transform(review_cache['user_id'])
            review_cache['business_id_encoded'] = business_encoder.fit_transform(review_cache['business_id'])
            num_users = len(user_encoder.classes_)
            num_businesses = len(business_encoder.classes_)

            business2idx, idx2business, user2idx, idx2user = self.create_dict(business_encoder, user_encoder)
            collab_model = self.create_collab_model(num_businesses, num_users)

            user_ids = review_cache['user_id_encoded'].values
            business_ids = review_cache['business_id_encoded'].values
            stars = review_cache['stars'].values

            model_checkpoint = ModelCheckpoint(f'./model/model.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min')
            early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
                
            history = collab_model.fit([user_ids, business_ids], stars,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.2,
                callbacks=[early_stopping, model_checkpoint]
            )            

            with open('./model/user_encoder.pickle', 'wb') as f:
                pickle.dump(user_encoder, f)
            with open('./model/business_encoder.pickle', 'wb') as f:
                pickle.dump(business_encoder, f)
            
        else:
            business_encoder, user_encoder, num_businesses, num_users = self.load_encoders()

            business2idx, idx2business, user2idx, idx2user = self.create_dict(business_encoder, user_encoder)
            collab_model = self.create_collab_model(num_businesses, num_users)
            collab_model.load_weights('model/model.weights.h5')
            
            if model_refit:
                my_review = review_cache[review_cache['user_id'] == user_id].copy()
                my_review['user_id_encoded'] = user_encoder.fit_transform(my_review['user_id'])
                my_review['business_id_encoded'] = business_encoder.fit_transform(my_review['business_id'])

                user_ids = my_review['user_id_encoded'].values
                business_ids = my_review['business_id_encoded'].values
                stars = my_review['stars'].values

                model_checkpoint = ModelCheckpoint(f'./model/model.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min')
                early_stopping = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)

                history = collab_model.fit([user_ids, business_ids], stars,
                                        batch_size=batch_size,
                                        epochs=4,
                                        callbacks=[early_stopping, model_checkpoint]
                                    )     

        return business2idx, idx2business, user2idx, collab_model


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
        
 
    def recommend(self, user_id=None, review_cache=None, model_rebuild=False, model_refit=False):

        business2idx, idx2business, user2idx, model = self.load_data(user_id, review_cache, model_rebuild, model_refit)
        useridx = user2idx[user_id]
        scores = {}
        


        user_indices = np.array([useridx] * len(idx2business))
        business_indices = np.array(list(range(len(idx2business))))

        # Prepare arrays for all businesses
        user_indices = np.array([useridx] * len(idx2business))
        business_indices = np.array(list(range(len(idx2business))))

        # Predict ratings for all businesses in one batch
        predictions = model.predict([user_indices, business_indices]).flatten()

        # Create a dictionary of business_id to predicted score
        scores = {business: score for business, score in zip(idx2business.values(), predictions)}

        # Sort the businesses by predicted scores in descending order
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        # Convert to DataFrame
        scores_df = pd.DataFrame(sorted_scores, columns=['business_id', 'score_nn'])

        # # Convert to DataFrame and take top 10
        # top_scores_df = scores_df.head(10).copy()

        # # Extract business indices corresponding to the top 20 business IDs
        # top_business_indices = [business2idx.get(business_id) for business_id in top_scores_df['business_id'].values]

        # # Convert the list of business indices to a NumPy array
        # top_business_indices_array = np.array(top_business_indices)

        # # Concatenate user and business indices for SHAP analysis
        # data = np.column_stack((np.array([useridx] * len(top_business_indices)), top_business_indices_array))

        # # Prepare SHAP explainer and compute values
        # # Adjust the lambda function based on your model's expectation
        # explainer = shap.KernelExplainer(model=lambda x: model.predict([x[:, 0], x[:, 1]]), data=data)
        # shap_values = explainer.shap_values(data)

        # # Aggregate SHAP values and create explanations
        # shap_sum = np.abs(shap_values).sum(axis=1)
        # shap_sum = np.array(shap_sum).flatten()

        # explanations_df = pd.DataFrame({
        #     'business_id': top_scores_df['business_id'].values,
        #     'shap_sum': shap_sum
        # })

        return scores_df, None