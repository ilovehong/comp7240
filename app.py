import streamlit as st
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from hybrid_recommender import HybridRecommender
from sklearn.preprocessing import MinMaxScaler

# Configure the page to use a wide layout
st.set_page_config(layout="wide")

# Use st.cache_data to cache the HybridRecommender instance
@st.cache_resource
def get_recommender():
    return HybridRecommender()

recommender = get_recommender()


# Sidebar UI for filters
st.sidebar.title('Tailor Your Experience')

# User switcher in the sidebar, at the top of the left menu
user_selection = st.sidebar.selectbox("Change User Profile:", recommender.get_user_list())

selected_categories = st.sidebar.multiselect(
    'Pick Your Interest(s):', 
    recommender.get_category_list(), 
    default=[]
)

# Algorithm selector in the sidebar, below the categories
selected_algorithms = st.sidebar.multiselect(
    "Choose Scoring Algorithm(s):",
    ["Content Based", "Collaborative SVD", "Collaborative NN"],
    default=["Content Based", "Collaborative SVD", "Collaborative NN"]
)


def get_new_rating():
    new_rating_list = []
    
    for key, rating in st.session_state.items():
        if key.startswith("rating_") and isinstance(rating, int) and rating > 0:
            # Correctly extract business_id without unpacking error
            business_id = key.split('_')[1]
            new_rating_list.append((user_selection, business_id, rating, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    return pd.DataFrame(new_rating_list, columns=['user_id', 'business_id', 'stars', 'date'])


business_df, review_df, svd_explanation_df, cb_explanation_df, nn_explanation = recommender.recommend(user_id=user_selection, categories=selected_categories, new_rating=get_new_rating(), algos=selected_algorithms)

# Tabs for displaying content
recommended_list_tab, my_rating_tab = st.tabs(["Recommended Top 10", "My Rating Timeline"])

with recommended_list_tab:
    for index, row in enumerate(business_df.itertuples(), start=1):
        st.markdown(f"<span style='font-size: 18px;'>{row.name} ({row.stars} stars)</span>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns([1, 2 , 1, 1], gap="small")

        with col1:
            photo_path = os.path.join("static/photos", f"{row.photo_id}.jpg")
            if os.path.exists(photo_path):
                st.image(photo_path, caption=row.name, width=150)
            else:
                st.image("static/photos/no-image.jpg", caption="No image available", width=150)

                            # Adjusted Score
            if row.adjusted_score != 0:
                score = "{:.1f}".format(row.adjusted_score)
                st.markdown(f"<span style='font-size: 14px;'>Adjusted Score: {score}</span>", unsafe_allow_html=True)

            # Weighted Score
            if row.weighted_score != 0:
                weighted_score = "{:.1f}".format(row.weighted_score) if row.weighted_score != 0 else 'NA'
                st.markdown(f"<span style='font-size: 14px;'>Weighted Score: {weighted_score}</span>", unsafe_allow_html=True)

                    # Move the stars slider to the bottom
            unique_key = f"rating_{row.business_id}_{index}"
            user_rating = st.slider("Your rating:", 0, 5, key=unique_key)

        with col2:


            # Content Based Score
            if row.score_cb != 0:
                score_cb = "{:.1f}".format(row.score_cb) if row.score_cb != 0 else 'NA'
                st.markdown(f"<div style='text-align: center;'><span style='font-size: 14px;'>Content Based Score: {score_cb}</span></div>", unsafe_allow_html=True)

            if cb_explanation_df is not None:
                cb_explanation_df_subset = cb_explanation_df[cb_explanation_df['business_id'] == row.business_id]
                cb_explanation_df_subset = cb_explanation_df_subset[['feature', 'strength', 'match']].head(7)
                # Initialize the MinMaxScaler with the desired range
                scaler = MinMaxScaler(feature_range=(1, 10))
                cb_explanation_df_subset[['strength', 'match']] = scaler.fit_transform(cb_explanation_df_subset[['strength', 'match']])


                # Create a long-form DataFrame suitable for Plotly Express
                long_df = cb_explanation_df_subset.melt(id_vars=['feature'], value_vars=['strength', 'match'], var_name='Metric', value_name='Value')
                fig = px.bar(long_df, 
                            y='feature', 
                            x='Value', 
                            color='Metric', 
                            barmode='group',
                            orientation='h',
                            title='Feature Relevance and Preference Comparison')

                # Update the figure layout to remove axis titles and the chart title
                fig.update_layout(
                    xaxis_title='',
                    yaxis_title='',
                    xaxis={'visible': True, 'showticklabels': True},
                    yaxis={'visible': True, 'showticklabels': True},
                    title=''  # Remove the chart title
                )
                # Adjust layout for better readability, if necessary
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with col3:

            # Collaborative SVD Score
            if row.score_svd != 0:
                score_svd = "{:.1f}".format(row.score_svd) if row.score_svd != 0 else 'NA'
                st.markdown(f"<div style='text-align: center;'><span style='font-size: 14px;'>Collaborative SVD Score: {score_svd}</span></div>", unsafe_allow_html=True)



            # Create a placeholder for the plot
            plot_placeholder = st.empty()

            # Histogram for number of user IDs per each star rating per corresponding business ID
            if svd_explanation_df is not None:
                svd_explanation_subset = svd_explanation_df[(svd_explanation_df['business_id'] == row.business_id)]
                if not svd_explanation_subset.empty:
                    # Calculate the width of each bar
                    num_bars = 5  # There are 5 possible star ratings
                    bar_width = 0.8 / num_bars

                    # Create x-axis values (stars) as a list
                    x_values = list(range(1, 6))

                    fig = go.Figure(data=[go.Bar(x=x_values, y=[svd_explanation_subset[svd_explanation_subset['stars'] == i]['user_id'].count() for i in x_values], width=bar_width)])
                    fig.update_layout(
                        title='Neighbors Rating',
                        xaxis_title='Rating',
                        yaxis_title='Number of Neighbors',
                        xaxis=dict(tickmode='array', tickvals=[1, 2, 3, 4, 5]),  # Set tick marks for x-axis
                        width=200,  # Adjust the width of the plot as needed
                        height=300,  # Adjust the height of the plot as needed
                    )
                    plot_placeholder.plotly_chart(fig)
                else:
                    plot_placeholder.text('No data available')


        with col4:

            # Collaborative NN Score
            if row.score_nn != 0:
                score_nn = "{:.1f}".format(row.score_nn) if row.score_nn != 0 else 'NA'
                st.markdown(f"<div style='text-align: center;'><span style='font-size: 14px;'>Collaborative NN Score: {score_nn}</span></div>", unsafe_allow_html=True)

            if nn_explanation is not None:
                st.write("<span style='font-size: 12px;'>This algorithm learns from what users have liked before to suggest new things they might also enjoy.</span>", unsafe_allow_html=True)

        st.write("---")  # Adds a visual separator for each business listing


with my_rating_tab:
    st.header("My Ratings")

    # Assuming 'review_df' contains columns 'photo_id', 'stars', and 'date'
    # and that each 'photo_id' corresponds to a photo file name stored in a directory named "static/photos"
    
    # Calculate the number of rows needed to display 4 businesses per row
    num_reviews = len(review_df)
    num_rows = (num_reviews + 3) // 4  # Integer division rounded up

    for i in range(num_rows):
        # Create a row of columns for each set of 4 businesses
        cols = st.columns(4)
        for j in range(4):
            # Calculate index of the review in review_df
            idx = i*4 + j
            if idx < num_reviews:
                review = review_df.iloc[idx]
                with cols[j]:
                    # Display business photo
                    photo_path = os.path.join("static/photos", f"{review.photo_id}.jpg")
                    if os.path.exists(photo_path):
                        st.image(photo_path, width=150)
                    else:
                        st.image("static/photos/no-image.jpg", caption="No image available", width=150)

                    # First, display the business name
                    st.write(f"**{review.item_name}**")  # Using markdown to bold the name
                    # Then, display review stars and date below the photo
                    st.write(f"Rating: {review.stars}")
                    st.write(f"Date: {review.date}")
