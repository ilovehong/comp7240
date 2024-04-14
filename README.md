# Recommender System Deployment

This GitHub project provides a recommender system implementation that utilizes the Yelp dataset. The system recommends items based on a combination of content-based and collaborative filtering techniques.

To deploy the recommender system, follow the instructions below:

## 1. Download the Yelp Dataset
- Download the Yelp dataset from the official Yelp dataset website (https://www.yelp.com/dataset).
- Extract the downloaded dataset files.
- Copy all the JSON files to the `dataset/jsons` folder in the project directory.
- Copy the photos to the `static/photos` folder in the project directory.

## 2. Prepare Library Files
- Open the Jupyter Notebook files in the following order: `1_data_preprocessing.ipynb`, `2_content_based.ipynb`, `3_collaborative_svd.ipynb`, `4_collaborative_nn.ipynb` and `5_collaborative_knn.ipynb`.
- Run each notebook file one by one to prepare the required library files for the recommender system.
- These files include preprocessed data, trained models, and other necessary artifacts.

## 3. Deploy the App
- Make sure you have Streamlit installed. You can install it using the following command: `pip install streamlit`.
- In the project directory, open the command line or terminal.
- Run the following command: `streamlit run app.py`.
- This will start the Streamlit server and deploy the recommender system app.
- Access the app by opening a web browser and navigating to the URL provided by Streamlit.

## Additional Notes
- The recommender system combines content-based and collaborative filtering techniques to generate recommendations.
- The content-based filtering is implemented in the `2_content_based.ipynb` notebook, while the collaborative filtering is implemented in the `3_collaborative_svd.ipynb`, `4_collaborative_nn.ipynb` and `5_collaborative_knn.ipynb`.notebooks.
- The `1_data_preprocessing.ipynb` notebook is responsible for preprocessing the Yelp dataset and preparing it for further analysis.
- The `app.py` file contains the Streamlit app code that serves as the user interface for the recommender system.

Please feel free to explore the code and modify it according to your requirements. If you encounter any issues or have any questions, don't hesitate to reach out for assistance.

Happy recommending!