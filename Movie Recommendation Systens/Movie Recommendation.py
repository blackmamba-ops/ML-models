import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets
movies = pd.read_csv("Movies.csv")
additional_data = pd.read_csv("Movie AdditionalData.csv")

# Merge the datasets based on the common movie_id column
movies = pd.merge(movies, additional_data, left_on='movie_id', right_on='id', how='left')
movies.set_index('movie_id', inplace=True)
movies.drop(columns=['title_y'], inplace=True)

# Combine relevant text features into a single feature 'content'
movies["content"] = movies["genres"] + " " + movies["keywords"] + " " + movies["cast"] + " " + movies["crew"]

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Compute the TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(movies["content"])

# Reset index to ensure movie indices match with the cosine similarity matrix
movies.reset_index(inplace=True)

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations
def get_recommendations(movie_title, cosine_sim=cosine_sim, movies=movies):
    matching_movies = movies[movies["title_x"].str.lower() == movie_title.lower()]
    if matching_movies.empty:
        print(f"No movie found with the title '{movie_title}'.")
        return None

    input_movie_idx = matching_movies.index[0]

    # Create a dictionary to map movie titles to their indices in cosine_sim array
    movie_indices = {movie_title.lower(): idx for idx, movie_title in enumerate(movies['title_x'].str.lower())}

    idx = movie_indices[movie_title.lower()]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top 10 similar movies (excluding the input movie itself)
    movie_indices = [x[0] for x in sim_scores if x[0] != input_movie_idx][:10]

    return movies[["title_x", "homepage"]].iloc[movie_indices]

# User input for movie title
user_input = input("Enter a movie title: ")
recommendations = get_recommendations(user_input)
if recommendations is not None:
    print(f"Recommended movies for '{user_input}':")
    print(recommendations)

