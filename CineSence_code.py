import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CINESENCE - The Recommender", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_csv(filename):
    return pd.read_csv(os.path.join(BASE_DIR, filename))

@st.cache_data
def load_movies():
    return load_csv("movies.csv")

movies = load_movies()

movies["genres_clean"] = movies["genres"].str.replace("|", " ", regex=False)
movies["actors_clean"] = movies["actors"].str.replace("|", " ", regex=False)

def build_similarity(text_column):
    cv = CountVectorizer(stop_words="english")
    matrix = cv.fit_transform(text_column)
    return cosine_similarity(matrix)

title_sim = build_similarity(movies["title"])
genre_sim = build_similarity(movies["genres_clean"])
actor_sim = build_similarity(movies["actors_clean"])

def recommend(movie_title, sim_matrix):
    if movie_title not in movies["title"].values:
        return pd.DataFrame()
    
    idx = movies[movies["title"] == movie_title].index[0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    result = []
    number = 1

    for i, score in scores:
        result.append({
            "Title": movies.iloc[i]["title"],
            "Genres": movies.iloc[i]["genres"],
            "Actors": movies.iloc[i]["actors"],
            "OTT Platform": movies.iloc[i]["ott"]
        })
        number += 1

    return pd.DataFrame(result)

st.title("���� CINESENCE - The Movie Recommender")

tabs = st.tabs(["Title-Based", "Genre-Based", "Actor-Based"])

with tabs[0]:
    st.header("���� Title-Based Recommendation")
    movie = st.selectbox("Choose a movie:", movies["title"].tolist())
    if st.button("Recommend", key="title_btn"):
        df = recommend(movie, title_sim)
        st.subheader(f"Movies similar to: {movie}")
        st.dataframe(df, use_container_width=True)

with tabs[1]:
    st.header("���� Genre-Based Recommendation")
    movie = st.selectbox("Choose a movie (Genres):", movies["title"].tolist(), key="genre_movie")
    if st.button("Recommend by Genre"):
        df = recommend(movie, genre_sim)
        st.subheader(f"Movies with similar genres to: {movie}")
        st.dataframe(df, use_container_width=True)

with tabs[2]:
    st.header("���� Actor-Based Recommendation")
    movie = st.selectbox("Choose a movie (Actors):", movies["title"].tolist(), key="actor_movie")
    if st.button("Recommend by Actors"):
        df = recommend(movie, actor_sim)
        st.subheader(f"Movies with similar actors to: {movie}")
        st.dataframe(df, use_container_width=True)

st.markdown("---")
st.caption("By CineSence")

