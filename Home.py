import streamlit as st

headline = 'Cine Scope'

st.set_page_config(
    page_title=headline,
    page_icon='🎥',
    layout='wide',
    initial_sidebar_state='expanded',
)


st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.pexels.com/photos/109669/pexels-photo-109669.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2');
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
        color: white;  /* Set text color to white */
    }
    h1, h2, h3, h4, h5, h6, p {
        color: white;  /* Set all header and paragraph text to white */
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title(headline)
st.markdown("""
""", unsafe_allow_html=False)

# Adding a brief introduction about the project in bold
st.markdown("""
**Movies play a crucial role in global culture, shaping trends and reflecting societal shifts. The advent of streaming platforms and online ratings has provided a wealth of data about user preferences, movie trends, and audience sentiments. In this project, we aim to leverage the MovieLens dataset to build an interactive visual analytics system that provides insights into movie ratings, user preferences, and trends over time. Through various visualizations, we will uncover patterns in movie ratings, genre popularity, and user behaviors, offering valuable insights for movie enthusiasts, researchers, and industry professionals.**
""", unsafe_allow_html=False)

# TODO: Homepage components
st.markdown("""
**Join us as we decode the secrets of movie success, shaping the future of movie creation and engagement!**
""", unsafe_allow_html=False)