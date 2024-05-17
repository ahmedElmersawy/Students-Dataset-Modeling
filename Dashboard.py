import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import pydeck as pdk
from geopy.geocoders import Nominatim

@st.cache_data
def load_data():
    return pd.read_csv("Students_data.csv")

df = load_data()

st.markdown(
    """
    <style>
    body {
        background-color: #0000000;
    }
    .sidebar .sidebar-content {
        background-color: #34495e;
        color: #ecf0f1;
        padding: 2rem;
    }
    .sidebar .sidebar-content .stRadio label {
        color: #ecf0f1;
    }
    .sidebar .sidebar-content .stSelectbox label {
        color: #ecf0f1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title('Dashboard Settings')
columns = st.sidebar.multiselect('Select Columns to Visualize', df.columns)

for column in columns:
    st.subheader(column)
    if column in ['race', 'class', 'from2', 'from3', 'from4']:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x=column, palette='Set2')
        plt.xticks(rotation=0)
        st.pyplot(plt)
    elif column in ['GPA', 'Algbera', 'Calculus1', 'Calculus2', 'Statistics', 'Probability', 'Measure', 'Functional_analysis']:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x=column, bins=10, color='red')
        st.pyplot(plt)
    elif column == 'from1':
        # Define locations
        locations = {
            "A": "Abu Dhabi",
            "B": "Baghdad",
            "C": "Cairo",
            "D": "New York",
            "E": "Erbil",
            "F": "Faisalabad",
            "G": "Gaza City",
            "H": "Damanhur",
            "I": "Isfahan",
            "J": "Jeddah",
            "K": "Karachi",
            "L": "Luxor",
            "M": "Muscat",
            "N": "Najaf",
            "O": "Oman",
            "P": "Petra",
            "Q": "Qom",
            "R": "Riyadh",
            "S": "Sanaa",
            "T": "Tehran",
            "U": "Umm Qais",
            "V": "Valiasr",
            "W": "Wadi Rum",
            "X": "Xanthos",
            "Y": "Yazd",
            "Z": "Zahedan",
            "a": "Amsterdam",
            "b": "Berlin",
            "c": "Cape Town",
            "d": "Dublin",
            "e": "Eindhoven",
            "f": "Fukuoka",
            "g": "Geneva",
            "h": "Tunis",
            "i": "Istanbul",
            "j": "Jerusalem",
            "k": "Kyoto",
            "l": "Lisbon",
            "m": "Marrakech",
            "n": "Nairobi",
            "o": "Oslo",
            "p": "Prague",
            "q": "Quebec City",
            "r": "Reykjavik",
            "s": "Seville",
            "t": "Taipei",
            "u": "Utrecht",
            "v": "Vienna",
            "w": "Wellington",
            "x": "Xi'an",
            "y": "Yerevan",
            "z": "Zurich"
        }

        # Geolocator setup
        geolocator = Nominatim(user_agent="from1")

        # Geocode each location
        coordinates = {}
        for location, address in locations.items():
            location_coords = geolocator.geocode(address)
            if location_coords:
                coordinates[location] = location_coords
            else:
                st.warning(f"Failed to geocode location: {location}")

        # Create a PyDeck map centered around the mean coordinates
        map_center = [0, 0]
        if coordinates:
            map_center = [sum(coord.latitude for coord in coordinates.values()) / len(coordinates),
                          sum(coord.longitude for coord in coordinates.values()) / len(coordinates)]

        map_data = pd.DataFrame({
            'lat': [coord.latitude for coord in coordinates.values()],
            'lon': [coord.longitude for coord in coordinates.values()],
            'name': list(coordinates.keys())
        })

        view_state = pdk.ViewState(
            latitude=map_center[0],
            longitude=map_center[1],
            zoom=4
        )

        layer = pdk.Layer(
            'ScatterplotLayer',
            data=map_data,
            get_position='[lon, lat]',
            get_radius=10000,
            get_fill_color=[255, 255, 255],
            pickable=True
        )

        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state
        )

        st.subheader("Map with Studens Location")
        st.pydeck_chart(r)

    elif column == 'y':
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x=column, palette='Set1')
        plt.xticks(rotation=0)
        st.pyplot(plt)

    elif 'gender' in columns:
        st.subheader('Pie Chart for Gender')
        gender_counts = df['gender'].value_counts()
        plt.figure(figsize=(8, 6))
        plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue'])
        plt.axis('equal')
        st.pyplot(plt)
