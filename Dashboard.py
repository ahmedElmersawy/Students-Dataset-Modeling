import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

@st.cache_data
def load_data():
    return pd.read_csv("Students_data.csv")

# Load and preprocess data
df = load_data()

# Preprocess the data
labelencoder = LabelEncoder()
for feature in ['from1', 'from2', 'from3', 'from4']:
    df[feature] = labelencoder.fit_transform(df[feature])

features = ['from1', 'from2', 'from3', 'from4']
X = df[features]
y = df['y']

# Normalize the data
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Instantiate and fit the model
logReg = LogisticRegression()
logReg.fit(X_train, y_train)

# Make predictions
y_pred = logReg.predict(X_test)

# Calculate accuracy and MSE
accuracy = metrics.accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Streamlit dashboard layout
st.markdown(
    """
    <style>
    body {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #343a40;
        color: #ffffff;
    }
    .sidebar .sidebar-content .stRadio label {
        color: #ffffff;
    }
    .sidebar .sidebar-content .stSelectbox label {
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title('Dashboard Settings')
columns = st.sidebar.multiselect('Select Columns to Visualize', df.columns)
show_regression = st.sidebar.checkbox('Show Regression Results and Plots')
predict_new_data = st.sidebar.checkbox('Predict New Data')

# Visualization section
st.title('Student Data Dashboard')

for column in columns:
    st.subheader(column)
    if column in ['race', 'class', 'from2', 'from3', 'from4']:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x=column, palette='Set2')
        plt.xticks(rotation=0)
        st.pyplot(plt)
    elif column in ['GPA', 'Algebra', 'Calculus1', 'Calculus2', 'Statistics', 'Probability', 'Measure', 'Functional_analysis']:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x=column, bins=10, color='red')
        plt.xlabel('Grades')
        plt.ylabel('Frequency')
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

        # Geocode each location with error handling
        coordinates = {}
        for location, address in locations.items():
            try:
                location_coords = geolocator.geocode(address)
                if location_coords:
                    coordinates[location] = location_coords
                else:
                    st.warning(f"Failed to geocode location: {location}")
            except GeocoderUnavailable:
                st.error(f"Geocoding service is unavailable for location: {location}")
                break

        if coordinates:
            # Create a PyDeck map centered around the mean coordinates
            map_center = [
                sum(coord.latitude for coord in coordinates.values()) / len(coordinates),
                sum(coord.longitude for coord in coordinates.values()) / len(coordinates)
            ]

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
                get_fill_color=[255, 0, 0],
                pickable=True
            )

            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state
            )

            st.subheader("Map with Students Location")
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

# Regression Results section
if show_regression:
    st.sidebar.subheader('Regression Results')
    st.sidebar.write(f'Accuracy: {accuracy*100:.2f}%')
    st.sidebar.write(f'Mean Squared Error: {mse:.2f}')

    # Plot regression results
    for feature in features:
        fig, ax = plt.subplots()
        ax.scatter(X_test[feature], y_test, color='blue', label='Actual')
        ax.scatter(X_test[feature], y_pred, color='red', label='Predicted')
        ax.set_title(f'{feature} vs y')
        ax.set_xlabel(feature)
        ax.set_ylabel('y')
        ax.legend()
        st.pyplot(fig)

# Predict New Data section
if predict_new_data:
    st.sidebar.subheader('Predict New Data')
    new_data = {}
    for feature in features:
        value = st.sidebar.number_input(f'Enter {feature}', min_value=float(df[feature].min()), max_value=float(df[feature].max()), step=0.01)
        new_data[feature] = value

    if st.sidebar.button('Predict'):
        new_data_df = pd.DataFrame([new_data])
        new_data_scaled = pd.DataFrame(scaler.transform(new_data_df), columns=new_data_df.columns)
        new_prediction = logReg.predict(new_data_scaled)
        st.sidebar.write('Predicted result:', new_prediction[0])
