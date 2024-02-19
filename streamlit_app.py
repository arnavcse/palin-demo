import numpy as np
import streamlit as st

# Load your image files
demo_image_path = "demo.jpeg"
gif_image_path = "1.gif"

# Display demo.png image
st.image(demo_image_path, use_column_width=True)

# Load your neural network weights and biases
weights_input_hidden = np.array([[-10.2294859, 8.09507879, -7.76310088, 0.87299146],
                                 [-5.14719342, 4.01541671, -3.79080752, 6.22177305],
                                 [20.57968484, -16.38281642, 15.59209513, -1.18677865],
                                 [2.23084496, -6.2968456, -7.7911219, -0.62522003],
                                 [-18.04541078, 14.19414722, -13.508314, 7.33701767],
                                 [17.92633635, -14.40129001, 13.71816351, 5.96703394],
                                 [-2.26863642, -2.64160222, -8.72014773, 0.46199896],
                                 [-20.56505124, 16.41616763, -15.58385872, 1.61225809],
                                 [5.08204562, -4.10567535, 3.90111721, 6.09747744],
                                 [10.21438964, -8.15958061, 7.69688485, -1.14692891]])
weights_hidden_output = np.array([[-36.67667482],
                                  [-19.05778607],
                                  [19.73534923],
                                  [-2.41522533]])
bias_input_hidden = np.array([[-2.9601965, 7.66022278, 1.2310203, -2.21047831]])
bias_hidden_output = np.array([[10.97669777]])
thresh = 0.949347883045725


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


st.title("**🌲 •Palindrome Classifier•  🌲**")

# Using Markdown for the input text to include an emoji
user_input = st.text_input("• ⭕ **Enter a _10-bit_ binary string (e.g., 1010101010 )**  **★**", "")

# Display the "Classify" button with larger size
classify_button = st.button(" **•⭐ Classify 🧠•** ", key="classify_button", help="Click to classify")

# Adjusting the size of the classify button using CSS
st.markdown(
    """
    <style>
    .stButton>button {
        width: 200px !important;
        height: 50px !important;
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if classify_button:
    binary = np.array(list(user_input), dtype=int)
    input_layer = binary.reshape(1, -1)
    hidden_layer = sigmoid(np.dot(input_layer, weights_input_hidden) + bias_input_hidden)
    output_layer = sigmoid(np.dot(hidden_layer, weights_hidden_output) + bias_hidden_output)
    pred = output_layer[0]
    if pred > thresh:
        st.write("⤷ ✅ **_•Palindrome_** 🙌🏼")
    else:
        st.write("⤷ ❌ **_•Not Palindrome_** 😔")

# Add the message below the Classify button
st.markdown("✷ **Made in ❤️ by 4 IIT-Bombay students.**")

st.markdown("✷ **Hosted in ⛅️**")

# Display 1.gif image
st.image(gif_image_path, use_column_width=True)
