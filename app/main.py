import streamlit as st
from PIL import Image
import easyocr
import numpy as np
import requests
import nltk
from nltk.corpus import wordnet
import spacy
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# Ensure nltk data is downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load spaCy models for different languages
en_model = spacy.load('en_core_web_sm')  # English
fr_model = spacy.load('fr_core_news_sm')  # French
nl_model = spacy.load('nl_core_news_sm')  # Dutch

# Language map
language_models = {
    'en': en_model,
    'fr': fr_model,
    'nl': nl_model
}

st.title('Receipt OCR, Language Detection, and Product Matching App')

# Initialize session state variables
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None
if 'image' not in st.session_state:
    st.session_state['image'] = None
if 'rotated_image' not in st.session_state:
    st.session_state['rotated_image'] = None
if 'rotation_angle' not in st.session_state:
    st.session_state['rotation_angle'] = 0
if 'ocr_model' not in st.session_state:
    st.session_state['ocr_model'] = None
if 'extracted_text' not in st.session_state:
    st.session_state['extracted_text'] = ""
if 'detected_lang' not in st.session_state:
    st.session_state['detected_lang'] = None

# Sidebar components
with st.sidebar:
    st.header("Upload and Settings")

    # File uploader for images
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        st.session_state['uploaded_file'] = uploaded_file
        st.session_state['image'] = Image.open(st.session_state['uploaded_file'])
        st.session_state['rotated_image'] = st.session_state['image']  # Initialize rotated image
        st.session_state['rotation_angle'] = 0  # Reset rotation when a new image is uploaded

    # OCR method selection
    ocr_method = st.selectbox("Select OCR Method", ["EasyOCR", "Unstructured (simulated)"])

    # Rotate Image buttons
    rotate_left = st.button('Rotate Left 90°')
    rotate_right = st.button('Rotate Right 90°')

    # Rotate Image by Degrees slider
    rotate_custom = st.slider('Rotate Image by Degrees', -180, 180, value=st.session_state['rotation_angle'],
                              key='rotate_slider')

    # Clear uploaded file
    clear_file = st.button('Clear Uploaded File')

# Handle Image Rotation
if st.session_state['image'] is not None:
    if rotate_left:
        st.session_state['rotation_angle'] += 90
    if rotate_right:
        st.session_state['rotation_angle'] -= 90

    # Apply custom rotation from slider
    if rotate_custom != st.session_state['rotation_angle']:
        st.session_state['rotation_angle'] = rotate_custom

    # Apply the rotation to the image and save the rotated image in session state
    st.session_state['rotated_image'] = st.session_state['image'].rotate(st.session_state['rotation_angle'],
                                                                         expand=True)

# Clear file logic
if clear_file:
    st.session_state['uploaded_file'] = None
    st.session_state['image'] = None
    st.session_state['rotated_image'] = None
    st.session_state['rotation_angle'] = 0
    st.write("File cleared. Please upload a new image.")

# If the image is uploaded, display the rotated image
if st.session_state['rotated_image'] is not None:
    st.image(st.session_state['rotated_image'], caption='Uploaded Image', use_column_width=True)

    # Button to process the image
    if st.button('Process Image'):
        # Only load the OCR model once for EasyOCR
        if ocr_method == 'EasyOCR':
            if st.session_state['ocr_model'] is None:
                st.session_state['ocr_model'] = easyocr.Reader(['en', 'fr', 'nl'])  # Support English, French, Dutch

            # Use EasyOCR to extract text from the rotated image
            reader = st.session_state['ocr_model']
            result = reader.readtext(np.array(st.session_state['rotated_image']))
            st.session_state['extracted_text'] = ' '.join([res[1] for res in result])

        elif ocr_method == 'Unstructured (simulated)':
            # Simulate the unstructured model processing (same result extraction method for demo purposes)
            st.session_state['extracted_text'] = "Simulated OCR model has extracted this placeholder text."

    # Show extracted text if available
    if st.session_state['extracted_text']:
        st.write('### Extracted Text:')
        st.write(st.session_state['extracted_text'])

        # Language Detection
        try:
            detected_lang = detect(st.session_state['extracted_text'])
            st.session_state['detected_lang'] = detected_lang
            st.write(f"### Detected Language: **{detected_lang.upper()}**")

            # Use corresponding spaCy model
            if detected_lang in language_models:
                nlp = language_models[detected_lang]
            else:
                st.write("Unsupported language detected, defaulting to English.")
                nlp = language_models['en']

            # Use spaCy NLP model to extract product names
            doc = nlp(st.session_state['extracted_text'])
            product_names = [ent.text for ent in doc.ents if ent.label_ in ['PRODUCT', 'ORG', 'GPE', 'PERSON', 'NORP']]

            # Remove duplicates
            product_names = list(set(product_names))

            # Show extracted product names
            st.write('### Extracted Product Names:')
            for product in product_names:
                st.write(f"- {product}")


            # Product Matching and Classification
            def search_products(query):
                url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={query}&search_simple=1&action=process&json=1"
                response = requests.get(url)

                if response.status_code == 200:
                    products = response.json().get('products', [])
                    return products
                else:
                    st.write(f"Error: Unable to fetch products (status code: {response.status_code})")
                    return []


            st.write('### Product Matching and Classification:')
            for product in product_names:
                st.markdown(f"#### Product: **{product}**")
                products = search_products(product)
                if products:
                    with st.expander('Matched Products'):
                        for p in products[:3]:
                            st.markdown(
                                f"- **{p.get('product_name', 'No Name')}** (Brand: {p.get('brands', 'No Brand')})")
                            categories = p.get('categories', 'Unknown')
                            st.markdown(f"  - Categories: {categories}")
                else:
                    st.write('No products found.')

        except LangDetectException:
            st.write("Language detection failed. Please try again with more text.")
else:
    st.write("Please upload an image file to begin.")
