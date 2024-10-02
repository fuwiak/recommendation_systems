import streamlit as st
from PIL import Image
import easyocr
import requests
import re
import numpy as np
import nltk
from nltk.corpus import wordnet
import spacy
from io import BytesIO

# Ensure nltk data is downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

st.title('Receipt OCR and Product Matching App')

# Initialize session state variables
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None
if 'image' not in st.session_state:
    st.session_state['image'] = None
if 'rotate_custom' not in st.session_state:
    st.session_state['rotate_custom'] = 0
if 'rotate_slider' not in st.session_state:
    st.session_state['rotate_slider'] = 0

# Sidebar components
with st.sidebar:
    st.header("Upload and Settings")
    # File uploader for images
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        st.session_state['uploaded_file'] = uploaded_file
        st.session_state['image'] = Image.open(st.session_state['uploaded_file'])
        st.session_state['rotate_custom'] = 0
        st.session_state['rotate_slider'] = 0

    # Option to select OCR method
    ocr_method = st.selectbox("Select OCR Method", ["EasyOCR", "Unstructured (simulated)"])

    # Buttons to rotate image
    rotate_left_90 = st.button('Rotate Left 90°')
    rotate_right_90 = st.button('Rotate Right 90°')
    reset_rotation = st.button('Reset Rotation to 0°')

    # Rotate Image by Degrees slider
    rotate_custom = st.slider('Rotate Image by Degrees', -180, 180, value=st.session_state['rotate_slider'], key='rotate_slider')
    st.session_state['rotate_custom'] = rotate_custom

    clear_file = st.button('Clear Uploaded File')

if clear_file:
    st.session_state['uploaded_file'] = None
    st.session_state['image'] = None
    st.session_state['rotate_custom'] = 0
    st.session_state['rotate_slider'] = 0
    st.write("File cleared. Please upload a new image.")

if st.session_state['uploaded_file'] is not None:
    image = st.session_state['image']

    # Rotate image if buttons are pressed
    if rotate_left_90:
        image = image.rotate(90, expand=True)
        st.session_state['image'] = image
    if rotate_right_90:
        image = image.rotate(-90, expand=True)
        st.session_state['image'] = image
    if reset_rotation:
        st.session_state['rotate_custom'] = 0
        st.session_state['rotate_slider'] = 0
        image = st.session_state['image']
    if st.session_state['rotate_custom'] != 0:
        image = image.rotate(-st.session_state['rotate_custom'], expand=True)
        st.session_state['image'] = image

    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Process Image'):
        if ocr_method == 'EasyOCR':
            # Use EasyOCR to extract text
            reader = easyocr.Reader(['en'])
            result = reader.readtext(np.array(image))
            extracted_text = ' '.join([res[1] for res in result])
        else:
            # Simulate Unstructured OCR (since it may not support images directly)
            reader = easyocr.Reader(['en'])
            result = reader.readtext(np.array(image))
            extracted_text = ' '.join([res[1] for res in result])

        st.write('### Extracted Text:')
        st.write(extracted_text)

        # Use spaCy NLP model to extract product names
        doc = nlp(extracted_text)
        product_names = []
        for ent in doc.ents:
            if ent.label_ in ['PRODUCT', 'ORG', 'GPE', 'PERSON', 'NORP']:
                product_names.append(ent.text)

        # Remove duplicates
        product_names = list(set(product_names))

        # Focus on 'bacon' if present
        if 'bacon' in extracted_text.lower():
            if 'bacon' not in product_names:
                product_names.append('bacon')

        st.write('### Extracted Product Names:')
        for product in product_names:
            st.write(f"- {product}")

        # Product Classification and Similar Product Suggestions
        # Function to search products in OpenFoodFacts
        def search_products(query):
            url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={query}&search_simple=1&action=process&json=1"
            response = requests.get(url)

            if response.status_code == 200:
                products = response.json().get('products', [])
                return products
            else:
                st.write(f"Error: Unable to fetch products (status code: {response.status_code})")
                return []

        # Improved display using expanders and markdown
        st.write('### Product Matching and Classification:')
        for product in product_names:
            st.markdown(f"#### Product: **{product}**")
            products = search_products(product)
            if products:
                with st.expander('Matched Products'):
                    for p in products[:3]:
                        st.markdown(f"- **{p.get('product_name', 'No Name')}** (Brand: {p.get('brands', 'No Brand')})")
                        categories = p.get('categories', 'Unknown')
                        st.markdown(f"  - Categories: {categories}")
            else:
                st.write('No products found.')

            # Suggest similar products
            st.markdown('**Similar Product Suggestions:**')
            synonyms = []
            for syn in wordnet.synsets(product):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != product.lower() and synonym not in synonyms:
                        synonyms.append(synonym)
            if synonyms:
                st.write(f"Similar products to **{product}**: {', '.join(synonyms[:5])}")
                with st.expander('Similar Products Matches'):
                    for synonym in synonyms[:5]:
                        products = search_products(synonym)
                        if products:
                            st.markdown(f'**Matched Products for {synonym}:**')
                            for p in products[:2]:
                                st.markdown(f"- **{p.get('product_name', 'No Name')}** (Brand: {p.get('brands', 'No Brand')})")
                        else:
                            st.write(f'No products found for {synonym}.')
            else:
                st.write(f"No synonyms found for **{product}**.")

else:
    st.write("Please upload an image file to begin.")
