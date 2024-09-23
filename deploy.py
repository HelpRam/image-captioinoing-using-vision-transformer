import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast
from PIL import Image
import torch

# Function to apply custom styles from a CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Apply custom styles
local_css("style.css")

# Load the model, tokenizer, and image processor
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, tokenizer, image_processor

model, tokenizer, image_processor = load_model()

# Function to generate caption
def generate_caption(image):
    img = image_processor(image, return_tensors="pt").to(device)
    output = model.generate(**img)
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return caption

# App Layout and Design

# Add a custom background image or color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F0F8FF; /* Replace with your background color */
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the uploaded logo image
# st.image("/mnt/data/image.png", width=150)  # Logo displayed from uploaded file

# App title with custom color and font
st.markdown(
    "<h1 style='text-align: center; color: #32CD32; font-family: Arial, Helvetica, sans-serif;'>🎨 Image Caption Generator 🖼️</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="text-align: center; font-size: 18px; font-family: 'Arial', sans-serif; color: #4B0082;">
    Upload an image and let the AI generate a caption for it!
    </div>
    """,
    unsafe_allow_html=True
)

# File uploader with button color customization
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Custom button style for 'Generate Caption'
    if st.button("📝 Generate Caption", help="Click to generate caption"):
        with st.spinner('Generating caption...'):
            caption = generate_caption(image)
        
        # Highlight the generated caption with custom styling
        st.markdown(f"""
        <div style='text-align: center; margin-top: 20px;'>
            <h3 style='color: #FF4500; font-family: Arial, Helvetica, sans-serif;'>Generated Caption:</h3>
            <p style='color: #1E90FF; font-size: 24px; font-weight: bold; font-family: Arial, sans-serif;'>
            {caption}
            </p>
        </div>
        """, unsafe_allow_html=True)

# Footer with name in red color and social media links
st.markdown(
    """
    <div style='text-align: center; margin-top: 50px;'>
        <p style='font-family: sans-serif; font-size: 14px; color: #6c757d;'>
        Made with ❤️ by <a href='https://your-website-url.com' target='_blank' style='color: red; text-decoration: none;'>Ram</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
