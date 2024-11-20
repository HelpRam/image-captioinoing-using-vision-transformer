

# **Image Captioning using Vision Transformer and GPT for Text Generation**

This project utilizes **Vision Transformer (ViT)** for image feature extraction and **GPT** (from Hugging Face) for generating descriptive captions for images. The goal of this project is to showcase how advanced AI models can be combined to generate meaningful and accurate captions for images.

### **Project Overview**

In this project, a **Vision Transformer (ViT)** model is used for extracting visual features from images. These features are then fed into a **GPT** model (fine-tuned on image-related text) to generate synchronized captions that describe the content of the image. The entire workflow is deployed as an interactive web application using **Streamlit**, enabling users to upload images and get captions in real time.

---

## **Key Features**

- **Image Upload**: Users can upload any image in JPG format.
- **Vision Transformer (ViT)**: The pre-trained ViT model is used to extract image features.
- **GPT-based Caption Generation**: Using a pre-trained GPT model from Hugging Face, captions are generated based on image features.
- **Streamlit App**: The app provides a clean, user-friendly interface for uploading images and displaying generated captions.
- **Real-time Interaction**: Immediate caption generation after uploading the image.

---

## **Project Architecture**

### **Model Architecture**

1. **Vision Transformer (ViT)**:
   - **Vision Transformer (ViT)** is a deep learning model that applies Transformer architecture (originally designed for NLP) to images. It divides the image into fixed-size patches, linearly embeds them, and processes them as a sequence.
   - The model uses attention mechanisms to capture long-range dependencies between different parts of the image, making it particularly effective for tasks like image captioning.

2. **GPT (Generative Pretrained Transformer)**:
   - **GPT** is a powerful language model from Hugging Face, pre-trained on vast text data. In this project, it is fine-tuned for the task of generating captions based on the features extracted by ViT.
   - GPT takes the visual features as input and generates a coherent caption describing the image in natural language.

---

## **Technologies Used**

- **Python 3.8+**
- **Streamlit**: For creating the interactive web interface.
- **Hugging Face Transformers**: For pre-trained Vision Transformer (ViT) and GPT models.
- **PyTorch**: For deep learning and model inference.
- **Pillow**: For handling image uploads and processing.
- **NumPy**: For array and tensor manipulation.
- **Transformers Library**: To load the models and generate captions.

---

## **Installation and Setup**

Follow these steps to set up the project locally:

1. **Clone the Repository**:
   Clone this repository to your local machine:

   ```bash
   git clone https://github.com/HelpRam/image-captioinoing-using-vision-transformer.git
   cd image-captioning-vt-gpt
   ```

2. **Set Up a Virtual Environment**:
   It’s recommended to set up a virtual environment to manage dependencies.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   Install the required dependencies using `requirements.txt`.

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App**:
   Start the Streamlit application by running:

   ```bash
   streamlit run app.py
   ```

   This will start the application at `http://localhost:8501/` in your browser.

---

## **Usage**

1. **Upload an Image**: On the homepage of the app, click the **Upload** button to choose an image file in JPG format from your local machine.
2. **Generate Caption**: Once the image is uploaded, the system will process the image and use the **Vision Transformer** model to extract features and the **GPT** model to generate a caption for the image.
3. **View Caption**: The generated caption will be displayed below the image.

---


## **Dependencies**

- **streamlit**: To create the web app.
- **transformers**: For Hugging Face models.
- **torch**: For working with PyTorch models.
- **Pillow**: For image handling.
- **numpy**: For array manipulations.

Install all dependencies using the following command:

```bash
pip install -r requirements.txt
```

---

## **Contributing**

Contributions are welcome! Feel free to fork the repository, open issues, or create pull requests to enhance this project. Some ideas for contributions could include:

- Fine-tuning the models on a custom dataset.
- Adding more features to the web app (e.g., adjusting caption styles).
- Improving caption generation quality.

---

## **Licensing**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## **Screenshots**



---


