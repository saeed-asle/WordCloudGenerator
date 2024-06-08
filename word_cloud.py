import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import numpy as np
from PIL import Image

def load_text(file_path):
    with open(file_path, "r", encoding='utf-8') as file:
        text = file.read()
    return text

def load_image_mask(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)  # Convert the image to a NumPy array
    # Use the red channel for the mask
    mask = image_array[:, :, 0]
    # Transform the mask to match WordCloud requirements
    transformed_mask = np.ndarray((mask.shape[0], mask.shape[1]), np.int32)
    for i in range(len(mask)):
        transformed_mask[i] = list(map(lambda val: 255 if val == 0 else val, mask[i]))
    return transformed_mask

def generate_word_cloud(text, mask, save_path, contour_width=0):
    wc = WordCloud(
        background_color="white",
        max_words=1000,
        mask=mask,
        contour_width=contour_width,
        contour_color='firebrick'
    )
    wc.generate(text)
    wc.to_file(save_path)

def display_word_cloud(word_cloud_image_path):
    word_cloud = Image.open(word_cloud_image_path)
    plt.figure(figsize=[20, 10])
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

text = load_text("civil.txt")
image_mask = load_image_mask("0.jpg")
word_cloud_save_path = "Cloud Image.jpg"
generate_word_cloud(text, image_mask, word_cloud_save_path)
display_word_cloud(word_cloud_save_path)


