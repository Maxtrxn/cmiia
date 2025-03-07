import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import filedialog

# Utiliser un backend adapté pour Tkinter
import matplotlib
matplotlib.use('TkAgg')

# Charger le modèle MobileNetV2 pour l'extraction des caractéristiques
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path, model):
    """Extrait les caractéristiques d'une image donnée."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

def load_database(image_folder):
    """Charge et extrait les caractéristiques de toutes les images de la base de données."""
    feature_dict = {}
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
            feature_dict[img_name] = extract_features(img_path, model)
    return feature_dict

def find_top_similar(input_img_path, database_features, top_n=5):
    """Trouve les N images les plus similaires à celle fournie par l'utilisateur."""
    input_features = extract_features(input_img_path, model)
    similarities = {}
    for img_name, features in database_features.items():
        sim = cosine_similarity([input_features], [features])[0][0]
        similarities[img_name] = sim
    top_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_matches

def display_results(input_img_path, similar_images, image_folder):
    """Affiche l'image de base et les images les plus similaires dans une fenêtre matplotlib."""
    fig, axes = plt.subplots(1, len(similar_images) + 1, figsize=(20, 5))
    
    # Afficher l'image de base
    input_img = cv2.imread(input_img_path)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(input_img)
    axes[0].set_title("Image de base")
    axes[0].axis("off")
    
    # Afficher les images similaires
    for i, (img_name, score) in enumerate(similar_images):
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f"Score: {score:.2f}")
        axes[i + 1].axis("off")
    
    plt.show(block=False)
    return fig

# Variables globales
image_db_path = "B2CMI/image_db"  # Dossier contenant les images de la base de données
database_features = load_database(image_db_path)
current_fig = None  # Pour stocker la figure matplotlib actuellement affichée

def process_image():
    global current_fig
    # Fermer la figure précédente s'il y en a une
    if current_fig is not None:
        plt.close(current_fig)
    # Ouvrir la boîte de dialogue pour choisir une image
    input_image = filedialog.askopenfilename(
        title="Choisir une image",
        filetypes=[("Fichiers image", "*.png *.jpg *.jpeg"), ("Tous les fichiers", "*.*")]
    )
    if input_image:
        top_similar_images = find_top_similar(input_image, database_features, top_n=5)
        current_fig = display_results(input_image, top_similar_images, image_db_path)

def main():
    # Création de la fenêtre principale Tkinter
    root = tk.Tk()
    root.title("Recherche d'images similaires")
    
    # Bouton pour lancer la sélection d'image (sert également de bouton restart)
    choose_button = tk.Button(root, text="Choisir une image", command=process_image)
    choose_button.pack(pady=20)
    
    # Bouton pour quitter l'application
    quit_button = tk.Button(root, text="Quitter", command=root.quit)
    quit_button.pack(pady=20)
    
    root.mainloop()

if __name__ == '__main__':
    main()
