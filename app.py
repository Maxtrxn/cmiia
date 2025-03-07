import os
import zipfile
import pickle
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import kagglehub

ImageTestPath = '/home/nicolas/L3/BureauCMI/images/Beagle_1.jpg'

# 1. Télécharger le dataset depuis Kaggle
print("Téléchargement du dataset...")
dataset_path = kagglehub.dataset_download("khushikhushikhushi/dog-breed-image-dataset")
print("Dataset téléchargé à :", dataset_path)

# 2. Extraire le dataset s'il s'agit d'une archive zip
if dataset_path.endswith('.zip'):
    extract_path = os.path.join(os.path.dirname(dataset_path), 'dog_breed_dataset')
    print("Extraction du dataset dans :", extract_path)
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
else:
    extract_path = dataset_path  # le dataset est déjà un dossier

# 3. Charger le modèle VGG16 pré-entraîné (sans couche de classification)
model = VGG16(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path, model):
    """
    Charge une image, la pré-traite et extrait ses caractéristiques avec le modèle.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

# 4. Charger ou créer la base de données des caractéristiques
features_db_file = 'features_db.pkl'

if os.path.exists(features_db_file):
    print("Chargement des caractéristiques depuis le fichier...")
    with open(features_db_file, 'rb') as f:
        features_db = pickle.load(f)
else:
    features_db = {}
    print("Indexation des images du dataset...")
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                try:
                    features_db[img_path] = extract_features(img_path, model)
                    print(f"Image traitée : {img_path}")
                except Exception as e:
                    print(f"Erreur lors du traitement de {img_path}: {e}")
    print("Sauvegarde des caractéristiques dans le fichier...")
    with open(features_db_file, 'wb') as f:
        pickle.dump(features_db, f)

def find_similar_images(query_img_path, model, features_db, top_k=10):
    """
    Pour une image requête, calcule la similarité (cosinus) avec toutes les images indexées
    et renvoie les top_k images les plus similaires.
    """
    query_features = extract_features(query_img_path, model)
    similarities = {}
    for img_path, feat in features_db.items():
        sim = cosine_similarity([query_features], [feat])[0][0]
        similarities[img_path] = sim
    similar_images = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return similar_images

# 5. Exemple d'utilisation : rechercher les images similaires à une image requête
# Remplacez par le chemin vers votre image de test
query_image = ImageTestPath
results = find_similar_images(query_image, model, features_db)

print("\nTop des images similaires :")
for path, sim in results:
    print(f"{path} avec une similarité de {sim:.2f}")


