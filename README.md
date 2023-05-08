# Rooftops_detection


Ce code contient l'implémentation d'un modèle de réseau neuronal convolutif (CNN) pour détecter les toits à partir d'images aériennes.

Le Dataset se compose d'images aériennes et de leurs étiquettes correspondantes. Les étiquettes sont stockées dans un fichier CSV (labels.csv) et chaque étiquette correspond à une variable binaire indiquant si l'image contient ou non un toit.

Le code charge d'abord le jeu de données et redimensionne les images à une taille commune de 64x64 pixels. Ensuite, il divise les données en ensembles d'entraînement et de test à l'aide de la fonction train_test_split de la bibliothèque scikit-learn.

L'architecture du modèle CNN est définie à l'aide de l'API Keras. Elle se compose de quatre paires de couches de convolution et de couches de max-pooling, suivies d'une couche de mise à plat, de deux couches entièrement connectées et d'une couche d'activation softmax. Le modèle est compilé à l'aide de la fonction de perte categorical_crossentropy et de l'optimiseur Adam. Pendant l'entraînement, le modèle est également surveillé à l'aide de la mesure de précision.

Le code utilise la classe ImageDataGenerator de l'API Keras pour générer une augmentation de données en temps réel pendant l'entraînement du modèle. Cela est fait pour augmenter la généralisation du modèle et éviter le surapprentissage. Le processus d'entraînement inclut également l'utilisation d'un ModelCheckpoint pour enregistrer le modèle qui fonctionne le mieux pendant l'entraînement.

Enfin, le code évalue les performances du modèle en calculant la précision et la matrice de confusion sur l'ensemble de test. Il trace également les courbes de précision et de perte pour les ensembles d'entraînement et de test à l'aide de la bibliothèque matplotlib.
