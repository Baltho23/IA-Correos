#Integrantes
#    Airton Sampayo
#    Juan Diego Marin
#    Keyner Barrios
#    Eliecer Ureche

import pandas as pd
import numpy as np

# Librerias para matriz de confusion
import matplotlib.pyplot as plt
import seaborn as sns

# Reemplazo del vectorizador de la libreria
class CountVectorizer:
    def __init__(self):
        self.vocabulary_ = {}
    
    def fit(self, documents):
        unique_words = set(word for doc in documents for word in doc.split())
        self.vocabulary_ = {word: idx for idx, word in enumerate(unique_words)}
    
    def transform(self, documents):
        vectors = []
        for doc in documents:
            vector = [0] * len(self.vocabulary_)
            for word in doc.split():
                if word in self.vocabulary_:
                    vector[self.vocabulary_[word]] += 1
            vectors.append(vector)
        return np.array(vectors)

#

def remove_urls(text):
    # Dividir el texto por espacios en blanco
    words = text.split()
    # Filtrar las palabras que no son enlaces URL
    words = [word for word in words if not ('http://' in word or 'https://' in word)]
    # Unir las palabras de nuevo en una cadena de texto
    return ' '.join(words)

def remove_accents(text):
    # Lista de caracteres acentuados
    accented_chars = 'áéíóúüÁÉÍÓÚÜ'
    # Lista de caracteres sin acento correspondientes
    unaccented_chars = 'aeiouuAEIOUU'
    # Reemplazar caracteres acentuados por sus equivalentes sin acento
    text = ''.join(unaccented_chars[accented_chars.index(char)] if char in accented_chars else char for char in text)
    return text

def clean_text(text):
    # Eliminar enlaces URL
    text = remove_urls(text)
    # Eliminar caracteres especiales y tildes
    cleaned_text = ''.join(char for char in remove_accents(text) if char.isalnum() or char.isspace())
    # Divide el texto por los espacios en blanco
    words = cleaned_text.split()
    # Filtra las palabras que no tienen longitud mayor a 2 y contienen caracteres alfabéticos
    filtered_words = [word for word in words if len(word) > 2 and any(c.isalpha() for c in word)]
    # Une las palabras filtradas de nuevo en una cadena de texto
    return ' '.join(filtered_words)

# Reemplazo del separador de datos para pruebas
def train_test_split(data, test_size=0.2, random_state=None):
    
    if random_state:
        data = data.sample(frac=1, random_state=random_state)
    else:
        data = data.sample(frac=1)
    
    split_index = int(len(data) * (1 - test_size))
    
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]
    
    return train_data, test_data

# Ruta al archivo de texto
file_tuits_bayes = 'tuitsBases.txt'

# Cargar el archivo en un DataFrame
df = pd.read_csv(file_tuits_bayes, delimiter=',', header=None, names=['status_id', 'screen_name', 'text'])

# Calcular cuántas veces aparece cada nombre de usuario
name_counts = df['screen_name'].value_counts()

# Filtrar el DataFrame original para mantener solo las filas donde el nombre de usuario aparezca al menos 6 veces
df_filtered = df[df['screen_name'].isin(name_counts.index[name_counts >= 6])]

# Aplicar la función para limpiar el texto
df_filtered['text'] = df_filtered['text'].apply(clean_text)

# Mostrar las primeras filas del DataFrame filtrado
print(df_filtered.head())

# Guardar el DataFrame filtrado en un nuevo archivo de texto (opcional)
file_tuits_bayes_nuevo = 'tuitsLimpios.txt'
df_filtered.to_csv(file_tuits_bayes_nuevo, index=False, sep=',')

# Dividir los datos en conjuntos de entrenamiento y prueba
train_df = pd.DataFrame()
test_df = pd.DataFrame()

for user in df_filtered['screen_name'].unique():
    user_tweets = df_filtered[df_filtered['screen_name'] == user]
    train, test = train_test_split(user_tweets, test_size=0.2, random_state=40)
    train_df = pd.concat([train_df, train])
    test_df = pd.concat([test_df, test])

# Calcular las probabilidades a priori
prior_probabilities = train_df['screen_name'].value_counts(normalize=True)

# Vectorizar los datos
vectorizer = CountVectorizer()
vectorizer.fit(train_df['text'])
X_train = vectorizer.transform(train_df['text'])
y_train = train_df['screen_name']

# Implementación del clasificador Naive Bayes desde cero
class NaiveBayes:
    def __init__(self):
        self.priors = None
        self.class_word_counts = None
    
    def fit(self, X, y):
        self.priors = {label: np.log(prob) for label, prob in prior_probabilities.items()}
        self.class_word_counts = {label: np.log(X[y == label].sum(axis=0) + 1) for label in prior_probabilities.index}
    
    def predict(self, X):
        predictions = []
        for row in X:
            scores = {label: prior for label, prior in self.priors.items()}
            for label, word_counts in self.class_word_counts.items():
                scores[label] += np.dot(row, word_counts)
            predicted_label = max(scores, key=scores.get)
            predictions.append(predicted_label)
        return predictions

# Entrenar el modelo Naive Bayes
model = NaiveBayes()
model.fit(X_train, y_train)

# Probar el modelo
X_test = vectorizer.transform(test_df['text'])
y_test = test_df['screen_name']
y_pred = model.predict(X_test)

# Función para calcular la matriz de confusión
def confusion_matrix(y_true, y_pred, labels):
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    for true, pred in zip(y_true, y_pred):
        matrix[label_to_index[true], label_to_index[pred]] += 1
    return matrix

# Función para calcular precisión, recall y F1-score
def calculate_metrics(conf_matrix):
    # Precision: TP / (TP + FP)
    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    # Recall: TP / (TP + FN)
    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score

# Etiquetas de clase
labels = prior_probabilities.index

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred, labels)

# Calcular precisión, recall y F1-score
precision, recall, f1_score = calculate_metrics(conf_matrix)

# Crear tabla de resumen
summary_table = pd.DataFrame({
    'User': labels,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1_score
})

# Calcular precisión general
overall_accuracy = np.mean(y_pred == y_test)

# Mostrar los resultados
print("Matriz de Confusión:")
print(conf_matrix)
print("\nTabla Resumen:")
print(summary_table)
print("\nPrecisión General:", overall_accuracy)

# Grafica de matriz de confusion
labels = ['lopezobrador_', 'UNAM_MX','MSFTMexico','CMLL_OFICIAL'] 

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicción')
plt.ylabel('Verdadero')
plt.title('Matriz de Confusión')
plt.show()
