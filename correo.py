import pandas as pd

# Función para eliminar URLs y palabras cortas de una cadena de texto
def clean_text(text):
    # Divide el texto por los espacios en blanco
    words = text.split()
    # Filtra las palabras que no son URLs ni tienen longitud igual o inferior a 2
    filtered_words = [word for word in words if not ('http://' in word or 'https://' in word or len(word) <= 2)]
    # Une las palabras filtradas de nuevo en una cadena de texto
    return ' '.join(filtered_words)

# Ruta al archivo de texto
file_tuistBases = r'C:\Users\aldai\OneDrive\Escritorio\Correo\tuitsBases.txt'

# Cargar el archivo en un DataFrame
df = pd.read_csv(file_tuistBases, delimiter=',', header=None)

# Eliminar la primera columna (índice)
df = df.drop(columns=[0])

# Aplicar la función para limpiar el texto a todas las columnas de texto en el DataFrame
df = df.applymap(lambda x: clean_text(x) if isinstance(x, str) else x)

# Mostrar las primeras filas del DataFrame
print(df.head())

# Ruta para el nuevo archivo
file_tuistBasesNuevo = r'C:\Users\aldai\OneDrive\Escritorio\Correo\tuitsBases_nuevo.txt'

# Guardar el DataFrame en un nuevo archivo de texto
df.to_csv(file_tuistBasesNuevo, index=False, header=False, sep=',')

print(f"Archivo guardado en: {file_tuistBasesNuevo}")
