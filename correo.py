import pandas as pd

# Función para eliminar URLs de una cadena de texto
def remove_urls(text):
    # Divide el texto por los espacios en blanco y filtra las partes que no contienen "http://" o "https://"
    return ' '.join(word for word in text.split() if not ('http://' in word or 'https://' in word))

# Ruta al archivo de texto
file_tuistBases = r'C:\Users\aldai\OneDrive\Escritorio\Correo\tuitsBases.txt'

# Cargar el archivo en un DataFrame
df = pd.read_csv(file_tuistBases, delimiter=',', header=None)

# Eliminar la primera columna (índice)
df = df.drop(columns=[0])

# Aplicar la función para eliminar URLs a todas las columnas de texto en el DataFrame
df = df.applymap(lambda x: remove_urls(x) if isinstance(x, str) else x)

# Ruta para el nuevo archivo
file_tuistBasesNuevo = r'C:\Users\aldai\OneDrive\Escritorio\Correo\tuitsBases_nuevo.txt'

# Guardar el DataFrame en un nuevo archivo de texto
df.to_csv(file_tuistBasesNuevo, index=False, header=False, sep=',')

print(f"Archivo guardado en: {file_tuistBasesNuevo}")
