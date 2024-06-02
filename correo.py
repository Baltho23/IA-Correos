import pandas as pd

def remove_urls(text):

    words = text.split()
    words = [word for word in words if not ('http://' in word or 'https://' in word)]
    return ' '.join(words)

def remove_accents(text):

    accented_chars = 'áéíóúüÁÉÍÓÚÜ'
    unaccented_chars = 'aeiouuAEIOUU'
    text = ''.join(unaccented_chars[accented_chars.index(char)] if char in accented_chars else char for char in text)
    return text

def clean_text(text):
    
    text = remove_urls(text)
    cleaned_text = ''.join(char for char in remove_accents(text) if char.isalnum() or char.isspace())
    words = cleaned_text.split()
    filtered_words = [word for word in words if len(word) > 2 and any(c.isalpha() for c in word)]
    return ' '.join(filtered_words)

file_tuistBases = r'C:\Users\aldai\OneDrive\Escritorio\Correo\tuitsBases.txt'

df = pd.read_csv(file_tuistBases, delimiter=',', header=None)

df = df.drop(columns=[0, 1])

name_counts = df[2].value_counts()

df_filtered = df[df[2].isin(name_counts.index[name_counts >= 6])]

df_filtered = df_filtered.applymap(lambda x: clean_text(x) if isinstance(x, str) else x)

print(df_filtered.head())

file_tuistBasesNuevo = r'C:\Users\aldai\OneDrive\Escritorio\Correo\tuitsBases_nuevo.txt'

df_filtered.to_csv(file_tuistBasesNuevo, index=False, header=False, sep=',')

print(f"Archivo guardado en: {file_tuistBasesNuevo}")
