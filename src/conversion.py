from pathlib import Path
import pickle

"""
Il seguente file è stato realizzato per trasformare, i file oggetto serializzati in file di testo.
"""

# Variabile che prende il valore del path in cui si trova il file
script_path = Path(__file__)

# Crea il percorso completo al file utilizzando pathlib
data_read = script_path.parent.parent / "Serialized" / \
    "BestConfigurationPCAKNN.pkl"

data_write = script_path.parent.parent / "BestConfiguration" / \
    "BestConfigurationPCAKNN.txt"

# Carica i dati serializzati dal file pickle
with open(data_read, 'rb') as file:
    dati = pickle.load(file)

# Converti i dati in una rappresentazione di testo
testo = str(dati)

# Scrivi la rappresentazione di testo su un nuovo file di testo
with open(data_write, 'w') as file:
    file.write(testo)
