import csv
import os
import re

RUTA_CSV = r"BASE IMAGENES.csv"
CARPETA_IMAGENES = r"D:\Mario\IPN\OCTAVO SEMESTRE\ANALITICA DE DATOS\Repo_nodulos\entropiesonthyroid\images\originalimages\originalimages\originalimages"
RUTA_SALIDA = r"pacientes_validos.csv"

REGEX_IMAGEN = re.compile(r"\.(png|jpe?g|bmp|tiff?|gif|webp)$", re.IGNORECASE)
REGEX_ID = re.compile(r"[Pp]0*(\d+)")


# 1. Normalizar ID:  'P00124' -> 'P0124' ,  'P0001 2024-07-01' -> 'P0001'
def normalizar_id(texto):
    m = REGEX_ID.search(str(texto))
    if not m:
        return None
    return "P" + str(int(m.group(1))).zfill(4)


# 2. Leer tabla: descarta la cabecera y las filas sin ID
def leer_tabla(ruta_csv):
    with open(ruta_csv, encoding="utf-8-sig", newline="") as f:
        todas = list(csv.reader(f, delimiter=";"))

    cabecera = todas[0]
    filas = [fila for fila in todas[1:] if fila and fila[0].strip() != ""]
    return cabecera, filas


# 3. Inventario recursivo de imagenes:  { 'P0001': [ruta, ruta, ...] }
def buscar_imagenes(carpeta, mapa=None):
    if mapa is None:
        mapa = {}
    for nombre in os.listdir(carpeta):
        ruta = os.path.join(carpeta, nombre)
        if os.path.isdir(ruta):
            buscar_imagenes(ruta, mapa)          # llamada recursiva
        elif REGEX_IMAGEN.search(nombre):
            id_norm = normalizar_id(nombre)
            if id_norm:
                mapa.setdefault(id_norm, []).append(ruta)
    return mapa

# 4. Cruzar tabla con imagenes
def cruzar(filas, mapa):
    validos, a_eliminar = [], []
    for fila in filas:
        if mapa.get(normalizar_id(fila[0])):
            validos.append(fila)
        else:
            a_eliminar.append(fila)
    return validos, a_eliminar


# 5. Guardar solo los validos
def guardar(cabecera, validos, ruta_salida):
    with open(ruta_salida, "w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.writer(f, delimiter=";")
        escritor.writerow(cabecera)
        escritor.writerows(validos)


# 6. Flujo principal
def main():
    cabecera, filas = leer_tabla(RUTA_CSV)
    mapa = buscar_imagenes(CARPETA_IMAGENES)
    validos, a_eliminar = cruzar(filas, mapa)
    validos.sort(key=lambda fila : int(REGEX_ID.search(fila[0]).group(1)))
    guardar(cabecera, validos, RUTA_SALIDA)

    print("Filas con ID:      ", len(filas))
    print("IDs con imagen:    ", len(mapa))
    print("Pacientes VALIDOS: ", len(validos))
    print("Pacientes ELIMINAR:", len(a_eliminar))
    print("Guardado en:", RUTA_SALIDA)


if __name__ == "__main__":
    main()