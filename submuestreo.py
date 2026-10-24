import argparse
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

CARPETA_IMAGENES = os.path.join(
    "entropiesonthyroid", "images", "originalimages", "originalimages", "originalimages"
)
CARPETA_SALIDA = "resultados"

REGEX_IMAGEN = re.compile(r"\.(png|jpe?g|bmp|tiff?|gif|webp)$", re.IGNORECASE)
REGEX_ID = re.compile(r"[Pp]0*(\d+)")


# 1. Normalizar ID:  'P00124' -> 'P0124' ,  'P0001 2024-07-01' -> 'P0001'
def normalizar_id(texto):
    m = REGEX_ID.search(str(texto))
    if not m:
        return None
    return "P" + str(int(m.group(1))).zfill(4)


# 2. Cargar imagenes: la etiqueta es el nombre de la subcarpeta (benign / malignant),
#    igual que imageDatastore(...,'LabelSource','foldernames') en MATLAB
def cargar_imagenes(carpeta):
    filas = []
    for raiz, _, archivos in os.walk(carpeta):
        for nombre in sorted(archivos):
            if REGEX_IMAGEN.search(nombre):
                filas.append({
                    "ruta": os.path.join(raiz, nombre),
                    "paciente": normalizar_id(nombre),
                    "etiqueta": os.path.basename(raiz),
                })
    datos = pd.DataFrame(filas, columns=["ruta", "paciente", "etiqueta"])
    if datos["etiqueta"].nunique() < 2:
        raise ValueError(f"Se encontraron menos de 2 clases en {carpeta}: "
                         f"{datos['etiqueta'].unique().tolist()}")
    return datos.sort_values("ruta").reset_index(drop=True)


# 3. Submuestreo estratificado: conserva el N% de cada clase de YTr
#    (minimo 1 ejemplo por clase para no perder ninguna)
def submuestreo(XTr, YTr, N, semilla=None):
    if N < 0.20 or N > 0.80:
        raise ValueError("El valor de N debe estar en el intervalo [0.20, 0.80].")

    XTr, YTr = np.asarray(XTr), np.asarray(YTr)
    rng = np.random.default_rng(semilla)
    idx_sub = []
    for clase in np.unique(YTr):
        idx_clase = np.flatnonzero(YTr == clase)
        m = max(1, int(round(N * len(idx_clase))))
        idx_sub.extend(rng.choice(idx_clase, size=m, replace=False))

    idx_sub = rng.permutation(idx_sub)
    return XTr[idx_sub], YTr[idx_sub]


# 4. Particion K-fold estratificada (equivalente a cvpartition(Y,'KFold',k,'Stratify',true)).
#    Con por_paciente=True las imagenes de un mismo paciente quedan en el mismo fold.
def particion_kfold(datos, folds, semilla, por_paciente=True):
    if por_paciente:
        cvp = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=semilla)
        return list(cvp.split(datos["ruta"], datos["etiqueta"], groups=datos["paciente"]))
    cvp = StratifiedKFold(n_splits=folds, shuffle=True, random_state=semilla)
    return list(cvp.split(datos["ruta"], datos["etiqueta"]))


def resumen(nombre, Y):
    conteo = pd.Series(Y).value_counts().sort_index()
    texto = ", ".join(f"{c}={n} ({n / len(Y):.1%})" for c, n in conteo.items())
    print(f"  {nombre:<6} n={len(Y):<5} {texto}")


# 5. Flujo principal: el submuestreo se aplica SOLO al entrenamiento de cada fold
def main():
    parser = argparse.ArgumentParser(description="Submuestreo estratificado sobre K-fold")
    parser.add_argument("--data-dir", default=CARPETA_IMAGENES)
    parser.add_argument("--N", type=float, default=0.30, help="fraccion en [0.20, 0.80]")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--sin-agrupar", action="store_true",
                        help="no agrupar por paciente (permite fuga entre train y test)")
    parser.add_argument("--entrenar", action="store_true",
                        help="entrenar la red en cada fold y mostrar el accuracy")
    parser.add_argument("--modelo", default="vgg19", choices=["vgg19", "resnet18"])
    parser.add_argument("--epocas", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--paciencia", type=int, default=2)
    parser.add_argument("--val", type=float, default=0.15,
                        help="fraccion del submuestreo usada como validacion")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sin-preentrenado", action="store_true")
    args = parser.parse_args()

    if args.N < 0.20 or args.N > 0.80:
        parser.error("N debe estar en el intervalo [0.20, 0.80]")

    datos = cargar_imagenes(args.data_dir)
    print("Imagenes:", len(datos), "| Pacientes:", datos["paciente"].nunique())
    resumen("total", datos["etiqueta"])

    if args.entrenar:
        from entrenamiento import entrenar_y_evaluar   # solo se necesita torch si se entrena
    clases = sorted(datos["etiqueta"].unique())
    pacientes_por_ruta = dict(zip(datos["ruta"], datos["paciente"]))

    registros, resultados = [], []
    for run in range(1, args.runs + 1):
        semilla_run = args.semilla + run
        particion = particion_kfold(datos, args.folds, semilla_run, not args.sin_agrupar)
        for kfold, (idx_tr, idx_te) in enumerate(particion, start=1):
            XTr, YTr = datos["ruta"].values[idx_tr], datos["etiqueta"].values[idx_tr]
            XTe, YTe = datos["ruta"].values[idx_te], datos["etiqueta"].values[idx_te]
            subXTr, subYTr = submuestreo(XTr, YTr, args.N, semilla=semilla_run * 100 + kfold)

            print(f"\nRun {run} - Fold {kfold}")
            resumen("train", YTr)
            resumen("sub", subYTr)
            resumen("test", YTe)

            if args.entrenar:
                pacientesTr = np.array([pacientes_por_ruta[r] for r in subXTr])
                acc, cmat = entrenar_y_evaluar(subXTr, subYTr, pacientesTr, XTe, YTe,
                                               clases, args, semilla_run * 100 + kfold)
                print(f"  Accuracy en Fold {kfold}: {acc:.4f}")
                print("  Matriz de confusion (filas=real, columnas=prediccion):", clases)
                print("  " + str(cmat).replace("\n", "\n  "))
                resultados.append({"run": run, "fold": kfold, "accuracy": acc})

            for conjunto, X, Y in (("train", XTr, YTr), ("sub", subXTr, subYTr), ("test", XTe, YTe)):
                registros += [{"run": run, "fold": kfold, "conjunto": conjunto,
                               "ruta": x, "etiqueta": y} for x, y in zip(X, Y)]

    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    salida = os.path.join(CARPETA_SALIDA, f"folds_N{round(args.N * 100)}.csv")
    pd.DataFrame(registros).to_csv(salida, index=False, encoding="utf-8-sig")
    print("\nGuardado en:", salida)

    if resultados:
        tabla = pd.DataFrame(resultados)
        print(f"\nACCURACY POR FOLD (N = {args.N:.0%}, modelo = {args.modelo})")
        for _, fila in tabla.iterrows():
            print(f"  Run {fila['run']:.0f} - Fold {fila['fold']:.0f}: {fila['accuracy']:.4f}")
        print(f"  Promedio: {tabla['accuracy'].mean():.4f} +/- {tabla['accuracy'].std():.4f}")
        salida_acc = os.path.join(CARPETA_SALIDA, f"accuracy_N{round(args.N * 100)}_{args.modelo}.csv")
        tabla.to_csv(salida_acc, index=False)
        print("Guardado en:", salida_acc)


if __name__ == "__main__":
    main()
