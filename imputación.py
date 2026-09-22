"""
Imputación de atributos vacíos en pacientes_validos.csv

Una sola técnica, dos reglas:
  - Columnas numéricas (EDAD, TALLA, PESO, IMC, T4L, TSH) -> mediana.
  - Columnas binarias/categóricas (SEXO, comorbilidades, DX POR IMC) -> moda.

No se imputan: identificadores/fecha (ID, NSS, FECHA), texto libre (OTROS)
ni las etiquetas (NODULO BENIGNO, CANCER DE TIROIDES).
"""

import pandas as pd

RUTA_ENTRADA = "pacientes_validos.csv"
RUTA_SALIDA = "pacientes_imputados.csv"

# Columnas que se imputan con la MEDIANA (numéricas continuas).
NUMERICAS = ["EDAD", "TALLA", "PESO", "IMC", "T4L", "TSH"]

# Columnas que NO se imputan (identificadores, fecha, texto libre y etiquetas).
EXCLUIR = [
    "ID", "NSS", "FECHA DE REALIZACION DE USG", "OTROS",
    "NODULO BENIGNO", "CANCER DE TIROIDES",
]


def clave(nombre):
    """Normaliza el nombre de columna (quita espacios sobrantes) para comparar."""
    return nombre.strip().upper()


def main():
    df = pd.read_csv(RUTA_ENTRADA, sep=";", encoding="utf-8-sig", dtype=str)

    numericas = {clave(c) for c in NUMERICAS}
    excluir = {clave(c) for c in EXCLUIR}

    nulos_antes = df.isna().sum().sum() + (df == "").sum().sum()

    for col in df.columns:
        k = clave(col)
        if k in excluir:
            continue

        # Tratar cadenas vacías como valores faltantes.
        serie = df[col].replace(r"^\s*$", pd.NA, regex=True)

        if k in numericas:
            # Numérica -> mediana.
            num = pd.to_numeric(serie, errors="coerce")
            relleno = num.median()
            df[col] = num.fillna(relleno)
        else:
            # Binaria / categórica -> moda (valor más frecuente).
            moda = serie.mode(dropna=True)
            if not moda.empty:
                df[col] = serie.fillna(moda[0])

    df.to_csv(RUTA_SALIDA, sep=";", encoding="utf-8-sig", index=False)

    nulos_despues = df.isna().sum().sum() + (df == "").sum().sum()
    print(f"Pacientes:         {len(df)}")
    print(f"Nulos antes:       {nulos_antes}")
    print(f"Nulos despues:     {nulos_despues}")
    print(f"Guardado en:       {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
