import pandas as pd
import re

RUTA_ENTRADA = "pacientes_validos.csv"
RUTA_SALIDA = "pacientes_imputados.csv"

NUMERICAS = ["EDAD", "TALLA", "PESO", "IMC", "T4L", "TSH"]
EXCLUIR = ["ID", "NSS", "FECHA DE REALIZACION DE USG", "OTROS",
           "NODULO BENIGNO", "CANCER DE TIROIDES"]


def clave(nombre):
    return nombre.strip().upper()  # normaliza el nombre


def contar_nulos(t):
    total = 0  # contador de nulos
    for i in range(len(t)):  # recorro filas
        for j in range(len(t.columns)):  # recorro columnas
            v = t.iloc[i, j]  # valor de la celda
            if pd.isna(v) or v == "":  # nulo o vacio
                total = total + 1
    return total


def mediana(nums):
    v = sorted(float(x) for x in nums if not pd.isna(x))  # valores validos ordenados
    if len(v) == 0:  # sin datos
        return float("nan")
    m = len(v) // 2  # indice de en medio
    return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2  # impar o par


def moda(vals):
    v = [x for x in vals if not pd.isna(x)]  # quito nulos
    if len(v) == 0:  # sin datos
        return None
    mejor, cuenta = None, 0
    for a in v:  # cuento cada valor recorriendo todo
        c = sum(1 for b in v if b == a)
        if c > cuenta or (c == cuenta and a < mejor):  # mas frecuente, empate al menor
            mejor, cuenta = a, c
    return mejor


def main():
    nulos_antes = contar_nulos(pd.read_csv(RUTA_ENTRADA, sep=";", encoding="utf-8-sig", dtype=str))  # cuento antes
    df = pd.read_csv(RUTA_ENTRADA, sep=";", encoding="utf-8-sig", dtype=str)  # leo de nuevo

    excluir = [clave(c) for c in EXCLUIR]  # excluidas normalizadas
    numericas = [clave(c) for c in NUMERICAS]  # numericas normalizadas

    for col in df.columns:  # recorro columnas
        if clave(col) in excluir:  # me salto excluidas
            continue
        limpia = [pd.NA if isinstance(v, str) and re.search(r"^\s*$", v) else v
                  for v in df[col].tolist()]  # vacios a nulo
        if clave(col) in numericas:  # rama numerica
            num = pd.to_numeric(pd.Series(limpia, dtype=object), errors="coerce").tolist()
            relleno = mediana(num)  # mediana a mano
            df[col] = pd.Series([relleno if pd.isna(v) else v for v in num], index=df.index)
        else:  # rama categorica
            m = moda(limpia)  # moda a mano
            if m is not None:
                df[col] = pd.Series([m if pd.isna(v) else v for v in limpia],
                                    index=df.index, dtype=object)

    df.to_csv(RUTA_SALIDA, sep=";", encoding="utf-8-sig", index=False)  # guardo

    pacientes = 0  # cuento pacientes con ciclo
    for _ in range(len(df)):
        pacientes = pacientes + 1

    print(f"Pacientes:         {pacientes}")
    print(f"Nulos antes:       {nulos_antes}")
    print(f"Nulos despues:     {contar_nulos(df)}")
    print(f"Guardado en:       {RUTA_SALIDA}")


if __name__ == "__main__":
    main()