import os
import tempfile
from collections import Counter

import numpy as np
import pytest

from submuestreo import cargar_imagenes, particion_kfold, submuestreo


def crear_dataset(carpeta, n_benign=60, n_malignant=20):
    # Misma estructura que el repo: <carpeta>/benign|malignant/P#### fecha n.png
    for clase, n in (("benign", n_benign), ("malignant", n_malignant)):
        os.makedirs(os.path.join(carpeta, clase))
        for i in range(n):
            paciente = i // 2 + (0 if clase == "benign" else 25)
            open(os.path.join(carpeta, clase, f"P{paciente:04d} 2024-07-15 {i}.png"), "w").close()


@pytest.fixture
def datos():
    with tempfile.TemporaryDirectory() as tmp:
        crear_dataset(tmp)
        yield cargar_imagenes(tmp)


def test_carga(datos):
    assert len(datos) == 80
    assert Counter(datos["etiqueta"]) == {"benign": 60, "malignant": 20}
    assert datos["paciente"].str.match(r"^P\d{4}$").all()


@pytest.mark.parametrize("N", [0.2, 0.3, 0.5, 0.8])
def test_proporciones_y_tamano(N):
    Y = np.array(["benign"] * 300 + ["malignant"] * 45)
    X = np.arange(len(Y))
    subX, subY = submuestreo(X, Y, N, semilla=0)
    for clase, n in Counter(Y).items():
        assert abs(Counter(subY)[clase] - N * n) <= 1
    assert len(set(subX)) == len(subX)
    assert all(Y[subX] == subY)


@pytest.mark.parametrize("N", [0.1, 0.19, 0.81, 0.9])
def test_rechaza_N_fuera_de_rango(N):
    with pytest.raises(ValueError):
        submuestreo([1, 2], ["a", "b"], N)


def test_reproducible():
    Y = np.array(["a"] * 50 + ["b"] * 10)
    X = np.arange(60)
    assert (submuestreo(X, Y, 0.3, semilla=7)[0] == submuestreo(X, Y, 0.3, semilla=7)[0]).all()


def test_conserva_clase_minoritaria():
    Y = np.array(["a"] * 50 + ["b"] * 2)
    _, subY = submuestreo(np.arange(52), Y, 0.2, semilla=1)
    assert "b" in subY


@pytest.mark.parametrize("por_paciente", [True, False])
def test_folds_disjuntos(datos, por_paciente):
    particion = particion_kfold(datos, 5, semilla=0, por_paciente=por_paciente)
    todos_test = np.concatenate([te for _, te in particion])
    assert sorted(todos_test) == list(range(len(datos)))
    for tr, te in particion:
        assert not set(tr) & set(te)
        assert set(datos["etiqueta"].values[te]) == {"benign", "malignant"}
        subX, _ = submuestreo(datos["ruta"].values[tr], datos["etiqueta"].values[tr], 0.3, semilla=0)
        assert not set(subX) & set(datos["ruta"].values[te])
        if por_paciente:
            assert not set(datos["paciente"].values[tr]) & set(datos["paciente"].values[te])
