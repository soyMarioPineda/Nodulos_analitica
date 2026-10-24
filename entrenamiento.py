import copy

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

MEDIA_IMAGENET = [0.485, 0.456, 0.406]
DESV_IMAGENET = [0.229, 0.224, 0.225]


class ImagenesNodulos(Dataset):
    def __init__(self, rutas, etiquetas, clases, tam=224):
        self.rutas = list(rutas)
        self.y = [clases.index(e) for e in etiquetas]
        self.transform = transforms.Compose([
            transforms.Resize((tam, tam)),
            transforms.ToTensor(),
            transforms.Normalize(MEDIA_IMAGENET, DESV_IMAGENET),
        ])

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, i):
        imagen = Image.open(self.rutas[i]).convert("RGB")   # ultrasonido en gris -> 3 canales
        return self.transform(imagen), self.y[i]


# 1. Red preentrenada con la ultima capa cambiada al numero de clases
#    (como en el script del profesor: dropout 0.25 y capa nueva con lr x5)
def crear_modelo(nombre, num_clases, preentrenado=True):
    if nombre == "vgg19":
        red = models.vgg19(weights="IMAGENET1K_V1" if preentrenado else None)
        red.classifier[5] = nn.Dropout(0.25)
        red.classifier[6] = nn.Linear(4096, num_clases)
        cabeza = red.classifier[6]
    elif nombre == "resnet18":
        red = models.resnet18(weights="IMAGENET1K_V1" if preentrenado else None)
        red.fc = nn.Sequential(nn.Dropout(0.25), nn.Linear(red.fc.in_features, num_clases))
        cabeza = red.fc
    else:
        raise ValueError(f"Modelo no soportado: {nombre}")
    return red, cabeza


# 2. Validacion separada del entrenamiento (por paciente) para el early stopping.
#    El conjunto de prueba NO se usa para elegir el modelo.
def separar_validacion(rutas, etiquetas, pacientes, fraccion, semilla):
    n_splits = max(2, round(1 / fraccion))
    cvp = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=semilla)
    idx_tr, idx_val = next(cvp.split(rutas, etiquetas, groups=pacientes))
    return idx_tr, idx_val


def perdida_promedio(red, cargador, criterio, dispositivo):
    red.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in cargador:
            x, y = x.to(dispositivo), y.to(dispositivo)
            total += criterio(red(x), y).item() * len(y)
            n += len(y)
    return total / n


def predecir(red, cargador, dispositivo):
    red.eval()
    pred = []
    with torch.no_grad():
        for x, _ in cargador:
            pred.append(red(x.to(dispositivo)).argmax(1).cpu())
    return torch.cat(pred).numpy()


# 3. Entrena con el submuestreo y evalua en el conjunto de prueba del fold
def entrenar_y_evaluar(subXTr, subYTr, pacientesTr, XTe, YTe, clases, args, semilla):
    torch.manual_seed(semilla)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

    idx_tr, idx_val = separar_validacion(subXTr, subYTr, pacientesTr, args.val, semilla)
    ds_tr = ImagenesNodulos(subXTr[idx_tr], subYTr[idx_tr], clases)
    ds_val = ImagenesNodulos(subXTr[idx_val], subYTr[idx_val], clases)
    ds_te = ImagenesNodulos(XTe, YTe, clases)
    cargar = dict(batch_size=args.batch, num_workers=args.workers)
    dl_tr = DataLoader(ds_tr, shuffle=True, **cargar)
    dl_val = DataLoader(ds_val, **cargar)
    dl_te = DataLoader(ds_te, **cargar)

    # Pesos de clase calculados sobre el entrenamiento de ESTE fold: N_total / (K * n_clase)
    conteo = np.bincount(ds_tr.y, minlength=len(clases))
    pesos = len(ds_tr) / (len(clases) * np.maximum(conteo, 1))
    criterio = nn.CrossEntropyLoss(weight=torch.tensor(pesos, dtype=torch.float32, device=dispositivo))

    red, cabeza = crear_modelo(args.modelo, len(clases), not args.sin_preentrenado)
    red.to(dispositivo)
    ids_cabeza = {id(p) for p in cabeza.parameters()}
    resto = [p for p in red.parameters() if id(p) not in ids_cabeza]
    optimizador = torch.optim.Adam([
        {"params": resto, "lr": args.lr},
        {"params": cabeza.parameters(), "lr": args.lr * 5},
    ])

    mejor_perdida, mejor_estado, sin_mejora = float("inf"), None, 0
    for epoca in range(1, args.epocas + 1):
        red.train()
        for x, y in dl_tr:
            x, y = x.to(dispositivo), y.to(dispositivo)
            optimizador.zero_grad()
            criterio(red(x), y).backward()
            optimizador.step()

        perdida_val = perdida_promedio(red, dl_val, criterio, dispositivo)
        print(f"    epoca {epoca}: perdida validacion = {perdida_val:.4f}")
        if perdida_val < mejor_perdida:
            mejor_perdida, mejor_estado, sin_mejora = perdida_val, copy.deepcopy(red.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= args.paciencia:
                break

    red.load_state_dict(mejor_estado)
    y_pred = predecir(red, dl_te, dispositivo)
    y_real = np.array(ds_te.y)
    return float(np.mean(y_pred == y_real)), confusion_matrix(y_real, y_pred, labels=range(len(clases)))
