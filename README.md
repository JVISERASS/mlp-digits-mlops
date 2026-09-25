# MLP digits · DVC + MLflow (DagsHub)

MLP pequeño en PyTorch que clasifica el dataset `digits` de scikit-learn (1797 imágenes 8x8, 10 clases).

- **Código:** GitHub → https://github.com/JVISERASS/mlp-digits-mlops
- **Datos:** `data/digits.csv`, versionado con DVC en el bucket S3 de DagsHub (`s3://mlp-digits-mlops/dvc`)
- **Experimentos:** MLflow de DagsHub → https://dagshub.com/JVISERASS/mlp-digits-mlops.mlflow

## Uso

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# credenciales DVC (solo en local, no se suben a git)
dvc remote modify dagshub-s3 --local access_key_id <TOKEN>
dvc remote modify dagshub-s3 --local secret_access_key <TOKEN>
dvc pull

export MLFLOW_TRACKING_URI=https://dagshub.com/JVISERASS/mlp-digits-mlops.mlflow
export MLFLOW_TRACKING_USERNAME=JVISERASS
export MLFLOW_TRACKING_PASSWORD=<TOKEN>
python train.py --hidden 128 64 --dropout 0.2 --epochs 40
```

Cada run registra lo siguiente:

- hiperparámetros y número de parámetros del modelo;
- `train_loss`, `val_loss` y `val_accuracy` por época;
- `test_accuracy` y `test_f1_macro`;
- la matriz de confusión y el modelo;
- el tag `data_md5`, que enlaza el run con la versión DVC del dataset.

Split: 64 % train, 16 % val y 20 % test (360 muestras), estratificado y con seed 42.

## Análisis de resultados

| Run | Capas ocultas | Params | lr | Dropout | val_acc (época 5) | val_loss mín. (época) | val_loss final | Test acc | Test F1 |
|---|---|---|---|---|---|---|---|---|---|
| A | 16 | 1 210 | 1e-3 | 0 | 0.757 | 0.142 (39) | 0.142 | 0.9583 | 0.9577 |
| B | 64 | 4 810 | 1e-3 | 0 | 0.882 | 0.082 (39) | 0.082 | 0.9667 | 0.9663 |
| C | 128-64 | 17 226 | 1e-3 | 0.2 | 0.951 | **0.050 (27)** | 0.070 | **0.9750** | **0.9747** |
| D | 64 | 4 810 | 5e-2 | 0 | 0.948 | 0.414 (1) | **10.27** | 0.9667 | 0.9665 |

**Conclusiones**

1. **Capacidad.** Más neuronas dan mejor resultado y convergen antes. A (16) sigue aprendiendo en la época 40 (loss aún bajando) y queda corto de capacidad. B (64) mejora, pero tampoco ha convergido del todo en 40 épocas.
2. **El mejor modelo es C (dos capas y dropout 0.2).** Tiene la mejor accuracy/F1 en test (0.975) y la val_loss más baja. Su val_loss toca mínimo en la época 27 y luego sube un poco, lo que indica un inicio de sobreajuste. Con *early stopping* se quedaría con el modelo de esa época.
3. **Un learning rate alto (D, lr=0.05) es inestable.** Llega rápido a buena accuracy, pero la val_loss se dispara hasta 10.3. El modelo acierta la clase, pero con confianza extrema cuando falla. La accuracy sola ocultaría el problema; por eso conviene registrar también la loss.
4. **Cuidado con las diferencias pequeñas.** El test tiene 360 muestras: 0.0083 de accuracy son unos 3 ejemplos. Para afirmar que C es mejor que B habría que repetir con varias seeds o usar validación cruzada.
5. **Reproducibilidad.** Los 4 runs tienen el mismo `data_md5` (`0df1043d`), así que las diferencias se deben solo a los hiperparámetros. Con `git checkout <commit> && dvc pull` se recupera el código y los datos exactos de cualquier run.

**Siguientes pasos:** early stopping sobre `val_loss`, un lr intermedio (~5e-3) con scheduler, y registrar el mejor modelo en el Model Registry de MLflow.
