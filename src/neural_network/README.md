### Script de Antrenare Optimizată (`train.py`)
==============================================

Acest script (`train.py`) este motorul principal al procesului de învățare automată. Spre deosebire de o antrenare standard YOLO, această configurație ("Ultimate") este calibrată specific pentru **detectarea defectelor fine** pe suprafețe metalice, utilizând modelul **YOLOv8 Medium**.

## 1. De ce "Antrenare Optimizată"?
--------------------------------

Scriptul implementează o serie de tehnici avansate pentru a depăși limitările antrenării standard:

1.  **Upgrade la YOLOv8 Medium (`yolov8m.pt`):**
    -   Trecerea de la varianta *Nano* la *Medium* oferă o capacitate mult mai mare de extragere a trăsăturilor, esențială pentru defecte subtile precum *crazing* sau *pitted_surface*.
2.  **Rezoluție Înaltă (HD - 832px):**
    -   Antrenarea se face la **832x832 pixeli** (standardul este 640). Acest lucru permite rețelei să "vadă" defecte mici care s-ar pierde prin redimensionare (pixelare).
3.  **Optimizator AdamW:**
    -   Folosește algoritmul `AdamW` cu `Cosine Annealing` (`cos_lr=True`) pentru o convergență mai stabilă și o generalizare mai bună pe finalul antrenamentului.

## 2. Hiperparametri Cheie
-----------------------

| **Parametru** | **Valoare** | **Explicație** |
| --- | --- | --- |
| **Model** | `yolov8m.pt` | Balans optim între precizie și viteză (Medium). |
| **Epoci** | `150` | Durată extinsă pentru a permite modelului mai complex să conveargă. |
| **Patience** | `40` | Early Stopping: Antrenamentul se oprește dacă nu există îmbunătățiri timp de 40 de epoci (previne overfitting). |
| **Image Size** | `832` | Rezoluție crescută pentru detalii fine. |
| **Batch Size** | `16` | Optimizat pentru GPU-uri cu ~12-16GB VRAM. |
| **Optimizer** | `AdamW` | Gestionează mai bine datele "zgomotoase" decât SGD. |
| **Learning Rate** | `0.0005` | Rată mică de învățare (`lr0`) pentru stabilitate. |

## 3. Strategia de Augmentare (Data Augmentation)
-----------------------------------------------

Pentru a combate numărul limitat de date și a crește robustețea, scriptul activează augmentări agresive **în timpul antrenamentului** (on-the-fly):
-   **Mosaic (`1.0`):** Combină 4 imagini într-una singură (standard YOLO).
-   **MixUp (`0.15`):** Suprapune două imagini cu transparență (15% șansă). Ajută modelul să nu memoreze formele exacte.
-   **Copy-Paste (`0.3`):** Decupează defecte dintr-o imagine și le lipește în alta (30% șansă). Crește varietatea instanțelor de defecte.
-   **Degrees (`10.0`):** Rotește ușor imaginile (+/- 10 grade) pentru a simula orientarea variabilă a benzii de oțel.
-   **Flip (`UD/LR`):** Răsturnare pe verticală și orizontală (defectele nu au o orientare "corectă").

## 4. Cerințe de Sistem
--------------------

Datorită rezoluției de 832px și a modelului Medium, acest script necesită resurse hardware semnificative:
-   **GPU:** NVIDIA cu minim **12 GB VRAM** (recomandat 16 GB, ex: Tesla T4, RTX 3060/4070).
-   **RAM:** Minim 16 GB.

## 5. Utilizare
-------------

Asigurați-vă că sunteți în rădăcina proiectului și că ați instalat dependențele (`ultralytics`).

bash

```
python src/training/train.py

```

## 6. Output (Rezultate)
---------------------

Rezultatele antrenamentului vor fi salvate automat în directorul `models/defect_detector_ult/`. Structura va fi:
-   `weights/best.pt`: Cel mai bun model (folosit pentru inferență).
-   `weights/last.pt`: Ultimul checkpoint salvat.
-   `results.csv`: Log-urile detaliate ale antrenamentului (Loss, mAP).
-   `confusion_matrix.png`: Matricea de confuzie finală.
-   `val_batch*_pred.jpg`: Exemple vizuale de predicții pe setul de validare.

## 7. Alte scripturi

- `detection.py`: Script pentru inferență în stadiu inițial;
- `evaluate.py`: Script pentru generare `docs/results/loss_curve.png` și `docs/results/confusion_matrix.png`;
- `gen_comparison_metrics.py`: Script pentru generare `docs/results/metrics_evolution.png`;
- `gen_opt_experiments.py`: Script pentru generare `results/optimization_experiments.csv`;
- `gen_plots.py`: Script pentru generare `docs/optimization/f1_comparison.png` și `docs/optimization/accuracy_comparison.png`;
- `test_metrics.py`: Script pentru generarea `results/test_metrics.json`;
- `get_error_analysis.py`: Script pentru generarea `error_analysis.py`.