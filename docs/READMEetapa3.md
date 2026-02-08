# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Velcea Alexandra 
**Data:** Decembrie 2025  

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului „Rețele Neuronale". Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN, respectând bunele practici privind calitatea, consistența și reproductibilitatea datelor.

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
project-name/
├── README.md
├── docs/
│   └── datasets/          # descriere seturi de date, surse, diagrame
├── data/
│   ├── raw/               # date brute
│   │    ├── train/        # date antrenare (imagini + adnotări)
│   │    └── validation/   # date validare (imagini + adnotări)
│   ├── processed/         # date curățate și transformate
│   ├── train/             # set de instruire
│   ├── validation/        # set de validare
│   └── test/              # set de testare
├── src/
│   ├── preprocessing/     # funcții pentru preprocesare
│   ├── data_acquisition/  # generare / achiziție date (dacă există)
│   └── neural_network/    # implementarea RN (în etapa următoare)
├── config/                # fișiere de configurare
├── requirements.txt       # dependențe Python (dacă aplicabil)
└── README.md              # readme
```

---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Set de date public NEU-DET (https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
* **Modul de achiziție:** Generare programatică
* **Perioada / condițiile colectării:** Decembrie 2025

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** aprox. 1100 imagini
* **Număr de caracteristici (features):** Imagine RGB (3 canale) + Adnotări Bounding Box
* **Tipuri de date:** Imagini + Text (adnotări)
* **Format fișiere:** JPG, XML

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
| --- | --- | --- | --- | --- |
| **Image Width** | numeric | pixeli | Lățimea imaginii redimensionate. | Fix: **200** |
| **Image Height** | numeric | pixeli | Înălțimea imaginii redimensionate. | Fix: **200** |
| **Defect Class** | categorial | - | Tipul defectului (Target). | `{crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches}` |
| **BBox X_center** | numeric | normalizat | Coordonata X a centrului defectului. | **0.0 -- 1.0** |
| **BBox Y_center** | numeric | normalizat | Coordonata Y a centrului defectului. | **0.0 -- 1.0** |
| **BBox Width** | numeric | normalizat | Lățimea relativă a defectului. | **0.0 -- 1.0** |
| **BBox Height** | numeric | normalizat | Înălțimea relativă a defectului. | **0.0 -- 1.0** |

**Fișier recomandat:**  `docs/README.md`

---

## 3. Analiza Exploratorie a Datelor (EDA)

### 3.1 Statistici descriptive aplicate

-   **Distribuția claselor:** Inițial echilibrată în setul raw/.
-   **Dimensiuni Bounding Box:**
    -   *Inclusion/Scratches:* Arii mici, necesită ancore fine.
    -   *Patches:* Arii mari, ocupă uneori >30% din imagine.
-   **Intensitatea pixelilor:** Imaginile sunt grayscale, având o distribuție a luminozității concentrată în zona medie (gri metalic).

### 3.2 Analiza calității datelor

-   **Inconsistențe:** Unele fișiere XML aveau coordonate `xmin > xmax`. Scriptul de conversie a tratat aceste erori prin clampare la dimensiunile imaginii.

### 3.3 Probleme identificate

-   **Rezoluție mică:** Redimensionarea la 200x200 (pentru viteză) face dificilă detectarea defectelor de tip *Crazing* (fisuri fine).
-   **Lipsa datelor de validare:** Setul original nu avea un split explicit. S-a creat unul manual.
-   **Overlap:** Defecte multiple suprapuse în aceeași imagine (ex: *Pitted Surface* + *Scratches*)

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

-   **Tratarea outlierilor:** Coordonatele bounding box care ieșeau din cadrul imaginii au fost trunchiate la [0, width/height].

### 4.2 Transformarea caracteristicilor

-   **Normalizare:** Coordonatele pixelilor au fost scalate în intervalul [0, 1] (cerință YOLO).
-   **Conversie Format:** Transformarea adnotărilor din **Pascal VOC (XML)** în **YOLO (TXT)**:
    -   Formula: $x_{center} = \frac{(x_{min} + x_{max})}{2 \cdot width}$

-   **Augmentare Sintetică:**
    -   S-au generat defecte specifice (ex: desenare linii pentru *scratches*, elipse pentru *patches*) peste imaginile de antrenament folosind scriptul `generate_augmentation.py`.
    -   Ajustări de luminozitate și contrast (Random Brightness/Contrast).

### 4.3 Structurarea seturilor de date

**Împărțire realizată (Stratificată):**

-   **Train (54%):** 105 imagini/clasă (Doar date augmentate).
-   **Validation (23%):** 45 imagini/clasă (Date originale).
-   **Test (23%):** 45 imagini/clasă (Date originale).

**Principii respectate:**

-   **Stratificare:** Fiecare clasă are exact același număr de exemple.
-   **Fără scurgere de informație (No Data Leakage):** Imaginile originale mutate în Test/Validation NU au fost folosite pentru generarea augmentărilor din data/train.

### 4.4 Salvarea rezultatelor preprocesării

-   Structura finală salvată în `data/train`, `data/validation`, `data/test`.
-   Configurația dataset-ului salvată în `data/data.yaml` pentru YOLOv8.

---

##  5. Fișiere Generate în Această Etapă

* `data/raw/` – date brute
* `data/processed/` – date curățate & transformate
* `data/train/`, `data/validation/`, `data/test/` – seturi finale
* `src/preprocessing/` – codul de preprocesare
* `docs/README.md` – descrierea dataset-ului

---

##  6. Stare Etapă (de completat de student)

- [x] Structură repository configurată
- [x] Dataset analizat (EDA realizată)
- [x] Date preprocesate
- [x] Seturi train/val/test generate
- [x] Documentație actualizată în README + `docs/README.md`

---
