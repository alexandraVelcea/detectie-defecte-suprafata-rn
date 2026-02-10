# detectie-defecte-suprafata-rn

# Sursă set de date: NEU-DET https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

# Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale
**Instituție:** POLITEHNICA București – FIIR
**Student:** Velcea Alexandra
**Link Repository GitHub:** https://github.com/alexandraVelcea/detectie-defecte-suprafata-rn
**Data:** 05.12.2024

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
│   ├── processed/         # date curățate și transformate
│   ├── train/             # set de instruire
│   ├── validation/        # set de validare
│   └── test/              # set de testare
├── src/
│   ├── preprocessing/     # funcții pentru preprocesare
│   ├── data_acquisition/  # generare / achiziție date (dacă există)
│   └── neural_network/    # implementarea RN (în etapa următoare)
├── config/                # fișiere de configurare
└── requirements.txt       # dependențe Python (dacă aplicabil)
```

---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Datele sunt preluate din setul de date public NEU-DET, respectiv generate folosind librăria Pillow.
* **Modul de achiziție:** Generare programatică
* **Perioada / condițiile colectării:** Decembrie 2025

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** 1800 
* **Număr de caracteristici (features):** 6
* **Tipuri de date:** Imagini
* **Format fișiere:** PNG

### Tabelul Caracteristicilor Dataset-ului NEU-DET

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|:---|:---|:---|:---|:---|
| **Image Resolution** | numeric | pixeli | Dimensiunea spațială a imaginii de input (lățime × înălțime). | Fix: **200 × 200** |
| **Pixel Intensity** | numeric | – | Valoarea de intensitate a fiecărui pixel (Grayscale), reprezentând luminozitatea suprafeței metalice. | **0 – 255** (8-bit integer) |
| **Defect Class** | categorial | – | Tipul defectului identificat pe suprafața metalică (Target Variable). | **{crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches}** |
| **BBox_Center_X** | numeric | pixeli (sau norm) | Coordonata orizontală a centrului defectului detectat. | **0 – 200** (sau 0.0 – 1.0 normalizat) |
| **BBox_Center_Y** | numeric | pixeli (sau norm) | Coordonata verticală a centrului defectului detectat. | **0 – 200** (sau 0.0 – 1.0 normalizat) |
| **BBox_Width** | numeric | pixeli (sau norm) | Lățimea dreptunghiului care încadrează defectul (bounding box). | **0 – 200** |
| **BBox_Height** | numeric | pixeli (sau norm) | Înălțimea dreptunghiului care încadrează defectul. | **0 – 200** |

**Fișier:**  `docs/README.md`

---

### 3.1 Statistici descriptive aplicate

Pentru a înțelege natura datelor vizuale, s-au calculat următoarele statistici pe setul de date brut și cel augmentat:

-   **Distribuția Claselor:**

    -   *Inițial (Raw):* Dezechilibru ușor între clase (variații între 250-300 imagini/clasă).
    -   *Final (După preprocesare):* **Distribuție perfect uniformă** (195 imagini pe fiecare clasă: 105 Train + 45 Val + 45 Test), eliminând bias-ul modelului către clasele majoritare.

-   **Analiza Bounding Box-urilor (Ancore):**

    -   **Dimensiuni:** S-a observat o varianță mare a ariei defectelor.
        -   *Inclusion/Scratches:* Ocupă în medie <5% din imagine (obiecte mici).
        -   *Patches:* Ocupă adesea >30% din imagine (obiecte mari).

    -   **Raport de aspect (Aspect Ratio):** *Scratches* au un raport extrem (foarte late și scunde sau înalte și înguste), în timp ce *Patches* tind spre 1:1 (pătrat).

-   **Intensitatea Pixelilor:**
    -   **Medie:** ~115 (pe o scară 0-255), indicând o luminozitate medie specifică oțelului gri.
    -   **Deviație Standard:** Scăzută în zonele fără defecte, ridicată în zonele cu *Rolled-in Scale* (contrast puternic).

### 3.2 Analiza calității datelor

-   **Detectarea valorilor lipsă:**
    -   S-au identificat **0% valori lipsă** în setul final.

-   **Inconsistențe geometrice:**
    -   S-au identificat coordonate de bounding box care depășeau dimensiunile imaginii (ex: `xmax > 200`). Acestea au fost corectate prin "clamping" la marginile imaginii în timpul conversiei XML-to-YOLO.

-   **Caracteristici redundante:**
    -   Informația de culoare (Hue/Saturation) este redundantă, deoarece defectele sunt texturale. Modelul se bazează preponderent pe canalul de Luminanță (Grayscale).

### 3.3 Probleme identificate și Soluții
-   **Problema 1: Rezoluția mică (200x200 px)**
    -   *Impact:* Defectele de tip **Crazing** (micro-fisuri) devin aproape invizibile la redimensionare, riscând să fie pierdute la convoluție.
    -   *Soluție:* S-a aplicat augmentare prin creșterea contrastului pentru a accentua liniile fine.

-   **Problema 2: Confuzii vizuale (Inter-class similarity)**
    -   *Impact:* Clasele **Rolled-in Scale** și **Pitted Surface** au texturi foarte similare (puncte negre), ceea ce poate duce la erori de clasificare.
    -   *Soluție:* Antrenarea modelului pentru mai multe epoci (100) pentru a învăța trăsături mai subtile.

-   **Problema 3: Variabilitate scăzută a fundalului**
    -   *Impact:* Modelul risca să memoreze fundalul specific NEU-DET în loc să învețe defectul.
    -   *Soluție:* Generarea de date sintetice (`generate_augmentation.py`) a introdus variații artificiale de zgomot și luminozitate pentru a forța generalizarea.

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

   * Tratarea outlierilor: Excluderea adnotărilor cu arie < 10px².

### 4.2 Transformarea caracteristicilor

   * Normalizare: Scalare valori pixeli [0, 255] -> [0, 1].
   * Encoding: Conversie etichete text (ex: "crazing") în ID numeric (0-5) pentru YOLO.
   * Formatare: Conversie coordonate Pascal VOC (xmin, ymin, xmax, ymax) -> YOLO (x_center, y_center, width, height).

### 4.3 Structurarea seturilor de date

**Împărțire recomandată:**
* 70–80% – train (6 x 210 imagini)
* 10–15% – validation (6 x 45 imagini)
* 10–15% – test (6 x 45 imagini)

**Principii respectate:**
* Stratificare pentru clasificare
* Fără scurgere de informație (data leakage)
* Statistici calculate DOAR pe train și aplicate pe celelalte seturi

### 4.4 Salvarea rezultatelor preprocesării

* Date preprocesate în `data/processed/`
* Seturi train/validation/test în foldere dedicate

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