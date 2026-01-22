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

* **Număr total de observații:** 
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


# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN**.
Trebuie să livrați un SCHELET COMPLET și FUNCȚIONAL al întregului Sistem cu Inteligență Artificială (SIA).

---

## Livrabile obligatorii

### 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
| --------------------------- | --------------------------------- | -------------------------------- |
| Detectarea automată a defectelor (zgârieturi, rugină) pe suprafețe metalice industriale | Analiză vizuală automată și localizare defecte (bounding box) cu acuratețe > 85% | RN Module (YOLOv8) + UI |
| Generarea de date de antrenament variate pentru scenarii rare (defecte specifice) | Generare imagini sintetice fotorealiste folosind AI Generativ (Imagen 3) | Data Acquisition (GenAI Script) |
| Alertarea operatorului în timp real la detectarea unui defect critic | Interfață vizuală cu marcare roșie a defectelor și timp de răspuns < 2s | Web Service / UI |

---

### 2. Contribuția voastră originală la setul de date – MINIM 40% din totalul observațiilor finale

### Contribuția originală la setul de date:

| **Tip contribuție** | **Exemple concrete din inginerie** | **Dovada minimă cerută** |
|---------------------|-------------------------------------|--------------------------|
| **Etichetare/adnotare manuală** | • Etichetat manual 100+ imagini defecte sudură | Imagini .png cu adnotări în fișiere xml |

**Total observații finale:** ~250 imagini (estimat pentru final)
**Observații originale:** ~100+ imagini (40%+)

**Tipul contribuției:**
[ ] Date generate prin simulare fizică
[ ] Date achiziționate cu senzori proprii
[ ] Etichetare/adnotare manuală
[x] **Date sintetice generate prin Pillow**

**Descriere detaliată:**
Pentru a compensa lipsa de diversitate în dataset-urile publice (precum NEU-DET), am dezvoltat un modul de generare a datelor sintetice folosind librăria Python **Pillow**.

Scriptul Python `generate_augmentation.py` utilizează librăria Pillow pentru a augmenta o parte din imaginile deja existente în setul de date. Aceste imagini sunt salvate automat, verificate și vor fi integrate în pipeline-ul de antrenare alături de datele reale. Această abordare permite simularea unor scenarii de iluminare și texturi dificil de capturat în mediul real fără echipament costisitor.

**Locația codului:** `src/data_acquisition/generate_data.py`
**Locația datelor:** `data/train/`

**Dovezi:**

- Grafic comparativ: `docs/generated_vs_real.png`
- Setup experimental: `docs/acquisition_setup.jpg` (dacă aplicabil)
- Tabel statistici: `docs/data_statistics.csv`

---

### 3. Diagrama State Machine a întregului sistem

**Locație fișier:** `docs/state_machine.png`

### Justificarea State Machine-ului ales:

Am ales arhitectura de tip **Clasificare/Detecție la cerere**, specifică sistemelor de controlul calitatății de pe liniile de producție.

**Stările principale sunt:**
1. **IDLE:** Sistemul așteaptă input (încărcare imagine de către operator).
2. **ACQUIRE_DATA:** Se încarcă imaginea (reală sau generată) și se verifică integritatea fișierului.
3. **PREPROCESS:** Ștergerea datelor de la ultima rulare.
4. **IS_VALID:** Verificarea vlidității datelor.
5. **INFERENCE (RN):** Modelul YOLOv8 procesează imaginea pentru a identifica coordonatele defectelor.
6. **DEFECT_NOT_FOUND:** Nu a fost găsit un defect.
7. **CLASSIFY_DEFECT:** Clasifică defectul.
8. **INVALID:** Imaginea nu este validă; se revine în starea IDLE.
9. **GENERATE RESULTS:** Se generează rezultate statistice. 

**Tranziția critică sunt:**
- **INVALID → IDLE:** Dacă imaginea este coruptă sau formatul nu este suportat.

---

### 4. Scheletul complet al celor 3 Module

| **Modul** | **Tehnologie** | **Status Etapa 4** |
|-----------|----------------|--------------------|
| **1. Data Acquisition** | Python (`google-genai`, `PIL`) | **Funcțional.** Scriptul se conectează la API, generează imagini pe baza prompt-ului și le salvează local cu timestamp. |
| **2. Neural Network** | Python (`ultralytics` YOLOv8) | **Funcțional.** Arhitectura este definită (YOLOv8n), fișierul de config `data.yaml` este creat, antrenamentul poate fi inițiat. |
| **3. Web Service / UI** | Python (`matplotlib`/`opencv` sau Streamlit) | **Funcțional.** Script de inferență care ia o imagine, rulează modelul și afișează rezultatul cu bounding boxes. |

**Total observații finale:** ~1170 imagini

**Observații originale (Sintetice):** 630 imagini (6 clase × 105 imagini) -> **~54% din total**

**Tipul contribuției:**

[ ] Date generate prin simulare fizică
[ ] Date achiziționate cu senzori proprii
[ ] Etichetare/adnotare manuală
[x] **Date sintetice generate prin Pillow (Scripting)**

**Descriere detaliată:**

Deoarece defectele industriale reale sunt costisitor de colectat, am dezvoltat un modul software propriu (`generate_augmentation.py`) care simulează defecte.

Scriptul ia imagini "curate" sau cu defecte minore și desenează algoritmic noi defecte:
* **Scratches:** Linii Bezier aleatorii cu variații de culoare.
* **Patches/Inclusions:** Elipse și forme neregulate cu textură modificată.
* **Crazing:** Rețele de linii fine interconectate.

**Locația codului:** `src/data_acquisition/generate_augmentation.py`
**Locația datelor:** `data/train/images` (imaginile cu prefixul `aug_`)

**Dovezi:**
- Scriptul sursă: `src/data_acquisition/generate_augmentation.py`
- Fișiere XML generate automat care corespund noilor defecte.

---

### 3. Diagrama State Machine a întregului sistem

**Locație fișier:** `docs/state_machine.png`

### Justificarea State Machine-ului ales:

Am ales o arhitectură orientată pe evenimente (Event-Driven), specifică aplicațiilor de monitorizare industrială.

**Stările principale sunt:**

1\.  **IDLE:** Sistemul așteaptă input (încărcare fișier).
2\.  **ACQUIRE_DATA:** Se citește imaginea brută.
3\.  **IS_VALID:** Verificare format (JPG/PNG) și dimensiuni minime. Dacă invalid -> `INVALID`.
4\.  **PREPROCESS:** Redimensionare la 200x200px, normalizare pixeli (0-1).
5\.  **INFERENCE (RN):** Rularea modelului YOLOv8 pre-antrenat.
6\.  **CHECK_DETECTIONS:** Se verifică dacă există bounding boxes cu confidence > Threshold.
    * *Dacă DA* -> `CLASSIFY_DEFECT`
    * *Dacă NU* -> `DEFECT_NOT_FOUND`
7\.  **CLASSIFY_DEFECT:** Identificarea tipului (ex: Crazing) și desenarea conturului.
8\.  **GENERATE_RESULTS:** Afișare imagine marcată în UI și salvare log CSV.
9\.  **INVALID:** Afișare eroare utilizator și revenire la IDLE.

---

## Structura Repository-ului la Finalul Etapei 4

```text
detectie-defecte-suprafata/
├── data/
|   ├── processed/
    |   ├── train
    |   ├── validation
│   ├── raw/                  # Dataset NEU-DET original
│   ├── generated/            # Imagini create cu scriptul GenAI (Contribuție proprie)
│   └── data.yaml             # Configurare pentru YOLO
├── src/
│   ├── data_acquisition/
│   │   ├── generate_data.py  # Scriptul de generare imagini (Modul 1)
│   │   └── check_models.py   # Utilitar verificare API
│   └── neural_network/
│       ├── train_yolo.py     # Script antrenament (Modul 2)
│       ├── detect.py         # Script inferență
│       └── main.py           # Entry point aplicatie (Modul 3)
├── docs/
│   ├── README.md             # Fișier README
│   └── state_machine.png     # Diagrama stărilor
├── models/
│   └── yolov8n.pt            # Modelul (pre-trained sau fine-tuned)
├── requirements.txt          # Dependențe (ultralytics, google-genai, pillow)
└── .env                      # API Keys (ignorat de git)



# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Antrenarea efectivă a modelului RN definit în Etapa 4, evaluarea performanței și integrarea în aplicația completă.

**Pornire obligatorie:** Arhitectura completă și funcțională din Etapa 4:

- State Machine definit și justificat
- Cele 3 module funcționale (Data Logging, RN, UI)
- Minimum 40% date originale în dataset

---

## PREREQUISITE – Verificare Etapa 4 (OBLIGATORIU)

**Înainte de a începe Etapa 5, verificați că aveți din Etapa 4:**

- [x] **State Machine** definit și documentat în `docs/state_machine.*`
- [x] **Contribuție ≥40% date originale** în `data/generated/` (verificabil)
- [x] **Modul 1 (Data Logging)** funcțional - produce CSV-uri
- [x] **Modul 2 (RN)** cu arhitectură definită dar NEANTRENATĂ (`models/untrained_model.h5`)
- [x] **Modul 3 (UI/Web Service)** funcțional cu model dummy
- [x] **Tabelul "Nevoie → Soluție → Modul"** complet în README Etapa 4

** Dacă oricare din punctele de mai sus lipsește → reveniți la Etapa 4 înainte de a continua.**

---

## Pregătire Date pentru Antrenare

### Dacă ați adăugat date noi în Etapa 4 (contribuția de 40%):

**TREBUIE să refaceți preprocesarea pe dataset-ul COMBINAT:**
Exemplu:

```bash

# 1. Combinare date vechi (Etapa 3) + noi (Etapa 4)
python  src/preprocessing/combine_datasets.py

# 2. Refacere preprocesare COMPLETĂ
python  src/preprocessing/data_cleaner.py
python  src/preprocessing/feature_engineering.py
python  src/preprocessing/data_splitter.py  --stratify  --random_state  42

# Verificare finală:
# data/train/ → trebuie să conțină date vechi + noi
# data/validation/ → trebuie să conțină date vechi + noi
# data/test/ → trebuie să conțină date vechi + noi

```

** ATENȚIE - Folosiți ACEIAȘI parametri de preprocesare:**

- Același `scaler` salvat în `config/preprocessing_params.pkl`
- Aceiași proporții split: 70% train / 15% validation / 15% test
- Același `random_state=42` pentru reproducibilitate

---

## Cerințe Structurate pe 3 Niveluri

### Nivel 1 – Obligatoriu pentru Toți (70% din punctaj)

Completați **TOATE** punctele următoare:

1.  **Antrenare model** definit în Etapa 4 pe setul final de date (≥40% originale)
2.  **Minimum 10 epoci**, batch size 8–32
3.  **Împărțire stratificată** train/validation/test: 70% / 15% / 15%
4.  **Tabel justificare hiperparametri** (vezi secțiunea de mai jos - OBLIGATORIU)
5.  **Metrici calculate pe test set:**
-  **Acuratețe ≥ 65%**
-  **F1-score (macro) ≥ 0.60**
6.  **Salvare model antrenat** în `models/trained_model.h5` (Keras/TensorFlow) sau `.pt` (PyTorch) sau `.lvmodel` (LabVIEW)
7.  **Integrare în UI din Etapa 4:**

- UI trebuie să încarce modelul ANTRENAT (nu dummy)
- Inferență REALĂ demonstrată
- Screenshot în `docs/screenshots/inference_real.png`

#### Tabel Hiperparametri și Justificări (OBLIGATORIU - Nivel 1)

Completați tabelul cu hiperparametrii folosiți și **justificați fiecare alegere**:

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
| --- | --- | --- |
| **Learning rate** | `0.01` (inițial) | Valoare standard pentru optimizatorul SGD în YOLO. Este suficient de mare pentru a ieși din minime locale la început, scăzând treptat (Scheduler Cosine) pentru fine-tuning. |
| **Batch size** | `2` | Compromis pentru hardware limitat (laptop/CPU). |
| **Number of epochs** | `100` | Dataset-ul fiind relativ mic, modelul are nevoie de mai multe iterații pentru a converge. Folosim **Early Stopping** (patience=15) pentru a opri procesul automat dacă apare overfitting. |
| **Optimizer** | `SGD` (cu Momentum) | Stochastic Gradient Descent (cu momentum 0.937) este standardul pentru antrenarea YOLO, oferind o generalizare mai bună pe imagini decât Adam. |
| **Loss function** | `CIoU` (Box) + `BCE` (Cls) | YOLO folosește o funcție compusă: **CIoU** pentru localizarea geometrică a defectului și **Binary Cross Entropy** pentru probabilitatea claselor (cele 6 tipuri de defecte). |
| **Activation functions** | `SiLU` (Hidden) | YOLOv8 utilizează intern **SiLU** (Swish), care performează mai bine decât ReLU în rețelele convoluționale adânci, prevenind problema "dying neurons". |

**Justificare detaliată batch size:**

```
Am ales batch_size=2 datorită limitărilor hardware.

```

**Resurse învățare rapidă:**

- Împărțire date: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html (video 3 min: https://youtu.be/1NjLMWSGosI?si=KL8Qv2SJ1d_mFZfr)
- Antrenare simplă Keras: https://keras.io/examples/vision/mnist_convnet/ (secțiunea „Training”)
- Antrenare simplă PyTorch: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-an-image-classifier (video 2 min: https://youtu.be/ORMx45xqWkA?si=FXyQEhh0DU8VnuVJ)
- F1-score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html (video 4 min: https://youtu.be/ZQlEcyNV6wc?si=VMCl8aGfhCfp5Egi)

---

### Nivel 2 – Recomandat (85-90% din punctaj)

Includeți **TOATE** cerințele Nivel 1 + următoarele:

1.  **Early Stopping** - oprirea antrenării dacă `val_loss` nu scade în 5 epoci consecutiv
2.  **Learning Rate Scheduler** - `ReduceLROnPlateau` sau `StepLR`
3.  **Augmentări relevante domeniu:**
   - Vibrații motor: zgomot gaussian calibrat, jitter temporal
   - Imagini industriale: slight perspective, lighting variation (nu rotații simple!)
   - Serii temporale: time warping, magnitude warping
4.  **Grafic loss și val_loss** în funcție de epoci salvat în `docs/loss_curve.png`
5.  **Analiză erori context industrial** (vezi secțiunea dedicată mai jos - OBLIGATORIU Nivel 2)

**Indicatori țintă Nivel 2:**

-  **Acuratețe ≥ 75%**
-  **F1-score (macro) ≥ 0.70**

**Resurse învățare (aplicații industriale):**
- Albumentations: https://albumentations.ai/docs/examples/
- Early Stopping + ReduceLROnPlateau în Keras: https://keras.io/api/callbacks/
- Scheduler în PyTorch: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate


---


### Nivel 3 – Bonus (până la 100%)

**Punctaj bonus per activitate:**

| **Activitate** | **Livrabil** |
|----------------|--------------|
| Comparare 2+ arhitecturi diferite | Tabel comparativ + justificare alegere finală în README |
| Export ONNX/TFLite + benchmark latență | Fișier `models/final_model.onnx` + demonstrație <50ms |
| Confusion Matrix + analiză 5 exemple greșite | `docs/confusion_matrix.png` + analiză în README |


**Resurse bonus:**

- Export ONNX din PyTorch: [PyTorch ONNX Tutorial](https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
- TensorFlow Lite converter: [TFLite Conversion Guide](https://www.tensorflow.org/lite/convert)
- Confusion Matrix analiză: [Scikit-learn Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

---


## Verificare Consistență cu State Machine (Etapa 4)


Antrenarea și inferența trebuie să respecte fluxul din State Machine-ul vostru definit în Etapa 4.

  
**Exemplu pentru monitorizare vibrații lagăr:**

  
| **Stare din Etapa 4** | **Implementare în Etapa 5** |
|-----------------------|-----------------------------|
| `ACQUIRE_DATA` | Citire batch date din `data/train/` pentru antrenare |
| `PREPROCESS` | Aplicare scaler salvat din `config/preprocessing_params.pkl` |
| `RN_INFERENCE` | Forward pass cu model ANTRENAT (nu weights random) |
| `THRESHOLD_CHECK` | Clasificare Normal/Uzură pe baza output RN antrenat |
| `ALERT` | Trigger în UI bazat pe predicție modelului real |


**În `src/app/main.py` (UI actualizat):**


Verificați că **TOATE stările** din State Machine sunt implementate cu modelul antrenat:


```python

# ÎNAINTE (Etapa 4 - model dummy):

model = keras.models.load_model('models/untrained_model.h5') # weights random
prediction = model.predict(input_scaled) # output aproape aleator


# ACUM (Etapa 5 - model antrenat):

model = keras.models.load_model('models/trained_model.h5') # weights antrenate
prediction = model.predict(input_scaled) # predicție REALĂ și corectă

```


---


## Analiză Erori în Context Industrial (OBLIGATORIU Nivel 2)


**Nu e suficient să raportați doar acuratețea globală.** Analizați performanța în contextul aplicației voastre industriale:

### 1. Pe ce clase greșește cel mai mult modelul?

**Exemplu robotică (predicție traiectorii):**

```
Confusion Matrix arată o confuzie majoră (cca. 15-18%) între clasele 'Rolled-in_scale' și 'Pitted_surface'.
Cauză: Ambele defecte se manifestă vizual ca zone rugoase, cu puncte întunecate pe suprafață. La rezoluția redusă de 200x200 pixeli, textura fină care le diferențiază se pierde, iar rețeaua le percepe ca fiind identice.

```

### 2. Ce caracteristici ale datelor cauzează erori?


```
Modelul are performanțe slabe pe imaginile cu contrast scăzut (defecte gri pe fundal gri) și pe cele cu defecte foarte fine (ex: 'Crazing' - micro-fisuri).
Redimensionarea forțată la 200x200 pixeli (pentru viteză) elimină detaliile de înaltă frecvență necesare pentru a distinge o fisură fină de zgomotul digital al camerei sau de textura naturală a oțelului.

```


**Completați pentru proiectul vostru:**

```

Modelul performează slab la o rată de încredere peste 0.25.

```

  
### 3. Ce implicații are pentru aplicația industrială?


```

FALSE NEGATIVES (Defect critic ratat - ex: Crazing/Fisură): CRITIC/INACCEPTABIL.
O fisură nedetectată poate duce la cedarea structurală a piesei în utilizare (ex: în industria auto).

FALSE POSITIVES (Alarmă falsă - ex: Urmă de ulei clasificată ca 'Patch'): TOLERABIL.
Costul este doar timpul operatorului uman pentru a re-verifica piesa și a infirma alarma.

Prioritate: Maximizarea Recall-ului (siguranța că prindem tot) este mai importantă decât Precizia.
Soluție: Scăderea pragului de detecție (conf) de la 0.5 la 0.25 în etapa de inferență.

```


### 4. Ce măsuri corective propuneți?


```

Măsuri corective propuse:

1. Creșterea rezoluției de intrare de la 200x200 la 640x640 pixeli pentru a păstra detaliile vizuale ale defectelor fine ('Crazing').
2. Implementarea preprocesării CLAHE (Contrast Limited Adaptive Histogram Equalization) înainte de inferență pentru a accentua contrastul defectelor față de fundalul metalic.
3. Generarea de date sintetice specifice (augmentare) cu modele de 'Crazing' mai groase/evidente pentru a forța rețeaua să învețe topologia defectului.
4. Ajustarea matricei de cost (Class Weights) în antrenare pentru a penaliza mai tare erorile pe clasele critice (Crazing, Inclusion).

```

---


## Structura Repository-ului la Finalul Etapei 5


**Clarificare organizare:** Vom folosi **README-uri separate** pentru fiecare etapă în folderul `docs/`:


```

proiect-rn-[prenume-nume]/

├── README.md # Overview general proiect (actualizat)
├── etapa3_analiza_date.md # Din Etapa 3
├── etapa4_arhitectura_sia.md # Din Etapa 4
├── etapa5_antrenare_model.md # ← ACEST FIȘIER (completat)
│
├── docs/
│ ├── state_machine.png # Din Etapa 4
│ ├── loss_curve.png # NOU - Grafic antrenare
│ ├── confusion_matrix.png # (opțional - Nivel 3)
│ └── screenshots/
│ ├── inference_real.png # NOU - OBLIGATORIU
│ └── ui_demo.png # Din Etapa 4
│
├── data/ # Din Etapa 3-4 (NESCHIMBAT)
│ ├── raw/
│ ├── generated/ # Contribuția voastră 40%
│ ├── processed/
│ ├── train/
│ ├── validation/
│ └── test/
│
├── src/
│ ├── data_acquisition/ # Din Etapa 4
│ ├── preprocessing/ # Din Etapa 3
│ │ └── combine_datasets.py # NOU (dacă ați adăugat date în Etapa 4)
│ ├── neural_network/
│ │ ├── model.py # Din Etapa 4
│ │ ├── train.py # NOU - Script antrenare
│ │ └── evaluate.py # NOU - Script evaluare
│ └── app/
│ └── main.py # ACTUALIZAT - încarcă model antrenat
│
├── models/
│ ├── untrained_model.h5 # Din Etapa 4
│ ├── trained_model.h5 # NOU - OBLIGATORIU
│ └── final_model.onnx # (opțional - Nivel 3 bonus)
│
├── results/ # NOU - Folder rezultate antrenare
│ ├── training_history.csv # OBLIGATORIU - toate epoch-urile
│ ├── test_metrics.json # Metrici finale pe test set
│ └── hyperparameters.yaml # Hiperparametri folosiți
│
├── config/
│ └── preprocessing_params.pkl # Din Etapa 3 (NESCHIMBAT)
│
├── requirements.txt # Actualizat
└── .gitignore

```

  

**Diferențe față de Etapa 4:**

- Adăugat `docs/etapa5_antrenare_model.md` (acest fișier)

- Adăugat `docs/loss_curve.png` (Nivel 2)

- Adăugat `models/trained_model.h5` - OBLIGATORIU

- Adăugat `results/` cu history și metrici

- Adăugat `src/neural_network/train.py` și `evaluate.py`

- Actualizat `src/app/main.py` să încarce model antrenat

  

---

  

## Instrucțiuni de Rulare (Actualizate față de Etapa 4)

  

### 1. Setup mediu (dacă nu ați făcut deja)

  

```bash

pip  install  -r  requirements.txt

```

  

### 2. Pregătire date (DACĂ ați adăugat date noi în Etapa 4)

  

```bash

# Combinare + reprocesare dataset complet

python  src/preprocessing/combine_datasets.py

python  src/preprocessing/data_cleaner.py

python  src/preprocessing/feature_engineering.py

python  src/preprocessing/data_splitter.py  --stratify  --random_state  42

```

  

### 3. Antrenare model

  

```bash

python  src/neural_network/train.py  --epochs  50  --batch_size  32  --early_stopping

  

# Output așteptat:

# Epoch 1/50 - loss: 0.8234 - accuracy: 0.6521 - val_loss: 0.7891 - val_accuracy: 0.6823

# ...

# Epoch 23/50 - loss: 0.3456 - accuracy: 0.8234 - val_loss: 0.4123 - val_accuracy: 0.7956

# Early stopping triggered at epoch 23

# ✓ Model saved to models/trained_model.h5

```

  

### 4. Evaluare pe test set

  

```bash

python  src/neural_network/evaluate.py  --model  models/trained_model.h5

  

# Output așteptat:

# Test Accuracy: 0.7823

# Test F1-score (macro): 0.7456

# ✓ Metrics saved to results/test_metrics.json

# ✓ Confusion matrix saved to docs/confusion_matrix.png

```

  

### 5. Lansare UI cu model antrenat

  

```bash

streamlit  run  src/app/main.py

  

# SAU pentru LabVIEW:

# Deschideți WebVI și rulați main.vi

```

  

**Testare în UI:**

1. Introduceți date de test (manual sau upload fișier)

2. Verificați că predicția este DIFERITĂ de Etapa 4 (când era random)

3. Verificați că confidence scores au sens (ex: 85% pentru clasa corectă)

4. Faceți screenshot → salvați în `docs/screenshots/inference_real.png`

  

---

  

## Checklist Final – Bifați Totul Înainte de Predare

  
### Prerequisite Etapa 4 (verificare)

- [x] State Machine există și e documentat în `docs/state_machine.*`
- [x] Contribuție ≥40% date originale verificabilă în `data/train/`
- [ ] Cele 3 module din Etapa 4 funcționale

### Preprocesare și Date

- [x] Dataset combinat (vechi + nou) preprocesat (dacă ați adăugat date)
- [x] Split train/val/test: 70/15/15% (verificat dimensiuni fișiere)
- [ ] Scaler din Etapa 3 folosit consistent (`config/preprocessing_params.pkl`)

### Antrenare Model - Nivel 1 (OBLIGATORIU)

- [x] Model antrenat de la ZERO (nu fine-tuning pe model pre-antrenat)
- [x] Minimum 10 epoci rulate (verificabil în `results/training_history.csv`)
- [x] Tabel hiperparametri + justificări completat în acest README
- [x] Metrici calculate pe test set: **Accuracy ≥65%**, **F1 ≥0.60**
- [x] Model salvat în `models/trained_model.h5` (sau .pt, .lvmodel)
- [x] `results/training_history.csv` există cu toate epoch-urile

### Integrare UI și Demonstrație - Nivel 1 (OBLIGATORIU)

- [x] Model ANTRENAT încărcat în UI din Etapa 4 (nu model dummy)
- [ ] UI face inferență REALĂ cu predicții corecte
- [ ] Screenshot inferență reală în `docs/screenshots/inference_real.png`
- [ ] Verificat: predicțiile sunt diferite față de Etapa 4 (când erau random)

### Documentație Nivel 2 (dacă aplicabil)

- [ ] Early stopping implementat și documentat în cod

- [ ] Learning rate scheduler folosit (ReduceLROnPlateau / StepLR)

- [ ] Augmentări relevante domeniu aplicate (NU rotații simple!)

- [ ] Grafic loss/val_loss salvat în `docs/loss_curve.png`

- [ ] Analiză erori în context industrial completată (4 întrebări răspunse)

- [ ] Metrici Nivel 2: **Accuracy ≥75%**, **F1 ≥0.70**

### Documentație Nivel 3 Bonus (dacă aplicabil)

- [ ] Comparație 2+ arhitecturi (tabel comparativ + justificare)

- [ ] Export ONNX/TFLite + benchmark latență (<50ms demonstrat)

- [ ] Confusion matrix + analiză 5 exemple greșite cu implicații

  

### Verificări Tehnice

- [ ] `requirements.txt` actualizat cu toate bibliotecile noi

- [x] Toate path-urile RELATIVE (nu absolute: `/Users/...` )

- [x] Cod nou comentat în limba română sau engleză (minimum 15%)

- [x] `git log` arată commit-uri incrementale (NU 1 commit gigantic)

- [ ] Verificare anti-plagiat: toate punctele 1-5 respectate

  

### Verificare State Machine (Etapa 4)

- [ ] Fluxul de inferență respectă stările din State Machine

- [ ] Toate stările critice (PREPROCESS, INFERENCE, ALERT) folosesc model antrenat

- [x] UI reflectă State Machine-ul pentru utilizatorul final

  

### Pre-Predare

- [ ] `docs/etapa5_antrenare_model.md` completat cu TOATE secțiunile

- [ ] Structură repository conformă: `docs/`, `results/`, `models/` actualizate

- [ ] Commit: `"Etapa 5 completă – Accuracy=X.XX, F1=X.XX"`

- [ ] Tag: `git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat"`

- [ ] Push: `git push origin main --tags`

- [ ] Repository accesibil (public sau privat cu acces profesori)

  

---

  

## Livrabile Obligatorii (Nivel 1)

  

Asigurați-vă că următoarele fișiere există și sunt completate:

  

1.  **`docs/etapa5_antrenare_model.md`** (acest fișier) cu:

- Tabel hiperparametri + justificări (complet)

- Metrici test set raportate (accuracy, F1)

- (Nivel 2) Analiză erori context industrial (4 paragrafe)

  

2.  **`models/trained_model.h5`** (sau `.pt`, `.lvmodel`) - model antrenat funcțional

  

3.  **`results/training_history.csv`** - toate epoch-urile salvate

  

4.  **`results/test_metrics.json`** - metrici finale:

  

Exemplu:

```json

{

"test_accuracy": 0.7823,

"test_f1_macro": 0.7456,

"test_precision_macro": 0.7612,

"test_recall_macro": 0.7321

}

```

  

5.  **`docs/screenshots/inference_real.png`** - demonstrație UI cu model antrenat

  

6.  **(Nivel 2)**  `docs/loss_curve.png` - grafic loss vs val_loss

  

7.  **(Nivel 3)**  `docs/confusion_matrix.png` + analiză în README

  

---

  

## Predare și Contact

  

**Predarea se face prin:**

1. Commit pe GitHub: `"Etapa 5 completă – Accuracy=X.XX, F1=X.XX"`

2. Tag: `git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat"`

3. Push: `git push origin main --tags`

  

---

**Mult succes! Această etapă demonstrează că Sistemul vostru cu Inteligență Artificială (SIA) funcționează în condiții reale!**

# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Nume Prenume]  
**Link Repository GitHub:** [URL complet]  
**Data predării:** [Data]

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Antrenarea efectivă a modelului RN definit în Etapa 4, evaluarea performanței și integrarea în aplicația completă.

**Pornire obligatorie:** Arhitectura completă și funcțională din Etapa 4:
- State Machine definit și justificat
- Cele 3 module funcționale (Data Logging, RN, UI)
- Minimum 40% date originale în dataset

---

## PREREQUISITE – Verificare Etapa 4 (OBLIGATORIU)

**Înainte de a începe Etapa 5, verificați că aveți din Etapa 4:**

- [ ] **State Machine** definit și documentat în `docs/state_machine.*`
- [ ] **Contribuție ≥40% date originale** în `data/generated/` (verificabil)
- [ ] **Modul 1 (Data Logging)** funcțional - produce CSV-uri
- [ ] **Modul 2 (RN)** cu arhitectură definită dar NEANTRENATĂ (`models/untrained_model.h5`)
- [ ] **Modul 3 (UI/Web Service)** funcțional cu model dummy
- [ ] **Tabelul "Nevoie → Soluție → Modul"** complet în README Etapa 4

** Dacă oricare din punctele de mai sus lipsește → reveniți la Etapa 4 înainte de a continua.**

---

## Pregătire Date pentru Antrenare 

### Dacă ați adăugat date noi în Etapa 4 (contribuția de 40%):

**TREBUIE să refaceți preprocesarea pe dataset-ul COMBINAT:**

Exemplu:
```bash
# 1. Combinare date vechi (Etapa 3) + noi (Etapa 4)
python src/preprocessing/combine_datasets.py

# 2. Refacere preprocesare COMPLETĂ
python src/preprocessing/data_cleaner.py
python src/preprocessing/feature_engineering.py
python src/preprocessing/data_splitter.py --stratify --random_state 42

# Verificare finală:
# data/train/ → trebuie să conțină date vechi + noi
# data/validation/ → trebuie să conțină date vechi + noi
# data/test/ → trebuie să conțină date vechi + noi
```

** ATENȚIE - Folosiți ACEIAȘI parametri de preprocesare:**
- Același `scaler` salvat în `config/preprocessing_params.pkl`
- Aceiași proporții split: 70% train / 15% validation / 15% test
- Același `random_state=42` pentru reproducibilitate

**Verificare rapidă:**
```python
import pandas as pd
train = pd.read_csv('data/train/X_train.csv')
print(f"Train samples: {len(train)}")  # Trebuie să includă date noi
```

---

##  Cerințe Structurate pe 3 Niveluri

### Nivel 1 – Obligatoriu pentru Toți (70% din punctaj)

Completați **TOATE** punctele următoare:

1. **Antrenare model** definit în Etapa 4 pe setul final de date (≥40% originale)
2. **Minimum 10 epoci**, batch size 8–32
3. **Împărțire stratificată** train/validation/test: 70% / 15% / 15%
4. **Tabel justificare hiperparametri** (vezi secțiunea de mai jos - OBLIGATORIU)
5. **Metrici calculate pe test set:**
   - **Acuratețe ≥ 65%**
   - **F1-score (macro) ≥ 0.60**
6. **Salvare model antrenat** în `models/trained_model.h5` (Keras/TensorFlow) sau `.pt` (PyTorch) sau `.lvmodel` (LabVIEW)
7. **Integrare în UI din Etapa 4:**
   - UI trebuie să încarce modelul ANTRENAT (nu dummy)
   - Inferență REALĂ demonstrată
   - Screenshot în `docs/screenshots/inference_real.png`

#### Tabel Hiperparametri și Justificări (OBLIGATORIU - Nivel 1)

Completați tabelul cu hiperparametrii folosiți și **justificați fiecare alegere**:

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
|--------------------|-------------------|-----------------|
| Learning rate | Ex: 0.001 | Valoare standard pentru Adam optimizer, asigură convergență stabilă |
| Batch size | Ex: 32 | Compromis memorie/stabilitate pentru N=[numărul vostru] samples |
| Number of epochs | Ex: 50 | Cu early stopping după 10 epoci fără îmbunătățire |
| Optimizer | Ex: Adam | Adaptive learning rate, potrivit pentru RN cu [numărul vostru] straturi |
| Loss function | Ex: Categorical Crossentropy | Clasificare multi-class cu K=[numărul vostru] clase |
| Activation functions | Ex: ReLU (hidden), Softmax (output) | ReLU pentru non-linearitate, Softmax pentru probabilități clase |

**Justificare detaliată batch size (exemplu):**
```
Am ales batch_size=32 pentru că avem N=15,000 samples → 15,000/32 ≈ 469 iterații/epocă.
Aceasta oferă un echilibru între:
- Stabilitate gradient (batch prea mic → zgomot mare în gradient)
- Memorie GPU (batch prea mare → out of memory)
- Timp antrenare (batch 32 asigură convergență în ~50 epoci pentru problema noastră)
```

**Resurse învățare rapidă:**
- Împărțire date: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html (video 3 min: https://youtu.be/1NjLMWSGosI?si=KL8Qv2SJ1d_mFZfr)  
- Antrenare simplă Keras: https://keras.io/examples/vision/mnist_convnet/ (secțiunea „Training”)  
- Antrenare simplă PyTorch: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-an-image-classifier (video 2 min: https://youtu.be/ORMx45xqWkA?si=FXyQEhh0DU8VnuVJ)  
- F1-score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html (video 4 min: https://youtu.be/ZQlEcyNV6wc?si=VMCl8aGfhCfp5Egi)


---

### Nivel 2 – Recomandat (85-90% din punctaj)

Includeți **TOATE** cerințele Nivel 1 + următoarele:

1. **Early Stopping** - oprirea antrenării dacă `val_loss` nu scade în 5 epoci consecutive
2. **Learning Rate Scheduler** - `ReduceLROnPlateau` sau `StepLR`
3. **Augmentări relevante domeniu:**
   - Vibrații motor: zgomot gaussian calibrat, jitter temporal
   - Imagini industriale: slight perspective, lighting variation (nu rotații simple!)
   - Serii temporale: time warping, magnitude warping
4. **Grafic loss și val_loss** în funcție de epoci salvat în `docs/loss_curve.png`
5. **Analiză erori context industrial** (vezi secțiunea dedicată mai jos - OBLIGATORIU Nivel 2)

**Indicatori țintă Nivel 2:**
- **Acuratețe ≥ 75%**
- **F1-score (macro) ≥ 0.70**

**Resurse învățare (aplicații industriale):**
- Albumentations: https://albumentations.ai/docs/examples/   
- Early Stopping + ReduceLROnPlateau în Keras: https://keras.io/api/callbacks/   
- Scheduler în PyTorch: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate 

---

### Nivel 3 – Bonus (până la 100%)

**Punctaj bonus per activitate:**

| **Activitate** |  **Livrabil** |
|----------------|--------------|
| Comparare 2+ arhitecturi diferite | Tabel comparativ + justificare alegere finală în README |
| Export ONNX/TFLite + benchmark latență | Fișier `models/final_model.onnx` + demonstrație <50ms |
| Confusion Matrix + analiză 5 exemple greșite | `docs/confusion_matrix.png` + analiză în README |

**Resurse bonus:**
- Export ONNX din PyTorch: [PyTorch ONNX Tutorial](https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
- TensorFlow Lite converter: [TFLite Conversion Guide](https://www.tensorflow.org/lite/convert)
- Confusion Matrix analiză: [Scikit-learn Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

---

## Verificare Consistență cu State Machine (Etapa 4)

Antrenarea și inferența trebuie să respecte fluxul din State Machine-ul vostru definit în Etapa 4.

**Exemplu pentru monitorizare vibrații lagăr:**

| **Stare din Etapa 4** | **Implementare în Etapa 5** |
|-----------------------|-----------------------------|
| `ACQUIRE_DATA` | Citire batch date din `data/train/` pentru antrenare |
| `PREPROCESS` | Aplicare scaler salvat din `config/preprocessing_params.pkl` |
| `RN_INFERENCE` | Forward pass cu model ANTRENAT (nu weights random) |
| `THRESHOLD_CHECK` | Clasificare Normal/Uzură pe baza output RN antrenat |
| `ALERT` | Trigger în UI bazat pe predicție modelului real |

**În `src/app/main.py` (UI actualizat):**

Verificați că **TOATE stările** din State Machine sunt implementate cu modelul antrenat:

```python
# ÎNAINTE (Etapa 4 - model dummy):
model = keras.models.load_model('models/untrained_model.h5')  # weights random
prediction = model.predict(input_scaled)  # output aproape aleator

# ACUM (Etapa 5 - model antrenat):
model = keras.models.load_model('models/trained_model.h5')  # weights antrenate
prediction = model.predict(input_scaled)  # predicție REALĂ și corectă
```

---

## Analiză Erori în Context Industrial (OBLIGATORIU Nivel 2)

**Nu e suficient să raportați doar acuratețea globală.** Analizați performanța în contextul aplicației voastre industriale:

### 1. Pe ce clase greșește cel mai mult modelul?

**Exemplu robotică (predicție traiectorii):**
```
Confusion Matrix arată că modelul confundă 'viraj stânga' cu 'viraj dreapta' în 18% din cazuri.
Cauză posibilă: Features-urile IMU (gyro_z) sunt simetrice pentru viraje în direcții opuse.
```

**Completați pentru proiectul vostru:**
```
[Descrieți confuziile principale între clase și cauzele posibile]
```

### 2. Ce caracteristici ale datelor cauzează erori?

**Exemplu vibrații motor:**
```
Modelul eșuează când zgomotul de fond depășește 40% din amplitudinea semnalului util.
În mediul industrial, acest nivel de zgomot apare când mai multe motoare funcționează simultan.
```

**Completați pentru proiectul vostru:**
```
[Identificați condițiile în care modelul are performanță slabă]
```

### 3. Ce implicații are pentru aplicația industrială?

**Exemplu detectare defecte sudură:**
```
FALSE NEGATIVES (defect nedetectat): CRITIC → risc rupere sudură în exploatare
FALSE POSITIVES (alarmă falsă): ACCEPTABIL → piesa este re-inspectată manual

Prioritate: Minimizare false negatives chiar dacă cresc false positives.
Soluție: Ajustare threshold clasificare de la 0.5 → 0.3 pentru clasa 'defect'.
```

**Completați pentru proiectul vostru:**
```
[Analizați impactul erorilor în contextul aplicației voastre și prioritizați]
```

### 4. Ce măsuri corective propuneți?

**Exemplu clasificare imagini piese:**
```
Măsuri corective:
1. Colectare 500+ imagini adiționale pentru clasa minoritară 'zgârietură ușoară'
2. Implementare filtrare Gaussian blur pentru reducere zgomot cameră industrială
3. Augmentare perspective pentru simulare unghiuri camera variabile (±15°)
4. Re-antrenare cu class weights: [1.0, 2.5, 1.2] pentru echilibrare
```

**Completați pentru proiectul vostru:**
```
[Propuneți minimum 3 măsuri concrete pentru îmbunătățire]
```

---

## Structura Repository-ului la Finalul Etapei 5

**Clarificare organizare:** Vom folosi **README-uri separate** pentru fiecare etapă în folderul `docs/`:

```
proiect-rn-[prenume-nume]/
├── README.md                           # Overview general proiect (actualizat)
├── etapa3_analiza_date.md         # Din Etapa 3
├── etapa4_arhitectura_sia.md      # Din Etapa 4
├── etapa5_antrenare_model.md      # ← ACEST FIȘIER (completat)
│
├── docs/
│   ├── state_machine.png              # Din Etapa 4
│   ├── loss_curve.png                 # NOU - Grafic antrenare
│   ├── confusion_matrix.png           # (opțional - Nivel 3)
│   └── screenshots/
│       ├── inference_real.png         # NOU - OBLIGATORIU
│       └── ui_demo.png                # Din Etapa 4
│
├── data/                               # Din Etapa 3-4 (NESCHIMBAT)
│   ├── raw/
│   ├── generated/                     # Contribuția voastră 40%
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── src/
│   ├── data_acquisition/              # Din Etapa 4
│   ├── preprocessing/                 # Din Etapa 3
│   │   └── combine_datasets.py        # NOU (dacă ați adăugat date în Etapa 4)
│   ├── neural_network/
│   │   ├── model.py                   # Din Etapa 4
│   │   ├── train.py                   # NOU - Script antrenare
│   │   └── evaluate.py                # NOU - Script evaluare
│   └── app/
│       └── main.py                    # ACTUALIZAT - încarcă model antrenat
│
├── models/
│   ├── untrained_model.h5             # Din Etapa 4
│   ├── trained_model.h5               # NOU - OBLIGATORIU
│   └── final_model.onnx               # (opțional - Nivel 3 bonus)
│
├── results/                            # NOU - Folder rezultate antrenare
│   ├── training_history.csv           # OBLIGATORIU - toate epoch-urile
│   ├── test_metrics.json              # Metrici finale pe test set
│   └── hyperparameters.yaml           # Hiperparametri folosiți
│
├── config/
│   └── preprocessing_params.pkl       # Din Etapa 3 (NESCHIMBAT)
│
├── requirements.txt                    # Actualizat
└── .gitignore
```

**Diferențe față de Etapa 4:**
- Adăugat `docs/etapa5_antrenare_model.md` (acest fișier)
- Adăugat `docs/loss_curve.png` (Nivel 2)
- Adăugat `models/trained_model.h5` - OBLIGATORIU
- Adăugat `results/` cu history și metrici
- Adăugat `src/neural_network/train.py` și `evaluate.py`
- Actualizat `src/app/main.py` să încarce model antrenat

---

## Instrucțiuni de Rulare (Actualizate față de Etapa 4)

### 1. Setup mediu (dacă nu ați făcut deja)

```bash
pip install -r requirements.txt
```

### 2. Pregătire date (DACĂ ați adăugat date noi în Etapa 4)

```bash
# Combinare + reprocesare dataset complet
python src/preprocessing/combine_datasets.py
python src/preprocessing/data_cleaner.py
python src/preprocessing/feature_engineering.py
python src/preprocessing/data_splitter.py --stratify --random_state 42
```

### 3. Antrenare model

```bash
python src/neural_network/train.py --epochs 50 --batch_size 32 --early_stopping

# Output așteptat:
# Epoch 1/50 - loss: 0.8234 - accuracy: 0.6521 - val_loss: 0.7891 - val_accuracy: 0.6823
# ...
# Epoch 23/50 - loss: 0.3456 - accuracy: 0.8234 - val_loss: 0.4123 - val_accuracy: 0.7956
# Early stopping triggered at epoch 23
# ✓ Model saved to models/trained_model.h5
```

### 4. Evaluare pe test set

```bash
python src/neural_network/evaluate.py --model models/trained_model.h5

# Output așteptat:
# Test Accuracy: 0.7823
# Test F1-score (macro): 0.7456
# ✓ Metrics saved to results/test_metrics.json
# ✓ Confusion matrix saved to docs/confusion_matrix.png
```

### 5. Lansare UI cu model antrenat

```bash
streamlit run src/app/main.py

# SAU pentru LabVIEW:
# Deschideți WebVI și rulați main.vi
```

**Testare în UI:**
1. Introduceți date de test (manual sau upload fișier)
2. Verificați că predicția este DIFERITĂ de Etapa 4 (când era random)
3. Verificați că confidence scores au sens (ex: 85% pentru clasa corectă)
4. Faceți screenshot → salvați în `docs/screenshots/inference_real.png`

---

## Checklist Final – Bifați Totul Înainte de Predare

### Prerequisite Etapa 4 (verificare)
- [ ] State Machine există și e documentat în `docs/state_machine.*`
- [ ] Contribuție ≥40% date originale verificabilă în `data/generated/`
- [ ] Cele 3 module din Etapa 4 funcționale

### Preprocesare și Date
- [ ] Dataset combinat (vechi + nou) preprocesat (dacă ați adăugat date)
- [ ] Split train/val/test: 70/15/15% (verificat dimensiuni fișiere)
- [ ] Scaler din Etapa 3 folosit consistent (`config/preprocessing_params.pkl`)

### Antrenare Model - Nivel 1 (OBLIGATORIU)
- [ ] Model antrenat de la ZERO (nu fine-tuning pe model pre-antrenat)
- [ ] Minimum 10 epoci rulate (verificabil în `results/training_history.csv`)
- [ ] Tabel hiperparametri + justificări completat în acest README
- [ ] Metrici calculate pe test set: **Accuracy ≥65%**, **F1 ≥0.60**
- [ ] Model salvat în `models/trained_model.h5` (sau .pt, .lvmodel)
- [ ] `results/training_history.csv` există cu toate epoch-urile

### Integrare UI și Demonstrație - Nivel 1 (OBLIGATORIU)
- [ ] Model ANTRENAT încărcat în UI din Etapa 4 (nu model dummy)
- [ ] UI face inferență REALĂ cu predicții corecte
- [ ] Screenshot inferență reală în `docs/screenshots/inference_real.png`
- [ ] Verificat: predicțiile sunt diferite față de Etapa 4 (când erau random)

### Documentație Nivel 2 (dacă aplicabil)
- [ ] Early stopping implementat și documentat în cod
- [ ] Learning rate scheduler folosit (ReduceLROnPlateau / StepLR)
- [ ] Augmentări relevante domeniu aplicate (NU rotații simple!)
- [ ] Grafic loss/val_loss salvat în `docs/loss_curve.png`
- [ ] Analiză erori în context industrial completată (4 întrebări răspunse)
- [ ] Metrici Nivel 2: **Accuracy ≥75%**, **F1 ≥0.70**

### Documentație Nivel 3 Bonus (dacă aplicabil)
- [ ] Comparație 2+ arhitecturi (tabel comparativ + justificare)
- [ ] Export ONNX/TFLite + benchmark latență (<50ms demonstrat)
- [ ] Confusion matrix + analiză 5 exemple greșite cu implicații

### Verificări Tehnice
- [ ] `requirements.txt` actualizat cu toate bibliotecile noi
- [ ] Toate path-urile RELATIVE (nu absolute: `/Users/...` )
- [ ] Cod nou comentat în limba română sau engleză (minimum 15%)
- [ ] `git log` arată commit-uri incrementale (NU 1 commit gigantic)
- [ ] Verificare anti-plagiat: toate punctele 1-5 respectate

### Verificare State Machine (Etapa 4)
- [ ] Fluxul de inferență respectă stările din State Machine
- [ ] Toate stările critice (PREPROCESS, INFERENCE, ALERT) folosesc model antrenat
- [ ] UI reflectă State Machine-ul pentru utilizatorul final

### Pre-Predare
- [ ] `docs/etapa5_antrenare_model.md` completat cu TOATE secțiunile
- [ ] Structură repository conformă: `docs/`, `results/`, `models/` actualizate
- [ ] Commit: `"Etapa 5 completă – Accuracy=X.XX, F1=X.XX"`
- [ ] Tag: `git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat"`
- [ ] Push: `git push origin main --tags`
- [ ] Repository accesibil (public sau privat cu acces profesori)

---

## Livrabile Obligatorii (Nivel 1)

Asigurați-vă că următoarele fișiere există și sunt completate:

1. **`docs/etapa5_antrenare_model.md`** (acest fișier) cu:
   - Tabel hiperparametri + justificări (complet)
   - Metrici test set raportate (accuracy, F1)
   - (Nivel 2) Analiză erori context industrial (4 paragrafe)

2. **`models/trained_model.h5`** (sau `.pt`, `.lvmodel`) - model antrenat funcțional

3. **`results/training_history.csv`** - toate epoch-urile salvate

4. **`results/test_metrics.json`** - metrici finale:

Exemplu:
```json
{
  "test_accuracy": 0.7823,
  "test_f1_macro": 0.7456,
  "test_precision_macro": 0.7612,
  "test_recall_macro": 0.7321
}
```

5. **`docs/screenshots/inference_real.png`** - demonstrație UI cu model antrenat

6. **(Nivel 2)** `docs/loss_curve.png` - grafic loss vs val_loss

7. **(Nivel 3)** `docs/confusion_matrix.png` + analiză în README

---

## Predare și Contact

**Predarea se face prin:**
1. Commit pe GitHub: `"Etapa 5 completă – Accuracy=X.XX, F1=X.XX"`
2. Tag: `git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat"`
3. Push: `git push origin main --tags`

---

**Mult succes! Această etapă demonstrează că Sistemul vostru cu Inteligență Artificială (SIA) funcționează în condiții reale!**

# README – Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Nume Prenume]  
**Link Repository GitHub:** [URL complet]  
**Data predării:** [Data]

---
## Scopul Etapei 6

Această etapă corespunde punctelor **7. Analiza performanței și optimizarea parametrilor**, **8. Analiza și agregarea rezultatelor** și **9. Formularea concluziilor finale** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Maturizarea completă a Sistemului cu Inteligență Artificială (SIA) prin optimizarea modelului RN, analiza detaliată a performanței și integrarea îmbunătățirilor în aplicația software completă.

**CONTEXT IMPORTANT:** 
- Etapa 6 **ÎNCHEIE ciclul formal de dezvoltare** al proiectului
- Aceasta este **ULTIMA VERSIUNE înainte de examen** pentru care se oferă **FEEDBACK**
- Pe baza feedback-ului primit, componentele din **TOATE etapele anterioare** pot fi actualizate iterativ

**Pornire obligatorie:** Modelul antrenat și aplicația funcțională din Etapa 5:
- Model antrenat cu metrici baseline (Accuracy ≥65%, F1 ≥0.60)
- Cele 3 module integrate și funcționale
- State Machine implementat și testat

---

## MESAJ CHEIE – ÎNCHEIEREA CICLULUI DE DEZVOLTARE ȘI ITERATIVITATE

**ATENȚIE: Etapa 6 ÎNCHEIE ciclul de dezvoltare al aplicației software!**

**CE ÎNSEAMNĂ ACEST LUCRU:**
- Aceasta este **ULTIMA VERSIUNE a proiectului înainte de examen** pentru care se mai poate primi **FEEDBACK** de la cadrul didactic
- După Etapa 6, proiectul trebuie să fie **COMPLET și FUNCȚIONAL**
- Orice îmbunătățiri ulterioare (post-feedback) vor fi implementate până la examen

**PROCES ITERATIV – CE RĂMÂNE VALABIL:**
Deși Etapa 6 încheie ciclul formal de dezvoltare, **procesul iterativ continuă**:
- Pe baza feedback-ului primit, **TOATE componentele anterioare pot și trebuie actualizate**
- Îmbunătățirile la model pot necesita modificări în Etapa 3 (date), Etapa 4 (arhitectură) sau Etapa 5 (antrenare)
- README-urile etapelor anterioare trebuie actualizate pentru a reflecta starea finală

**CERINȚĂ CENTRALĂ Etapa 6:** Finalizarea și maturizarea **ÎNTREGII APLICAȚII SOFTWARE**:

1. **Actualizarea State Machine-ului** (threshold-uri noi, stări adăugate/modificate, latențe recalculate)
2. **Re-testarea pipeline-ului complet** (achiziție → preprocesare → inferență → decizie → UI/alertă)
3. **Modificări concrete în cele 3 module** (Data Logging, RN, Web Service/UI)
4. **Sincronizarea documentației** din toate etapele anterioare

**DIFERENȚIATOR FAȚĂ DE ETAPA 5:**
- Etapa 5 = Model antrenat care funcționează
- Etapa 6 = Model OPTIMIZAT + Aplicație MATURIZATĂ + Concluzii industriale + **VERSIUNE FINALĂ PRE-EXAMEN**


**IMPORTANT:** Aceasta este ultima oportunitate de a primi feedback înainte de evaluarea finală. Profitați de ea!

---

## PREREQUISITE – Verificare Etapa 5 (OBLIGATORIU)

**Înainte de a începe Etapa 6, verificați că aveți din Etapa 5:**

- [x] **Model antrenat** salvat în `models/trained_model.h5` (sau `.pt`, `.lvmodel`)
- [x] **Metrici baseline** raportate: Accuracy ≥65%, F1-score ≥0.60
- [ ] **Tabel hiperparametri** cu justificări completat
- [x] **`results/training_history.csv`** cu toate epoch-urile
- [ ] **UI funcțional** care încarcă modelul antrenat și face inferență reală
- [ ] **Screenshot inferență** în `docs/screenshots/inference_real.png`
- [ ] **State Machine** implementat conform definiției din Etapa 4

**Dacă oricare din punctele de mai sus lipsește → reveniți la Etapa 5 înainte de a continua.**

---

## Cerințe

Completați **TOATE** punctele următoare:

1. **Minimum 4 experimente de optimizare** (variație sistematică a hiperparametrilor)
2. **Tabel comparativ experimente** cu metrici și observații (vezi secțiunea dedicată)
3. **Confusion Matrix** generată și analizată
4. **Analiza detaliată a 5 exemple greșite** cu explicații cauzale
5. **Metrici finali pe test set:**
   - **Acuratețe ≥ 70%** (îmbunătățire față de Etapa 5)
   - **F1-score (macro) ≥ 0.65**
6. **Salvare model optimizat** în `models/optimized_model.h5` (sau `.pt`, `.lvmodel`)
7. **Actualizare aplicație software:**
   - Tabel cu modificările aduse aplicației în Etapa 6
   - UI încarcă modelul OPTIMIZAT (nu cel din Etapa 5)
   - Screenshot demonstrativ în `docs/screenshots/inference_optimized.png`
8. **Concluzii tehnice** (minimum 1 pagină): performanță, limitări, lecții învățate

#### Tabel Experimente de Optimizare

Documentați **minimum 4 experimente** cu variații sistematice:

| **Exp#** | **Modificare față de Baseline (Etapa 5)** | **Accuracy** | **F1-score** | **Timp antrenare** | **Observații** |
|----------|------------------------------------------|--------------|--------------|-------------------|----------------|
| Baseline | Configurația din Etapa 5 | 0.72 | 0.68 | 15 min | Referință |
| Exp 1 | Learning rate 0.0001 → 0.001 | 0.74 | 0.70 | 12 min | Convergență mai rapidă |
| Exp 2 | Batch size 32 → 64 | 0.71 | 0.67 | 10 min | Stabilitate redusă |
| Exp 3 | +1 hidden layer (128 neuroni) | 0.76 | 0.73 | 22 min | Îmbunătățire semnificativă |
| Exp 4 | Dropout 0.3 → 0.5 | 0.73 | 0.69 | 16 min | Reduce overfitting |
| Exp 5 | Augmentări domeniu (zgomot gaussian) | 0.78 | 0.75 | 25 min | **BEST** - ales pentru final |

**Justificare alegere configurație finală:**
```
Am ales Exp 5 ca model final pentru că:
1. Oferă cel mai bun F1-score (0.75), critic pentru aplicația noastră de [descrieți]
2. Îmbunătățirea vine din augmentări relevante domeniului industrial (zgomot gaussian 
   calibrat la nivelul real de zgomot din mediul de producție: SNR ≈ 20dB)
3. Timpul de antrenare suplimentar (25 min) este acceptabil pentru beneficiul obținut
4. Testare pe date noi arată generalizare bună (nu overfitting pe augmentări)
```

**Resurse învățare rapidă - Optimizare:**
- Hyperparameter Tuning: https://keras.io/guides/keras_tuner/ 
- Grid Search: https://scikit-learn.org/stable/modules/grid_search.html
- Regularization (Dropout, L2): https://keras.io/api/layers/regularization_layers/

---

## 1. Actualizarea Aplicației Software în Etapa 6 

**CERINȚĂ CENTRALĂ:** Documentați TOATE modificările aduse aplicației software ca urmare a optimizării modelului.

### Tabel Modificări Aplicație Software

| **Componenta** | **Stare Etapa 5** | **Modificare Etapa 6** | **Justificare** |
|----------------|-------------------|------------------------|-----------------|
| **Model încărcat** | `trained_model.h5` | `optimized_model.h5` | +9% accuracy, -5% FN |
| **Threshold alertă (State Machine)** | 0.5 (default) | 0.35 (clasa 'defect') | Minimizare FN în context industrial |
| **Stare nouă State Machine** | N/A | `CONFIDENCE_CHECK` | Filtrare predicții cu confidence <0.6 |
| **Latență target** | 100ms | 50ms (ONNX export) | Cerință timp real producție |
| **UI - afișare confidence** | Da/Nu simplu | Bară progres + valoare % | Feedback operator îmbunătățit |
| **Logging** | Doar predicție | Predicție + confidence + timestamp | Audit trail complet |
| **Web Service response** | JSON minimal | JSON extins + metadata | Integrare API extern |

**Completați pentru proiectul vostru:**
```markdown
### Modificări concrete aduse în Etapa 6:

1. **Model înlocuit:** `models/trained_model.h5` → `models/optimized_model.h5`
   - Îmbunătățire: Accuracy +X%, F1 +Y%
   - Motivație: [descrieți de ce modelul optimizat e mai bun pentru aplicația voastră]

2. **State Machine actualizat:**
   - Threshold modificat: [valoare veche] → [valoare nouă]
   - Stare nouă adăugată: [nume stare] - [ce face]
   - Tranziție modificată: [descrieți]

3. **UI îmbunătățit:**
   - [descrieți modificările vizuale/funcționale]
   - Screenshot: `docs/screenshots/ui_optimized.png`

4. **Pipeline end-to-end re-testat:**
   - Test complet: input → preprocess → inference → decision → output
   - Timp total: [X] ms (vs [Y] ms în Etapa 5)
```

### Diagrama State Machine Actualizată (dacă s-au făcut modificări)

Dacă ați modificat State Machine-ul în Etapa 6, includeți diagrama actualizată în `docs/state_machine_v2.png` și explicați diferențele:

```
Exemplu modificări State Machine pentru Etapa 6:

ÎNAINTE (Etapa 5):
PREPROCESS → RN_INFERENCE → THRESHOLD_CHECK (0.5) → ALERT/NORMAL

DUPĂ (Etapa 6):
PREPROCESS → RN_INFERENCE → CONFIDENCE_FILTER (>0.6) → 
  ├─ [High confidence] → THRESHOLD_CHECK (0.35) → ALERT/NORMAL
  └─ [Low confidence] → REQUEST_HUMAN_REVIEW → LOG_UNCERTAIN

Motivație: Predicțiile cu confidence <0.6 sunt trimise pentru review uman,
           reducând riscul de decizii automate greșite în mediul industrial.
```

---

## 2. Analiza Detaliată a Performanței

### 2.1 Confusion Matrix și Interpretare

**Locație:** `docs/confusion_matrix_optimized.png`

**Analiză obligatorie (completați):**

```markdown
### Interpretare Confusion Matrix:

**Clasa cu cea mai bună performanță:** [Nume clasă]
- Precision: [X]%
- Recall: [Y]%
- Explicație: [De ce această clasă e recunoscută bine - ex: features distincte, multe exemple]

**Clasa cu cea mai slabă performanță:** [Nume clasă]
- Precision: [X]%
- Recall: [Y]%
- Explicație: [De ce această clasă e problematică - ex: confuzie cu altă clasă, puține exemple]

**Confuzii principale:**
1. Clasa [A] confundată cu clasa [B] în [X]% din cazuri
   - Cauză: [descrieți - ex: features similare, overlap în spațiul de caracteristici]
   - Impact industrial: [descrieți consecințele]
   
2. Clasa [C] confundată cu clasa [D] în [Y]% din cazuri
   - Cauză: [descrieți]
   - Impact industrial: [descrieți]
```

### 2.2 Analiza Detaliată a 5 Exemple Greșite

Selectați și analizați **minimum 5 exemple greșite** de pe test set:

| **Index** | **True Label** | **Predicted** | **Confidence** | **Cauză probabilă** | **Soluție propusă** |
|-----------|----------------|---------------|----------------|---------------------|---------------------|
| #127 | defect_mare | defect_mic | 0.52 | Imagine subexpusă | Augmentare brightness |
| #342 | normal | defect_mic | 0.48 | Zgomot senzor ridicat | Filtru median pre-inference |
| #567 | defect_mic | normal | 0.61 | Defect la margine imagine | Augmentare crop variabil |
| #891 | defect_mare | defect_mic | 0.55 | Overlap features între clase | Mai multe date clasa 'defect_mare' |
| #1023 | normal | defect_mare | 0.71 | Reflexie metalică interpretată ca defect | Augmentare reflexii |

**Analiză detaliată per exemplu (scrieți pentru fiecare):**
```markdown
### Exemplu #127 - Defect mare clasificat ca defect mic

**Context:** Imagine radiografică sudură, defect vizibil în centru
**Input characteristics:** brightness=0.3 (subexpus), contrast=0.7
**Output RN:** [defect_mic: 0.52, defect_mare: 0.38, normal: 0.10]

**Analiză:**
Imaginea originală are brightness scăzut (0.3 vs. media dataset 0.6), ceea ce 
face ca textura defectului să fie mai puțin distinctă. Modelul a "văzut" un 
defect, dar l-a clasificat în categoria mai puțin severă.

**Implicație industrială:**
Acest tip de eroare (downgrade severitate) poate duce la subestimarea riscului.
În producție, sudura ar fi acceptată când ar trebui re-inspectată.

**Soluție:**
1. Augmentare cu variații brightness în intervalul [0.2, 0.8]
2. Normalizare histogram înainte de inference (în PREPROCESS state)
```

---

## 3. Optimizarea Parametrilor și Experimentare

### 3.1 Strategia de Optimizare

Descrieți strategia folosită pentru optimizare:

```markdown
### Strategie de optimizare adoptată:

**Abordare:** [Manual / Grid Search / Random Search / Bayesian Optimization]

**Axe de optimizare explorate:**
1. **Arhitectură:** [variații straturi, neuroni]
2. **Regularizare:** [Dropout, L2, BatchNorm]
3. **Learning rate:** [scheduler, valori testate]
4. **Augmentări:** [tipuri relevante domeniului]
5. **Batch size:** [valori testate]

**Criteriu de selecție model final:** [ex: F1-score maxim cu constraint pe latență <50ms]

**Buget computațional:** [ore GPU, număr experimente]
```

### 3.2 Grafice Comparative

Generați și salvați în `docs/optimization/`:
- `accuracy_comparison.png` - Accuracy per experiment
- `f1_comparison.png` - F1-score per experiment
- `learning_curves_best.png` - Loss și Accuracy pentru modelul final

### 3.3 Raport Final Optimizare

```markdown
### Raport Final Optimizare

**Model baseline (Etapa 5):**
- Accuracy: 0.72
- F1-score: 0.68
- Latență: 48ms

**Model optimizat (Etapa 6):**
- Accuracy: 0.81 (+9%)
- F1-score: 0.77 (+9%)
- Latență: 35ms (-27%)

**Configurație finală aleasă:**
- Arhitectură: [descrieți]
- Learning rate: [valoare] cu [scheduler]
- Batch size: [valoare]
- Regularizare: [Dropout/L2/altele]
- Augmentări: [lista]
- Epoci: [număr] (early stopping la epoca [X])

**Îmbunătățiri cheie:**
1. [Prima îmbunătățire - ex: adăugare strat hidden → +5% accuracy]
2. [A doua îmbunătățire - ex: augmentări domeniu → +3% F1]
3. [A treia îmbunătățire - ex: threshold personalizat → -60% FN]
```

---

## 4. Agregarea Rezultatelor și Vizualizări

### 4.1 Tabel Sumar Rezultate Finale

| **Metrică** | **Etapa 4** | **Etapa 5** | **Etapa 6** | **Target Industrial** | **Status** |
|-------------|-------------|-------------|-------------|----------------------|------------|
| Accuracy | ~20% | 72% | 81% | ≥85% | Aproape |
| F1-score (macro) | ~0.15 | 0.68 | 0.77 | ≥0.80 | Aproape |
| Precision (defect) | N/A | 0.75 | 0.83 | ≥0.85 | Aproape |
| Recall (defect) | N/A | 0.70 | 0.88 | ≥0.90 | Aproape |
| False Negative Rate | N/A | 12% | 5% | ≤3% | Aproape |
| Latență inferență | 50ms | 48ms | 35ms | ≤50ms | OK |
| Throughput | N/A | 20 inf/s | 28 inf/s | ≥25 inf/s | OK |

### 4.2 Vizualizări Obligatorii

Salvați în `docs/results/`:

- [ ] `confusion_matrix_optimized.png` - Confusion matrix model final
- [ ] `learning_curves_final.png` - Loss și accuracy vs. epochs
- [ ] `metrics_evolution.png` - Evoluție metrici Etapa 4 → 5 → 6
- [ ] `example_predictions.png` - Grid cu 9+ exemple (correct + greșite)

---

## 5. Concluzii Finale și Lecții Învățate

**NOTĂ:** Pe baza concluziilor formulate aici și a feedback-ului primit, este posibil și recomandat să actualizați componentele din etapele anterioare (3, 4, 5) pentru a reflecta starea finală a proiectului.

### 5.1 Evaluarea Performanței Finale

```markdown
### Evaluare sintetică a proiectului

**Obiective atinse:**
- [ ] Model RN funcțional cu accuracy [X]% pe test set
- [ ] Integrare completă în aplicație software (3 module)
- [ ] State Machine implementat și actualizat
- [ ] Pipeline end-to-end testat și documentat
- [ ] UI demonstrativ cu inferență reală
- [ ] Documentație completă pe toate etapele

**Obiective parțial atinse:**
- [ ] [Descrieți ce nu a funcționat perfect - ex: accuracy sub target pentru clasa X]

**Obiective neatinse:**
- [ ] [Descrieți ce nu s-a realizat - ex: deployment în cloud, optimizare NPU]
```

### 5.2 Limitări Identificate

```markdown
### Limitări tehnice ale sistemului

1. **Limitări date:**
   - [ex: Dataset dezechilibrat - clasa 'defect_mare' are doar 8% din total]
   - [ex: Date colectate doar în condiții de iluminare ideală]

2. **Limitări model:**
   - [ex: Performanță scăzută pe imagini cu reflexii metalice]
   - [ex: Generalizare slabă pe tipuri de defecte nevăzute în training]

3. **Limitări infrastructură:**
   - [ex: Latență de 35ms insuficientă pentru linie producție 60 piese/min]
   - [ex: Model prea mare pentru deployment pe edge device]

4. **Limitări validare:**
   - [ex: Test set nu acoperă toate condițiile din producție reală]
```

### 5.3 Direcții de Cercetare și Dezvoltare

```markdown
### Direcții viitoare de dezvoltare

**Pe termen scurt (1-3 luni):**
1. Colectare [X] date adiționale pentru clasa minoritară
2. Implementare [tehnica Y] pentru îmbunătățire recall
3. Optimizare latență prin [metoda Z]
...

**Pe termen mediu (3-6 luni):**
1. Integrare cu sistem SCADA din producție
2. Deployment pe [platform edge - ex: Jetson, NPU]
3. Implementare monitoring MLOps (drift detection)
...

```

### 5.4 Lecții Învățate

```markdown
### Lecții învățate pe parcursul proiectului

**Tehnice:**
1. [ex: Preprocesarea datelor a avut impact mai mare decât arhitectura modelului]
2. [ex: Augmentările specifice domeniului > augmentări generice]
3. [ex: Early stopping esențial pentru evitare overfitting]

**Proces:**
1. [ex: Iterațiile frecvente pe date au adus mai multe îmbunătățiri decât pe model]
2. [ex: Testarea end-to-end timpurie a identificat probleme de integrare]
3. [ex: Documentația incrementală a economisit timp la final]

**Colaborare:**
1. [ex: Feedback de la experți domeniu a ghidat selecția features]
2. [ex: Code review a identificat bug-uri în pipeline preprocesare]
```

### 5.5 Plan Post-Feedback (ULTIMA ITERAȚIE ÎNAINTE DE EXAMEN)

```markdown
### Plan de acțiune după primirea feedback-ului

**ATENȚIE:** Etapa 6 este ULTIMA VERSIUNE pentru care se oferă feedback!
Implementați toate corecțiile înainte de examen.

După primirea feedback-ului de la evaluatori, voi:

1. **Dacă se solicită îmbunătățiri model:**
   - [ex: Experimente adiționale cu arhitecturi alternative]
   - [ex: Colectare date suplimentare pentru clase problematice]
   - **Actualizare:** `models/`, `results/`, README Etapa 5 și 6

2. **Dacă se solicită îmbunătățiri date/preprocesare:**
   - [ex: Rebalansare clase, augmentări suplimentare]
   - **Actualizare:** `data/`, `src/preprocessing/`, README Etapa 3

3. **Dacă se solicită îmbunătățiri arhitectură/State Machine:**
   - [ex: Modificare fluxuri, adăugare stări]
   - **Actualizare:** `docs/state_machine.*`, `src/app/`, README Etapa 4

4. **Dacă se solicită îmbunătățiri documentație:**
   - [ex: Detaliere secțiuni specifice]
   - [ex: Adăugare diagrame explicative]
   - **Actualizare:** README-urile etapelor vizate

5. **Dacă se solicită îmbunătățiri cod:**
   - [ex: Refactorizare module conform feedback]
   - [ex: Adăugare teste unitare]
   - **Actualizare:** `src/`, `requirements.txt`

**Timeline:** Implementare corecții până la data examen
**Commit final:** `"Versiune finală examen - toate corecțiile implementate"`
**Tag final:** `git tag -a v1.0-final-exam -m "Versiune finală pentru examen"`
```
---

## Structura Repository-ului la Finalul Etapei 6

**Structură COMPLETĂ și FINALĂ:**

```
proiect-rn-[prenume-nume]/
├── README.md                               # Overview general proiect (FINAL)
├── etapa3_analiza_date.md                  # Din Etapa 3
├── etapa4_arhitectura_sia.md               # Din Etapa 4
├── etapa5_antrenare_model.md               # Din Etapa 5
├── etapa6_optimizare_concluzii.md          # ← ACEST FIȘIER (completat)
│
├── docs/
│   ├── state_machine.png                   # Din Etapa 4
│   ├── state_machine_v2.png                # NOU - Actualizat (dacă modificat)
│   ├── loss_curve.png                      # Din Etapa 5
│   ├── confusion_matrix_optimized.png      # NOU - OBLIGATORIU
│   ├── results/                            # NOU - Folder vizualizări
│   │   ├── metrics_evolution.png           # NOU - Evoluție Etapa 4→5→6
│   │   ├── learning_curves_final.png       # NOU - Model optimizat
│   │   └── example_predictions.png         # NOU - Grid exemple
│   ├── optimization/                       # NOU - Grafice optimizare
│   │   ├── accuracy_comparison.png
│   │   └── f1_comparison.png
│   └── screenshots/
│       ├── ui_demo.png                     # Din Etapa 4
│       ├── inference_real.png              # Din Etapa 5
│       └── inference_optimized.png         # NOU - OBLIGATORIU
│
├── data/                                   # Din Etapa 3-5 (NESCHIMBAT)
│   ├── raw/
│   ├── generated/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── src/
│   ├── data_acquisition/                   # Din Etapa 4
│   ├── preprocessing/                      # Din Etapa 3
│   ├── neural_network/
│   │   ├── model.py                        # Din Etapa 4
│   │   ├── train.py                        # Din Etapa 5
│   │   ├── evaluate.py                     # Din Etapa 5
│   │   └── optimize.py                     # NOU - Script optimizare/tuning
│   └── app/
│       └── main.py                         # ACTUALIZAT - încarcă model OPTIMIZAT
│
├── models/
│   ├── untrained_model.h5                  # Din Etapa 4
│   ├── trained_model.h5                    # Din Etapa 5
│   ├── optimized_model.h5                  # NOU - OBLIGATORIU
│
├── results/
│   ├── training_history.csv                # Din Etapa 5
│   ├── test_metrics.json                   # Din Etapa 5
│   ├── optimization_experiments.csv        # NOU - OBLIGATORIU
│   ├── final_metrics.json                  # NOU - Metrici model optimizat
│
├── config/
│   ├── preprocessing_params.pkl            # Din Etapa 3
│   └── optimized_config.yaml               # NOU - Config model final
│
├── requirements.txt                        # Actualizat
└── .gitignore
```

**Diferențe față de Etapa 5:**
- Adăugat `etapa6_optimizare_concluzii.md` (acest fișier)
- Adăugat `docs/confusion_matrix_optimized.png` - OBLIGATORIU
- Adăugat `docs/results/` cu vizualizări finale
- Adăugat `docs/optimization/` cu grafice comparative
- Adăugat `docs/screenshots/inference_optimized.png` - OBLIGATORIU
- Adăugat `models/optimized_model.h5` - OBLIGATORIU
- Adăugat `results/optimization_experiments.csv` - OBLIGATORIU
- Adăugat `results/final_metrics.json` - metrici finale
- Adăugat `src/neural_network/optimize.py` - script optimizare
- Actualizat `src/app/main.py` să încarce model OPTIMIZAT
- (Opțional) `docs/state_machine_v2.png` dacă s-au făcut modificări

---

## Instrucțiuni de Rulare (Etapa 6)

### 1. Rulare experimente de optimizare

```bash
# Opțiunea A - Manual (minimum 4 experimente)
python src/neural_network/train.py --lr 0.001 --batch 32 --epochs 100 --name exp1
python src/neural_network/train.py --lr 0.0001 --batch 32 --epochs 100 --name exp2
python src/neural_network/train.py --lr 0.001 --batch 64 --epochs 100 --name exp3
python src/neural_network/train.py --lr 0.001 --batch 32 --dropout 0.5 --epochs 100 --name exp4
```

### 2. Evaluare și comparare

```bash
python src/neural_network/evaluate.py --model models/optimized_model.h5 --detailed

# Output așteptat:
# Test Accuracy: 0.8123
# Test F1-score (macro): 0.7734
# ✓ Confusion matrix saved to docs/confusion_matrix_optimized.png
# ✓ Metrics saved to results/final_metrics.json
# ✓ Top 5 errors analysis saved to results/error_analysis.json
```

### 3. Actualizare UI cu model optimizat

```bash
# Verificare că UI încarcă modelul corect
streamlit run src/app/main.py

# În consolă trebuie să vedeți:
# Loading model: models/optimized_model.h5
# Model loaded successfully. Accuracy on validation: 0.8123
```

### 4. Generare vizualizări finale

```bash
python src/neural_network/visualize.py --all

# Generează:
# - docs/results/metrics_evolution.png
# - docs/results/learning_curves_final.png
# - docs/optimization/accuracy_comparison.png
# - docs/optimization/f1_comparison.png
```

---

## Checklist Final – Bifați Totul Înainte de Predare

### Prerequisite Etapa 5 (verificare)
- [ ] Model antrenat există în `models/trained_model.h5`
- [ ] Metrici baseline raportate (Accuracy ≥65%, F1 ≥0.60)
- [ ] UI funcțional cu model antrenat
- [ ] State Machine implementat

### Optimizare și Experimentare
- [ ] Minimum 4 experimente documentate în tabel
- [ ] Justificare alegere configurație finală
- [ ] Model optimizat salvat în `models/optimized_model.h5`
- [ ] Metrici finale: **Accuracy ≥70%**, **F1 ≥0.65**
- [ ] `results/optimization_experiments.csv` cu toate experimentele
- [ ] `results/final_metrics.json` cu metrici model optimizat

### Analiză Performanță
- [ ] Confusion matrix generată în `docs/confusion_matrix_optimized.png`
- [ ] Analiză interpretare confusion matrix completată în README
- [ ] Minimum 5 exemple greșite analizate detaliat
- [ ] Implicații industriale documentate (cost FN vs FP)

### Actualizare Aplicație Software
- [ ] Tabel modificări aplicație completat
- [ ] UI încarcă modelul OPTIMIZAT (nu cel din Etapa 5)
- [ ] Screenshot `docs/screenshots/inference_optimized.png`
- [ ] Pipeline end-to-end re-testat și funcțional
- [ ] (Dacă aplicabil) State Machine actualizat și documentat

### Concluzii
- [ ] Secțiune evaluare performanță finală completată
- [ ] Limitări identificate și documentate
- [ ] Lecții învățate (minimum 5)
- [ ] Plan post-feedback scris

### Verificări Tehnice
- [ ] `requirements.txt` actualizat
- [ ] Toate path-urile RELATIVE
- [ ] Cod nou comentat (minimum 15%)
- [ ] `git log` arată commit-uri incrementale
- [ ] Verificare anti-plagiat respectată

### Verificare Actualizare Etape Anterioare (ITERATIVITATE)
- [ ] README Etapa 3 actualizat (dacă s-au modificat date/preprocesare)
- [ ] README Etapa 4 actualizat (dacă s-a modificat arhitectura/State Machine)
- [ ] README Etapa 5 actualizat (dacă s-au modificat parametri antrenare)
- [ ] `docs/state_machine.*` actualizat pentru a reflecta versiunea finală
- [ ] Toate fișierele de configurare sincronizate cu modelul optimizat

### Pre-Predare
- [ ] `etapa6_optimizare_concluzii.md` completat cu TOATE secțiunile
- [ ] Structură repository conformă modelului de mai sus
- [ ] Commit: `"Etapa 6 completă – Accuracy=X.XX, F1=X.XX (optimizat)"`
- [ ] Tag: `git tag -a v0.6-optimized-final -m "Etapa 6 - Model optimizat + Concluzii"`
- [ ] Push: `git push origin main --tags`
- [ ] Repository accesibil (public sau privat cu acces profesori)

---

## Livrabile Obligatorii

Asigurați-vă că următoarele fișiere există și sunt completate:

1. **`etapa6_optimizare_concluzii.md`** (acest fișier) cu:
   - Tabel experimente optimizare (minimum 4)
   - Tabel modificări aplicație software
   - Analiză confusion matrix
   - Analiză 5 exemple greșite
   - Concluzii și lecții învățate

2. **`models/optimized_model.h5`** (sau `.pt`, `.lvmodel`) - model optimizat funcțional

3. **`results/optimization_experiments.csv`** - toate experimentele
```

4. **`results/final_metrics.json`** - metrici finale:

Exemplu:
```json
{
  "model": "optimized_model.h5",
  "test_accuracy": 0.8123,
  "test_f1_macro": 0.7734,
  "test_precision_macro": 0.7891,
  "test_recall_macro": 0.7612,
  "false_negative_rate": 0.05,
  "false_positive_rate": 0.12,
  "inference_latency_ms": 35,
  "improvement_vs_baseline": {
    "accuracy": "+9.2%",
    "f1_score": "+9.3%",
    "latency": "-27%"
  }
}
```

5. **`docs/confusion_matrix_optimized.png`** - confusion matrix model final

6. **`docs/screenshots/inference_optimized.png`** - demonstrație UI cu model optimizat

---

## Predare și Contact

**Predarea se face prin:**
1. Commit pe GitHub: `"Etapa 6 completă – Accuracy=X.XX, F1=X.XX (optimizat)"`
2. Tag: `git tag -a v0.6-optimized-final -m "Etapa 6 - Model optimizat + Concluzii"`
3. Push: `git push origin main --tags`

---

**REMINDER:** Aceasta a fost ultima versiune pentru feedback. Următoarea predare este **VERSIUNEA FINALĂ PENTRU EXAMEN**!
