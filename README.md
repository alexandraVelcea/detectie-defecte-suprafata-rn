# detectie-defecte-suprafata-rn

# Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Velcea Alexandra 

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

* **Origine:** Datele sunt generate de Gemini AI Pro - imagini de defecte de suprafață.
* **Modul de achiziție:** Generare programatică
* **Perioada / condițiile colectării:** Decembrie 2025

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** [Ex: 15,000]
* **Număr de caracteristici (features):** [Ex: 12]
* **Tipuri de date:** Imagini
* **Format fișiere:** PNG

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|-------------------|---------|-------------|---------------|--------------------|
| feature_1 | numeric | mm | [...] | 0–150 |
| feature_2 | categorial | – | [...] | {A, B, C} |
| feature_3 | numeric | m/s | [...] | 0–2.5 |
| ... | ... | ... | ... | ... |

**Fișier recomandat:**  `data/README.md`

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

* **Medie, mediană, deviație standard**
* **Min–max și quartile**
* **Distribuții pe caracteristici** (histograme)
* **Identificarea outlierilor** (IQR / percentile)

### 3.2 Analiza calității datelor

* **Detectarea valorilor lipsă** (% pe coloană)
* **Detectarea valorilor inconsistente sau eronate**
* **Identificarea caracteristicilor redundante sau puternic corelate**

### 3.3 Probleme identificate

* [exemplu] Feature X are 8% valori lipsă
* [exemplu] Distribuția feature Y este puternic neuniformă
* [exemplu] Variabilitate ridicată în clase (class imbalance)

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* **Eliminare duplicatelor**
* **Tratarea valorilor lipsă:**
  * Feature A: imputare cu mediană
  * Feature B: eliminare (30% valori lipsă)
* **Tratarea outlierilor:** IQR / limitare percentile

### 4.2 Transformarea caracteristicilor

* **Normalizare:** Min–Max / Standardizare
* **Encoding pentru variabile categoriale**
* **Ajustarea dezechilibrului de clasă** (dacă este cazul)

### 4.3 Structurarea seturilor de date

**Împărțire recomandată:**
* 70–80% – train
* 10–15% – validation
* 10–15% – test

**Principii respectate:**
* Stratificare pentru clasificare
* Fără scurgere de informație (data leakage)
* Statistici calculate DOAR pe train și aplicate pe celelalte seturi

### 4.4 Salvarea rezultatelor preprocesării

* Date preprocesate în `data/processed/`
* Seturi train/val/test în foldere dedicate
* Parametrii de preprocesare în `config/preprocessing_config.*` (opțional)

---

##  5. Fișiere Generate în Această Etapă

* `data/raw/` – date brute
* `data/processed/` – date curățate & transformate
* `data/train/`, `data/validation/`, `data/test/` – seturi finale
* `src/preprocessing/` – codul de preprocesare
* `data/README.md` – descrierea dataset-ului

---

##  6. Stare Etapă (de completat de student)

- [ ] Structură repository configurată
- [ ] Dataset analizat (EDA realizată)
- [ ] Date preprocesate
- [ ] Seturi train/val/test generate
- [ ] Documentație actualizată în README + `data/README.md`

---


# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale
**Instituție:** POLITEHNICA București – FIIR
**Student:** [Numele Tau Aici]
**Link Repository GitHub:** [Link-ul Tau Aici]
**Data:** 05.12.2024

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

**Total observații finale:** ~2000 imagini (estimat pentru final)
**Observații originale:** ~800+ imagini (40%+)

**Tipul contribuției:**
[ ] Date generate prin simulare fizică
[ ] Date achiziționate cu senzori proprii
[ ] Etichetare/adnotare manuală
[x] **Date sintetice prin Generative AI**

**Descriere detaliată:**
Pentru a compensa lipsa de diversitate în dataset-urile publice (precum NEU-DET) și pentru a evita overfitting-ul, am dezvoltat un modul de generare a datelor sintetice folosind **Google GenAI SDK (modelul `imagen-3.0`)**.

Scriptul Python (`generate_data.py`) utilizează prompt-uri inginerești specifice (ex: *"industrial metal surface with deep rust and scratches, isometric view, photorealistic"*) pentru a crea variații unice ale defectelor. Aceste imagini sunt salvate automat, verificate și vor fi integrate în pipeline-ul de antrenare alături de datele reale. Această abordare permite simularea unor scenarii de iluminare și texturi dificil de capturat în mediul real fără echipament costisitor.

**Locația codului:** `src/data_acquisition/generate_data.py`
**Locația datelor:** `data/processed/`

**Dovezi:**

- Grafic comparativ: `docs/generated_vs_real.png`
- Setup experimental: `docs/acquisition_setup.jpg` (dacă aplicabil)
- Tabel statistici: `docs/data_statistics.csv`


---

### 3. Diagrama State Machine a Întregului Sistem (OBLIGATORIE)

**Locație fișier:** `docs/state_machine.png`

### Justificarea State Machine-ului ales:

Am ales arhitectura de tip **Clasificare/Detecție la cerere**, specifică sistemelor de controlul calitatății de pe liniile de producție.

**Stările principale sunt:**
1. **IDLE:** Sistemul așteaptă input (încărcare imagine de către operator).
2. **ACQUIRE_DATA/LOAD:** Se încarcă imaginea (reală sau generată) și se verifică integritatea fișierului.
3. **PREPROCESS:** Redimensionare la 640x640 (format YOLO) și normalizare pixelilor.
4. **INFERENCE (RN):** Modelul YOLOv8 procesează imaginea pentru a identifica coordonatele defectelor.
5. **DECISION:** Se verifică dacă scorul de încredere (confidence) este peste pragul setat (0.5).
   - Dacă **DA (Defect găsit):** Se trece în starea ALERT/LOG.
   - Dacă **NU (Curat):** Se trece în starea PASS.

**Tranzițiile critice sunt:**
- **PREPROCESS → ERROR:** Dacă imaginea este coruptă sau formatul nu este suportat.
- **INFERENCE → ALERT:** Critică pentru siguranță; declanșează marcarea vizuală a defectului pe UI.

Starea **ERROR** este esențială deoarece API-urile de generare pot da timeout sau utilizatorul poate încărca fișiere non-imagine, iar aplicația nu trebuie să se blocheze (crash), ci să revină în IDLE.

---

### 4. Scheletul complet al celor 3 Module

| **Modul** | **Tehnologie** | **Status Etapa 4** |
|-----------|----------------|--------------------|
| **1. Data Acquisition** | Python (`google-genai`, `PIL`) | **Funcțional.** Scriptul se conectează la API, generează imagini pe baza prompt-ului și le salvează local cu timestamp. |
| **2. Neural Network** | Python (`ultralytics` YOLOv8) | **Funcțional.** Arhitectura este definită (YOLOv8n), fișierul de config `data.yaml` este creat, antrenamentul poate fi inițiat. |
| **3. Web Service / UI** | Python (`matplotlib`/`opencv` sau Streamlit) | **Funcțional.** Script de inferență care ia o imagine, rulează modelul și afișează rezultatul cu bounding boxes. |

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
│   └── state_machine.png     # Diagrama stărilor
├── models/
│   └── yolov8n.pt            # Modelul (pre-trained sau fine-tuned)
├── README.md                 # Fișier README
├── requirements.txt          # Dependențe (ultralytics, google-genai, pillow)
└── .env                      # API Keys (ignorat de git)

