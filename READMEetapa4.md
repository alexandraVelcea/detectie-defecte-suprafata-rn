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
| **Generare imagini** | Generare a minim 40% din imaginile de antrenare și a adnotărilor asociate | Imagini .png cu adnotări în fișiere xml |

**Total observații finale:** ~250 imagini (estimat pentru final)
**Observații originale:** ~100+ imagini / categorie (40%+)

**Tipul contribuției:**
[ ] Date generate prin simulare fizică
[ ] Date achiziționate cu senzori proprii
[ ] Etichetare/adnotare manuală
[x] **Date sintetice generate prin Pillow**

**Descriere detaliată:**
Pentru a compensa lipsa de diversitate în dataset-urile publice (precum NEU-DET), am dezvoltat un modul de generare a datelor sintetice folosind librăria Python **Pillow**.

Scriptul Python `generate_augmentation.py` utilizează librăria Pillow pentru a augmenta o parte din imaginile deja existente în setul de date. Aceste imagini sunt salvate automat, verificate și vor fi integrate în pipeline-ul de antrenare alături de datele reale. Această abordare permite simularea unor scenarii de iluminare și texturi dificil de capturat în mediul real fără echipament costisitor.

**Locația codului:** `src/data_acquisition/generate_augmentation.py`
**Locația datelor:** `data/train/`

**Dovezi:**

- Grafic comparativ: `docs/generated_vs_real.png`
- Tabel statistici: `docs/data_statistics.csv`

---

### 3. Diagrama State Machine a întregului sistem

**Locație fișier:** `docs/state_machine.png`

### Justificarea State Machine-ului ales:

S-a ales o arhitectură orientată pe evenimente (Event-Driven), specifică aplicațiilor de monitorizare industrială.

**Stările principale sunt:**

1\.  **IDLE:** Sistemul așteaptă input (încărcare fișier).
2\.  **ACQUIRE_DATA:** Se citește imaginea brută.
3\.  **IS_VALID:** Verificare format (JPG/PNG) și dimensiuni minime. Dacă invalid -> `INVALID`.
4\.  **PREPROCESS:** Redimensionare la 200x200px, normalizare pixeli (0-1).
5\.  **INFERENCE (RN):** Rularea modelului YOLOv8 pre-antrenat.
6\.  **IS_VALID:** Se verifică dacă imaginea s-a încărcat. 
    * *Dacă DA* -> `FIND_DEFECT`
    * *Dacă NU* -> `INVALID`
7\.  **DEFECT_NOT_FOUND:** Nu a fost găsit niciun defect.
7\.  **CLASSIFY_DEFECT:** Identificarea tipului (ex: Crazing) și desenarea conturului.
8\.  **GENERATE_RESULTS:** Afișare imagine marcată în UI și salvare log CSV.
9\.  **INVALID:** Afișare eroare utilizator și revenire la IDLE.


---

### 4. Scheletul complet al celor 3 Module

| **Modul** | **Tehnologie** | **Status Etapa 4** |
|-----------|----------------|--------------------|
| **1. Data Acquisition** | Python (`PIL`) | **Funcțional.** Scriptul se conectează la API, generează imagini pe baza prompt-ului și le salvează local cu timestamp. |
| **2. Neural Network** | Python (`ultralytics` YOLOv8) | **Funcțional.** Arhitectura este definită (YOLOv8n), fișierul de config `data.yaml` este creat, antrenamentul poate fi inițiat. |
| **3. Web Service / UI** | Python (`matplotlib`/`opencv`/ Streamlit) | **Funcțional** Script de inferență care ia o imagine, rulează modelul și afișează rezultatul cu bounding boxes. |

**Total observații finale:** ~1200 imagini

**Observații originale (Sintetice):** 630 imagini (6 clase × 105 imagini) -> **~50% din total**

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
│   │   └── generate_data.py  # Scriptul de generare imagini (Modul 1)
│   └── neural_network/
│       ├── train.py     # Script antrenament (Modul 2)
│       ├── detect.py         # Script inferență
│       └── main.py           # Entry point aplicatie (Modul 3)
├── docs/
│   ├── README.md             # Fișier README
│   └── state_machine.png     # Diagrama stărilor
├── models/
│   └── yolov8n.pt            # Modelul (pre-trained sau fine-tuned)
├── requirements.txt          # Dependențe
└── .env                      # API Keys (ignorat de git)

