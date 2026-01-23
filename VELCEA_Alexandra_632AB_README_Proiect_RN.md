## 1. Identificare Proiect

| Câmp | Valoare |
|------|---------|
| **Student** | Velcea Alexandra |
| **Grupa / Specializare** | 632AB / Informatică Industrială |
| **Disciplina** | Rețele Neuronale |
| **Instituție** | POLITEHNICA București – FIIR |
| **Link Repository GitHub** | https://github.com/alexandraVelcea/detectie-defecte-suprafata-rn |
| **Acces Repository** | Public |
| **Stack Tehnologic** | Python |
| **Domeniul Industrial de Interes (DII)** | Producție |
| **Tip Rețea Neuronală** | CNN |

### Rezultate Cheie (Versiunea Finală vs Etapa 6)

| Metric | Țintă Minimă | Rezultat Etapa 6 | Rezultat Final | Îmbunătățire | Status |
|--------|--------------|------------------|----------------|--------------|--------|
| Accuracy (Test Set) | ≥70% | [X.XX%] | [X.XX%] | [+X.XX%] | [✓/✗] |
| F1-Score (Macro) | ≥0.65 | [X.XX] | [X.XX] | [+X.XX] | [✓/✗] |
| Latență Inferență | [target student] | [X ms] | [X ms] | [±X ms] | [✓/✗] |
| Contribuție Date Originale | ≥40% | [X%] | [X%] | - | [✓/✗] |
| Nr. Experimente Optimizare | ≥4 | [N] | [N] | - | [✓/✗] |

### Declarație de Originalitate & Politica de Utilizare AI

**Acest proiect reflectă munca, gândirea și deciziile mele proprii.**

Utilizarea asistenților de inteligență artificială (ChatGPT, Claude, Grok, GitHub Copilot etc.) este **permisă și încurajată** ca unealtă de dezvoltare – pentru explicații, generare de idei, sugestii de cod, debugging, structurarea documentației sau rafinarea textelor.

**Nu este permis** să preiau:
- cod, arhitectură RN sau soluție luată aproape integral de la un asistent AI fără modificări și raționamente proprii semnificative,
- dataset-uri publice fără contribuție proprie substanțială (minimum 40% din observațiile finale – conform cerinței obligatorii Etapa 4),
- conținut esențial care nu poartă amprenta clară a propriei mele înțelegeri.

**Confirmare explicită (bifez doar ce este adevărat):**

| Nr. | Cerință                                                                 | Confirmare |
|-----|-------------------------------------------------------------------------|------------|
| 1   | Modelul RN a fost antrenat **de la zero** (weights inițializate random, **NU** model pre-antrenat descărcat) | [x] DA     |
| 2   | Minimum **40% din date sunt contribuție originală** (generate/achiziționate/etichetate de mine) | [x] DA     |
| 3   | Codul este propriu sau sursele externe sunt **citate explicit** în Bibliografie | [x] DA     |
| 4   | Arhitectura, codul și interpretarea rezultatelor reprezintă **muncă proprie** (AI folosit doar ca tool, nu ca sursă integrală de cod/dataset) | [x] DA     |
| 5   | Pot explica și justifica **fiecare decizie importantă** cu argumente proprii | [x] DA     |

**Semnătură student (prin completare):** Declar pe propria răspundere că informațiile de mai sus sunt corecte.

---

## 2. Descrierea Nevoii și Soluția SIA

### 2.1 Nevoia Reală / Studiul de Caz

În industria metalurgică, asigurarea calității piesselor de oțel este o etapă critică, însă metoda tradițională de inspecție vizuală realizată manual de către operatori este lentă, subiectivă și predispusă la erori cauzate de oboseală. Pe liniile de laminare cu viteză mare, defectele subtile precum micro-fisurile sau incluziunile sunt adesea ratate, ceea ce poate compromite integritatea structurală a produsului finit.

Acest proiect propune o soluție automatizată de tip Computer Vision, utilizând rețele neuronale convoluționale (YOLOv8) pentru a detecta și clasifica în timp real șase tipuri de defecte specifice suprafețelor metalice. Rezolvarea acestei probleme este esențială pentru reducerea pierderilor economice asociate rebuturilor, garantarea standardelor de siguranță și creșterea eficienței procesului de control al calității prin eliminarea factorului uman din sarcinile repetitive.

### 2.2 Beneficii Măsurabile Urmărite

*[Listați 3-5 beneficii concrete cu metrici țintă]*

1. Eficiență Operațională (Viteză): Automatizarea procesului de inspecție permite analiza în timp real a suprafețelor metalice cu o latență de sub 50ms per imagine, depășind semnificativ viteza de reacție a unui operator uman.

2. Acuratețe și Fiabilitate: Obținerea unei precizii medii (mAP) de peste 85% în identificarea defectelor și maximizarea ratei de detecție (Recall) la > 90% pentru defectele critice, reducând riscul livrării de produse neconforme.

3. Reducerea Costurilor: Diminuarea pierderilor materiale prin identificarea timpurie a defectelor în procesul de producție, cu o țintă de reducere a rebuturilor nejustificate (False Positives) la sub 5%.

### 2.3 Tabel: Nevoie → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul** | **Modul software responsabil** | **Metric măsurabil** |
| --- | --- | --- | --- |
| Detectarea automată a defectelor (zgârieturi, rugină) pe linia de producție | Analiză vizuală automată folosind Computer Vision și localizare defecte (bounding box) | **RN Module** (YOLOv8 - Inference) | **mAP@50 > 85%**, **Recall > 90%** |
| Compensarea lipsei de date pentru defecte rare (ex: fisuri fine) | Generare programatică de imagini sintetice (augmentare) pentru diversificarea setului de date | **Data Acquisition** (Script Generare Pillow) | **+40% date sintetice** în dataset |
| Alertarea instantanee a operatorului pentru a preveni livrarea pieselor defecte | Interfață vizuală (UI) cu marcare colorată a defectelor și feedback vizual imediat | **UI / Web Service** (Desktop App) | **Latență < 50ms** / imagine |

---

## 3. Dataset și Contribuție Originală

### 3.1 Sursa și Caracteristicile Datelor

| Caracteristică | Valoare |
|----------------|---------|
| **Origine date** | Dataset public |
| **Sursa concretă** | https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database |
| **Sursă imagini demo** | https://www.researchgate.net/figure/Samples-images-of-six-classes-of-typical-surface-defects_fig3_340436222 |
| **Număr total observații finale (N)** | aprox. 1100 |
| **Număr features** | 6 |
| **Tipuri de date** | Categoriale / Imagini |
| **Format fișiere** | JPG / XML |
| **Perioada colectării/generării** | [ex: Noiembrie 2025 - Ianuarie 2026] |


### 3.2 Contribuția Originală (minim 40% OBLIGATORIU)

| Câmp | Valoare |
|------|---------|
| **Total observații finale (N)** | [număr] |
| **Observații originale (M)** | [număr] |
| **Procent contribuție originală** | [X%] |
| **Tip contribuție** | [Simulare fizică / Senzori proprii / Etichetare manuală / Date sintetice] |
| **Locație cod generare** | `src/data_acquisition/[nume_script.py]` |
| **Locație date originale** | `data/generated/` |

**Descriere metodă generare/achiziție:**

Pentru a compensa limitările cantitative ale setului de date public NEU-DET, am dezvoltat un modul propriu de generare a datelor sintetice (src/data_acquisition/generate_augmentation.py), utilizând librăria de procesare de imagine Pillow. Procesul a implicat generarea programatică a defectelor — linii Bezier cu variații de culoare pentru scratches, forme eliptice texturate pentru patches și rețele de linii interconectate pentru crazing — și suprapunerea acestora peste imagini de fundal metalic. Simultan cu generarea vizuală, scriptul a creat automat și fișierele de adnotare (XML/YOLO), eliminând eroarea umană din etichetare.

Parametrii utilizați au inclus variații aleatorii de transparență (alpha blending), luminozitate, contrast și dimensiuni geometrice (lungime, grosime, unghi de rotație), simulând astfel diversitatea condițiilor de iluminare din mediul industrial real. Această abordare este relevantă deoarece rezolvă problema dezechilibrului de clase (specific defectelor rare) și forțează rețeaua neuronală să învețe caracteristicile morfologice ale defectului, prevenind memorarea fundalului și crescând capacitatea de generalizare a modelului.

### 3.3 Preprocesare și Split Date

| Set | Procent | Număr Observații |
|-----|---------|------------------|
| Train | 70% | 270 |
| Validation | 15% | 45 |
| Test | 15% | 45 |

**Preprocesări aplicate:**

-   **Redimensionare spațială (Resizing):** Standardizarea tuturor imaginilor la rezoluția de **200x200 pixeli** pentru a asigura o dimensiune constantă a tensorului de intrare.
-   **Normalizare Min-Max (Scaling):** Scalarea valorilor de intensitate a pixelilor din intervalul [0, 255] în intervalul **[0, 1]** pentru a facilita convergența rețelei neuronale.
-   **Normalizare Coordonate Bounding Box:** Transformarea coordonatelor absolute (pixeli) în coordonate relative (0.0 - 1.0) conform formatului YOLO ($x_{center} = x_{abs} / width$).
-   **Conversie Format Adnotări:** Transcodarea etichetelor din format XML (Pascal VOC) în format TXT (YOLO Darknet).
-   **Curățare Dataset (Data Cleaning):** Eliminarea automată a imaginilor "orfane" (care nu au fișier de adnotare asociat) pentru a preveni erorile de antrenare.

**Referințe fișiere:** `docs/README.md`, `config/preprocessing_params.pkl`

---

## 4. Arhitectura SIA și State Machine

### 4.1 Cele 3 Module Software

| **Modul** | **Tehnologie** | **Status Etapa 4** |
|-----------|----------------|--------------------|
| **1. Data Acquisition** | Python (`google-genai`, `PIL`) | **Funcțional.** Scriptul se conectează la API, generează imagini pe baza prompt-ului și le salvează local cu timestamp. |
| **2. Neural Network** | Python (`ultralytics` YOLOv8) | **Funcțional.** Arhitectura este definită (YOLOv8n), fișierul de config `data.yaml` este creat, antrenamentul poate fi inițiat. |
| **3. Web Service / UI** | Python (Streamlit) | **Funcțional.** Script de inferență care ia o imagine, rulează modelul și afișează rezultatul cu bounding boxes. |

### 4.2 State Machine

**Locație diagramă:** `docs/state_machine.png` *(sau `state_machine_v2.png` dacă actualizată în Etapa 6)*

**Stări principale și descriere:**

| **Stare** | **Descriere** | **Condiție Intrare** | **Condiție Ieșire** |
| --- | --- | --- | --- |
| `IDLE` | Sistemul așteaptă input de la utilizator (stare de repaus). | Start aplicație / Finalizare ciclu anterior | Input primit (Buton "Load Image") |
| `ACQUIRE_DATA` | Încărcarea imaginii selectate în memorie pentru procesare. | Eveniment "Load Image" declanșat | Imagine încărcată cu succes |
| `PREPROCESS` | Curățarea datelor/variabilelor de la rularea anterioară (reset). | Imagine prezentă în memorie | Workspace curățat și pregătit |
| `IS_VALID` | Verificarea integrității fișierului (format suportat, dimensiuni, corupere). | Preprocesare inițială completă | Rezultat validare (True/False) |
| `INVALID` | Gestionarea erorilor de format; afișare mesaj eroare. | `IS_VALID` returnează **False** | Revenire automată în `IDLE` |
| `INFERENCE (RN)` | Rularea modelului YOLOv8 pentru detecția defectelor. | `IS_VALID` returnează **True** | Predicții brute disponibile (Listă) |
| `DEFECT_NOT_FOUND` | Gestionarea cazului în care piesa este conformă (nu sunt defecte). | Lista de predicții este goală | Rezultat setat ca "Normal" |
| `CLASSIFY_DEFECT` | Identificarea tipului de defect și a coordonatelor (bounding box). | Lista de predicții conține elemente | Clasă și scor încredere extrase |
| `GENERATE RESULTS` | Calculare statistici, salvare log-uri și actualizare interfață grafică. | Clasificare finalizată (Defect sau Normal) | Afișare rezultate -> Revenire `IDLE` |

**Justificare alegere arhitectură State Machine:**

*[1 paragraf: De ce această structură pentru problema voastră specifică?]*

[Completați aici]

### 4.3 Actualizări State Machine în Etapa 6 (dacă este cazul)

| Componentă Modificată | Valoare Etapa 5 | Valoare Etapa 6 | Justificare Modificare |
|----------------------|-----------------|-----------------|------------------------|
| [ex: Threshold alertă] | [0.5] | [0.35] | [Minimizare False Negatives] |
| [ex: Stare nouă adăugată] | N/A | `CONFIDENCE_CHECK` | [Filtrare predicții incerte] |
| [Completați dacă e cazul] | | | |

---

## 5. Modelul RN – Antrenare și Optimizare

### 5.1 Arhitectura Rețelei Neuronale

```
[Descrieți arhitectura - exemplu:]
Input (shape: [32, 32, 3]) 
  → Conv2D(32, 3x3, ReLU) → MaxPool(2x2)
  → Conv2D(64, 3x3, ReLU) → MaxPool(2x2)
  → Flatten
  → Dense(128, ReLU) → Dropout(0.3)
  → Dense(5, Softmax)
Output: 5 clase
```

**Justificare alegere arhitectură:**

*[1-2 propoziții: De ce această arhitectură? Ce alternative ați considerat și de ce le-ați respins?]*

[Completați aici]

### 5.2 Hiperparametri Finali (Model Optimizat - Etapa 6)

| Hiperparametru | Valoare Finală | Justificare Alegere |
|----------------|----------------|---------------------|
| Learning Rate | [ex: 0.001] | [ex: Valoare standard Adam, convergență stabilă] |
| Batch Size | [ex: 32] | [ex: Compromis memorie/stabilitate pentru N=15000 samples] |
| Epochs | [ex: 50] | [ex: Early stopping după 10 epoci fără îmbunătățire] |
| Optimizer | [ex: Adam] | [ex: Adaptive LR, potrivit pentru date de tip X] |
| Loss Function | [ex: Categorical Crossentropy] | [ex: Clasificare multi-clasă cu 5 clase] |
| Regularizare | [ex: Dropout 0.3 + L2(0.01)] | [ex: Prevenire overfitting observat în Exp 2] |
| Early Stopping | [ex: patience=10, monitor=val_loss] | [ex: Oprire automată la convergență] |

### 5.3 Experimente de Optimizare (minim 4 experimente)

| Exp# | Modificare față de Baseline | Accuracy | F1-Score | Timp Antrenare | Observații |
|------|----------------------------|----------|----------|----------------|------------|
| **Baseline** | Configurația din Etapa 5 | [X.XX%] | [X.XX] | [X min] | Referință |
| Exp 1 | [ex: LR 0.001 → 0.0001] | [X.XX%] | [X.XX] | [X min] | [ex: Convergență mai lentă, +2% acc] |
| Exp 2 | [ex: +1 hidden layer (64 neuroni)] | [X.XX%] | [X.XX] | [X min] | [ex: Overfitting observat] |
| Exp 3 | [ex: Dropout 0.3 → 0.5] | [X.XX%] | [X.XX] | [X min] | [ex: Reduce overfitting din Exp 2] |
| Exp 4 | [ex: Batch 32 → 64] | [X.XX%] | [X.XX] | [X min] | [ex: Stabilitate gradient mai bună] |
| Exp 5 | [ex: Augmentări domeniu specifice] | [X.XX%] | [X.XX] | [X min] | [ex: Generalizare îmbunătățită] |
| **FINAL** | [Configurația aleasă] | **[X.XX%]** | **[X.XX]** | [X min] | **Modelul folosit în producție** |

**Justificare alegere model final:**

*[1 paragraf: De ce această configurație? Ce compromisuri ați făcut între accuracy/timp/complexitate?]*

[Completați aici]

**Referințe fișiere:** `results/optimization_experiments.csv`, `models/optimized_model.h5`

---

## 6. Performanță Finală și Analiză Erori

### 6.1 Metrici pe Test Set (Model Optimizat)

| Metric | Valoare | Target Minim | Status |
|--------|---------|--------------|--------|
| **Accuracy** | [X.XX%] | ≥70% | [✓/✗] |
| **F1-Score (Macro)** | [X.XX] | ≥0.65 | [✓/✗] |
| **Precision (Macro)** | [X.XX] | - | - |
| **Recall (Macro)** | [X.XX] | - | - |

**Îmbunătățire față de Baseline (Etapa 5):**

| Metric | Etapa 5 (Baseline) | Etapa 6 (Optimizat) | Îmbunătățire |
|--------|-------------------|---------------------|--------------|
| Accuracy | [X.XX%] | [X.XX%] | [+X.XX%] |
| F1-Score | [X.XX] | [X.XX] | [+X.XX] |

**Referință fișier:** `results/final_metrics.json`

### 6.2 Confusion Matrix

**Locație:** `docs/confusion_matrix_optimized.png`

**Interpretare:**

| Aspect | Observație |
|--------|------------|
| **Clasa cu cea mai bună performanță** | [Nume clasă] - Precision [X%], Recall [Y%] |
| **Clasa cu cea mai slabă performanță** | [Nume clasă] - Precision [X%], Recall [Y%] |
| **Confuzii frecvente** | [ex: Clasa A confundată frecvent cu Clasa B - posibil din cauza similarității vizuale] |
| **Dezechilibru clase** | [ex: Clasa C are doar 5% din date - recall scăzut explicabil] |

### 6.3 Analiza Top 5 Erori

| # | Input (descriere scurtă) | Predicție RN | Clasă Reală | Cauză Probabilă | Implicație Industrială |
|---|--------------------------|--------------|-------------|-----------------|------------------------|
| 1 | [ex: Imagine sudură cu iluminare slabă] | [Clasa X] | [Clasa Y] | [ex: Contrast insuficient în zona defectului] | [ex: Defect nedetectat → produs defect la client] |
| 2 | [Completați] | [Completați] | [Completați] | [Completați] | [Completați] |
| 3 | [Completați] | [Completați] | [Completați] | [Completați] | [Completați] |
| 4 | [Completați] | [Completați] | [Completați] | [Completați] | [Completați] |
| 5 | [Completați] | [Completați] | [Completați] | [Completați] | [Completați] |

### 6.4 Validare în Context Industrial

**Ce înseamnă rezultatele pentru aplicația reală:**

*[1 paragraf: Traduceți metricile în impact real în domeniul vostru industrial]*

[ex: Din 100 de piese cu defecte reale, modelul detectează corect 78 (Recall=78%). 22 de piese defecte ajung la client - cost estimat: 22 × 50 RON = 1100 RON/lot. În același timp, din 100 piese bune, 8 sunt clasificate greșit ca defecte (FP=8%) - cost reinspecție: 8 × 5 RON = 40 RON/lot.]

**Pragul de acceptabilitate pentru domeniu:** [ex: Recall ≥ 85% pentru defecte critice]  
**Status:** [Atins / Neatins - cu diferența]  
**Plan de îmbunătățire (dacă neatins):** [ex: Augmentare date pentru clasa subreprezentată, ajustare threshold]

---

## 7. Aplicația Software Finală

### 7.1 Modificări Implementate în Etapa 6

| Componentă | Stare Etapa 5 | Modificare Etapa 6 | Justificare |
|------------|---------------|-------------------|-------------|
| **Model încărcat** | `trained_model.h5` | `optimized_model.h5` | [ex: +8% accuracy, -12% FN] |
| **Threshold decizie** | [ex: 0.5 default] | [ex: 0.35 pentru clasa 'defect'] | [ex: Minimizare FN în context producție] |
| **UI - feedback vizual** | [ex: Da/Nu text] | [ex: Bară confidence + valoare %] | [ex: Informare operator pentru decizii] |
| **Logging** | [ex: Doar predicție] | [ex: Predicție + confidence + timestamp] | [ex: Audit trail pentru QA] |
| [Alte modificări] | [Completați] | [Completați] | [Completați] |

### 7.2 Screenshot UI cu Model Optimizat

**Locație:** `docs/screenshots/inference_optimized.png`

*[Descriere scurtă: Ce se vede în screenshot? Ce demonstrează?]*

[Completați aici]

### 7.3 Demonstrație Funcțională End-to-End

**Locație dovadă:** `docs/demo/` *(GIF / Video / Secvență screenshots)*

**Fluxul demonstrat:**

| Pas | Acțiune | Rezultat Vizibil |
|-----|---------|------------------|
| 1 | Input | [ex: Upload imagine nouă (NU din train/test)] |
| 2 | Procesare | [ex: Bară de progres + preprocesare vizibilă] |
| 3 | Inferență | [ex: Predicție afișată: "Clasa: Defect, Confidence: 87%"] |
| 4 | Decizie | [ex: Alertă roșie + sunet pentru operator] |

**Latență măsurată end-to-end:** [X] ms  
**Data și ora demonstrației:** [DD.MM.YYYY, HH:MM]

---

## 8. Structura Repository-ului Final

```
proiect-rn-[nume-prenume]/
│
├── README.md                               # ← ACEST FIȘIER (Overview Final Proiect - Pe moodle la Evaluare Finala RN > Upload Livrabil 1 - Proiect RN (Aplicatie Sofware) - trebuie incarcat cu numele: NUME_Prenume_Grupa_README_Proiect_RN.md)
│
├── docs/
│   ├── etapa3_analiza_date.md              # Documentație Etapa 3
│   ├── etapa4_arhitectura_SIA.md           # Documentație Etapa 4
│   ├── etapa5_antrenare_model.md           # Documentație Etapa 5
│   ├── etapa6_optimizare_concluzii.md      # Documentație Etapa 6
│   │
│   ├── state_machine.png                   # Diagrama State Machine inițială
│   ├── state_machine_v2.png                # (opțional) Versiune actualizată Etapa 6
│   ├── confusion_matrix_optimized.png      # Confusion matrix model final
│   │
│   ├── screenshots/
│   │   ├── ui_demo.png                     # Screenshot UI schelet (Etapa 4)
│   │   ├── inference_real.png              # Inferență model antrenat (Etapa 5)
│   │   └── inference_optimized.png         # Inferență model optimizat (Etapa 6)
│   │
│   ├── demo/                               # Demonstrație funcțională end-to-end
│   │   └── demo_end_to_end.gif             # (sau .mp4 / secvență screenshots)
│   │
│   ├── results/                            # Vizualizări finale
│   │   ├── loss_curve.png                  # Grafic loss/val_loss (Etapa 5)
│   │   ├── metrics_evolution.png           # Evoluție metrici (Etapa 6)
│   │   └── learning_curves_final.png       # Curbe învățare finale
│   │
│   └── optimization/                       # Grafice comparative optimizare
│       ├── accuracy_comparison.png         # Comparație accuracy experimente
│       └── f1_comparison.png               # Comparație F1 experimente
│
├── data/
│   ├── README.md                           # Descriere detaliată dataset
│   ├── raw/                                # Date brute originale
│   ├── processed/                          # Date curățate și transformate
│   ├── generated/                          # Date originale (contribuția ≥40%)
│   ├── train/                              # Set antrenare (70%)
│   ├── validation/                         # Set validare (15%)
│   └── test/                               # Set testare (15%)
│
├── src/
│   ├── data_acquisition/                   # MODUL 1: Generare/Achiziție date
│   │   ├── README.md                       # Documentație modul
│   │   ├── generate.py                     # Script generare date originale
│   │   └── [alte scripturi achiziție]
│   │
│   ├── preprocessing/                      # Preprocesare date (Etapa 3+)
│   │   ├── data_cleaner.py                 # Curățare date
│   │   ├── feature_engineering.py          # Extragere/transformare features
│   │   ├── data_splitter.py                # Împărțire train/val/test
│   │   └── combine_datasets.py             # Combinare date originale + externe
│   │
│   ├── neural_network/                     # MODUL 2: Model RN
│   │   ├── README.md                       # Documentație arhitectură RN
│   │   ├── model.py                        # Definire arhitectură (Etapa 4)
│   │   ├── train.py                        # Script antrenare (Etapa 5)
│   │   ├── evaluate.py                     # Script evaluare metrici (Etapa 5)
│   │   ├── optimize.py                     # Script experimente optimizare (Etapa 6)
│   │   └── visualize.py                    # Generare grafice și vizualizări
│   │
│   └── app/                                # MODUL 3: UI/Web Service
│       ├── README.md                       # Instrucțiuni lansare aplicație
│       └── main.py                         # Aplicație principală
│
├── models/
│   ├── untrained_model.h5                  # Model schelet neantrenat (Etapa 4)
│   ├── trained_model.h5                    # Model antrenat baseline (Etapa 5)
│   ├── optimized_model.h5                  # Model FINAL optimizat (Etapa 6) ← FOLOSIT
│   └── final_model.onnx                    # (opțional) Export ONNX pentru deployment
│
├── results/
│   ├── training_history.csv                # Istoric antrenare - toate epocile (Etapa 5)
│   ├── test_metrics.json                   # Metrici baseline test set (Etapa 5)
│   ├── optimization_experiments.csv        # Toate experimentele optimizare (Etapa 6)
│   ├── final_metrics.json                  # Metrici finale model optimizat (Etapa 6)
│   └── error_analysis.json                 # Analiza detaliată erori (Etapa 6)
│
├── config/
│   ├── preprocessing_params.pkl            # Parametri preprocesare salvați (Etapa 3)
│   └── optimized_config.yaml               # Configurație finală model (Etapa 6)
│
├── requirements.txt                        # Dependențe Python (actualizat la fiecare etapă)
└── .gitignore                              # Fișiere excluse din versionare
```

### Legendă Progresie pe Etape

| Folder / Fișier | Etapa 3 | Etapa 4 | Etapa 5 | Etapa 6 |
|-----------------|:-------:|:-------:|:-------:|:-------:|
| `data/raw/`, `processed/`, `train/`, `val/`, `test/` | ✓ Creat | - | Actualizat* | - |
| `data/generated/` | - | ✓ Creat | - | - |
| `src/preprocessing/` | ✓ Creat | - | Actualizat* | - |
| `src/data_acquisition/` | - | ✓ Creat | - | - |
| `src/neural_network/model.py` | - | ✓ Creat | - | - |
| `src/neural_network/train.py`, `evaluate.py` | - | - | ✓ Creat | - |
| `src/neural_network/optimize.py`, `visualize.py` | - | - | - | ✓ Creat |
| `src/app/` | - | ✓ Creat | Actualizat | Actualizat |
| `models/untrained_model.*` | - | ✓ Creat | - | - |
| `models/trained_model.*` | - | - | ✓ Creat | - |
| `models/optimized_model.*` | - | - | - | ✓ Creat |
| `docs/state_machine.*` | - | ✓ Creat | - | (v2 opțional) |
| `docs/etapa3_analiza_date.md` | ✓ Creat | - | - | - |
| `docs/etapa4_arhitectura_SIA.md` | - | ✓ Creat | - | - |
| `docs/etapa5_antrenare_model.md` | - | - | ✓ Creat | - |
| `docs/etapa6_optimizare_concluzii.md` | - | - | - | ✓ Creat |
| `docs/confusion_matrix_optimized.png` | - | - | - | ✓ Creat |
| `docs/screenshots/` | - | ✓ Creat | Actualizat | Actualizat |
| `results/training_history.csv` | - | - | ✓ Creat | - |
| `results/optimization_experiments.csv` | - | - | - | ✓ Creat |
| `results/final_metrics.json` | - | - | - | ✓ Creat |
| **README.md** (acest fișier) | Draft | Actualizat | Actualizat | **FINAL** |

*\* Actualizat dacă s-au adăugat date noi în Etapa 4*

### Convenție Tag-uri Git

| Tag | Etapa | Commit Message Recomandat |
|-----|-------|---------------------------|
| `v0.3-data-ready` | Etapa 3 | "Etapa 3 completă - Dataset analizat și preprocesat" |
| `v0.4-architecture` | Etapa 4 | "Etapa 4 completă - Arhitectură SIA funcțională" |
| `v0.5-model-trained` | Etapa 5 | "Etapa 5 completă - Accuracy=X.XX, F1=X.XX" |
| `v0.6-optimized-final` | Etapa 6 | "Etapa 6 completă - Accuracy=X.XX, F1=X.XX (optimizat)" |

---

## 9. Instrucțiuni de Instalare și Rulare

### 9.1 Cerințe Preliminare

```
Python >= 3.8 (recomandat 3.10+)
pip >= 21.0
[sau LabVIEW >= 2020 pentru proiecte LabVIEW]
```

### 9.2 Instalare

```bash
# 1. Clonare repository
git clone [URL_REPOSITORY]
cd proiect-rn-[nume-prenume]

# 2. Creare mediu virtual (recomandat)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# sau: venv\Scripts\activate    # Windows

# 3. Instalare dependențe
pip install -r requirements.txt
```

### 9.3 Rulare Pipeline Complet

```bash
# Pasul 1: Preprocesare date (dacă rulați de la zero)
python src/preprocessing/data_cleaner.py
python src/preprocessing/data_splitter.py --stratify --random_state 42

# Pasul 2: Antrenare model (pentru reproducere rezultate)
python src/neural_network/train.py --config config/optimized_config.yaml

# Pasul 3: Evaluare model pe test set
python src/neural_network/evaluate.py --model models/optimized_model.h5

# Pasul 4: Lansare aplicație UI
streamlit run src/app/main.py
# sau: python src/app/main.py (pentru Flask/FastAPI)
# sau: [instrucțiuni LabVIEW dacă aplicabil]
```

### 9.4 Verificare Rapidă 

```bash
# Verificare că modelul se încarcă corect
python -c "from src.neural_network.model import load_model; m = load_model('models/optimized_model.h5'); print('✓ Model încărcat cu succes')"

# Verificare inferență pe un exemplu
python src/neural_network/evaluate.py --model models/optimized_model.h5 --quick-test
```

### 9.5 Structură Comenzi LabVIEW (dacă aplicabil)

```
[Completați dacă proiectul folosește LabVIEW]
1. Deschideți [nume_proiect].lvproj
2. Rulați Main.vi
3. ...
```

---

## 10. Concluzii și Discuții

### 10.1 Evaluare Performanță vs Obiective Inițiale

| Obiectiv Definit (Secțiunea 2) | Target | Realizat | Status |
|--------------------------------|--------|----------|--------|
| [Obiectiv 1 din 2.2] | [target] | [realizat] | [✓/✗] |
| [Obiectiv 2 din 2.2] | [target] | [realizat] | [✓/✗] |
| Accuracy pe test set | ≥70% | [X.XX%] | [✓/✗] |
| F1-Score pe test set | ≥0.65 | [X.XX] | [✓/✗] |
| [Metric specific domeniului] | [target] | [realizat] | [✓/✗] |

### 10.2 Ce NU Funcționează – Limitări Cunoscute

*[Fiți onești - evaluatorul apreciază identificarea clară a limitărilor]*

1. **Limitare 1:** [ex: Modelul eșuează pe imagini cu iluminare <50 lux - accuracy scade la 45%]
2. **Limitare 2:** [ex: Latența depășește 100ms pentru batch size >32 - neadecvat pentru real-time]
3. **Limitare 3:** [ex: Clasa "defect_minor" are recall doar 52% - date insuficiente]
4. **Funcționalități planificate dar neimplementate:** [ex: Export ONNX, integrare API extern]

### 10.3 Lecții Învățate (Top 5)

1. **[Lecție 1]:** [ex: Importanța EDA înainte de antrenare - am descoperit 8% valori lipsă care afectau convergența]
2. **[Lecție 2]:** [ex: Early stopping a prevenit overfitting sever - fără el, val_loss creștea după epoca 20]
3. **[Lecție 3]:** [ex: Augmentările specifice domeniului (zgomot gaussian calibrat) au adus +5% accuracy vs augmentări generice]
4. **[Lecție 4]:** [ex: Threshold-ul default 0.5 nu e optim pentru clase dezechilibrate - ajustarea la 0.35 a redus FN cu 40%]
5. **[Lecție 5]:** [ex: Documentarea incrementală (la fiecare etapă) a economisit timp major la integrare finală]

### 10.4 Retrospectivă

**Ce ați schimba dacă ați reîncepe proiectul?**

*[1-2 paragrafe: Decizii pe care le-ați lua diferit, cu justificare bazată pe experiența acumulată]*

[Completați aici]

### 10.5 Direcții de Dezvoltare Ulterioară

| Termen | Îmbunătățire Propusă | Beneficiu Estimat |
|--------|---------------------|-------------------|
| **Short-term** (1-2 săptămâni) | [ex: Augmentare date pentru clasa subreprezentată] | [ex: +10% recall pe clasa "defect_minor"] |
| **Medium-term** (1-2 luni) | [ex: Implementare model ensemble] | [ex: +3-5% accuracy general] |
| **Long-term** | [ex: Deployment pe edge device (Raspberry Pi)] | [ex: Latență <20ms, cost hardware redus] |

---

## 11. Bibliografie

*[Minimum 3 surse cu DOI/link funcțional - format: Autor, Titlu, Anul, Link]*

1. [Autor], [Titlu articol/carte], [Anul]. DOI: [link] sau URL: [link]
2. [Autor], [Titlu articol/carte], [Anul]. DOI: [link] sau URL: [link]
3. [Autor], [Titlu articol/carte], [Anul]. DOI: [link] sau URL: [link]
4. [Surse suplimentare dacă este cazul]

**Exemple format:**
- Abaza, B., 2025. AI-Driven Dynamic Covariance for ROS 2 Mobile Robot Localization. Sensors, 25, 3026. https://doi.org/10.3390/s25103026
- Keras Documentation, 2024. Getting Started Guide. https://keras.io/getting_started/

---

## 12. Checklist Final (Auto-verificare înainte de predare)

### Cerințe Tehnice Obligatorii

- [ ] **Accuracy ≥70%** pe test set (verificat în `results/final_metrics.json`)
- [ ] **F1-Score ≥0.65** pe test set
- [ ] **Contribuție ≥40% date originale** (verificabil în `data/generated/`)
- [ ] **Model antrenat de la zero** (NU pre-trained fine-tuning)
- [ ] **Minimum 4 experimente** de optimizare documentate (tabel în Secțiunea 5.3)
- [ ] **Confusion matrix** generată și interpretată (Secțiunea 6.2)
- [ ] **State Machine** definit cu minimum 4-6 stări (Secțiunea 4.2)
- [ ] **Cele 3 module funcționale:** Data Logging, RN, UI (Secțiunea 4.1)
- [ ] **Demonstrație end-to-end** disponibilă în `docs/demo/`

### Repository și Documentație

- [ ] **README.md** complet (toate secțiunile completate cu date reale)
- [ ] **4 README-uri etape** prezente în `docs/` (etapa3, etapa4, etapa5, etapa6)
- [ ] **Screenshots** prezente în `docs/screenshots/`
- [ ] **Structura repository** conformă cu Secțiunea 8
- [ ] **requirements.txt** actualizat și funcțional
- [ ] **Cod comentat** (minim 15% linii comentarii relevante)
- [ ] **Toate path-urile relative** (nu absolute: `/Users/...` sau `C:\...`)

### Acces și Versionare

- [ ] **Repository accesibil** cadrelor didactice RN (public sau privat cu acces)
- [ ] **Tag `v0.6-optimized-final`** creat și pushed
- [ ] **Commit-uri incrementale** vizibile în `git log` (nu 1 commit gigantic)
- [ ] **Fișiere mari** (>100MB) excluse sau în `.gitignore`

### Verificare Anti-Plagiat

- [ ] Model antrenat **de la zero** (weights inițializate random, nu descărcate)
- [ ] **Minimum 40% date originale** (nu doar subset din dataset public)
- [ ] Cod propriu sau clar atribuit (surse citate în Bibliografie)

---

## Note Finale

**Versiune document:** FINAL pentru examen  
**Ultima actualizare:** [DD.MM.YYYY]  
**Tag Git:** `v0.6-optimized-final`

---

*Acest README servește ca documentație principală pentru Livrabilul 1 (Aplicație RN). Pentru Livrabilul 2 (Prezentare PowerPoint), consultați structura din RN_Specificatii_proiect.pdf.*
