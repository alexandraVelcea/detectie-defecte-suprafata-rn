### Interfața Grafică (GUI) - NEU-DET
=====================================

Acest modul implementează interfața web pentru sistemul de detecție a defectelor de suprafață, utilizând framework-ul **Streamlit**. Aplicația permite utilizatorilor să încarce imagini, să vizualizeze predicțiile modelului YOLOv8 în timp real și să analizeze metricile de antrenare.

## 1. Fișiere Componente
---------------------

1\. `app.py` (Logica Principală)

Acesta este nucleul aplicației. Gestionează:

-   **Interfața Utilizator (UI):** Layout-ul paginii, sidebar-ul, butoanele de upload și afișarea imaginilor.
-   **Inferența YOLO:** Încarcă modelul optimizat și rulează predicția pe imaginea încărcată.
-   **Vizualizarea Rezultatelor:** Desenează bounding boxes peste defectele detectate și generează tabelul cu scoruri de încredere.
-   **Dashboard Analytics:** Citește fișierele de log (`training_history.csv` și `test_metrics.json`) pentru a afișa grafice de performanță (Loss/Accuracy).

2\. `main.py` (Entry Point)

Un script wrapper ("launcher") care facilitează pornirea aplicației.
-   Determină automat calea către `app.py`.
-   Execută programatic comanda `streamlit run`, eliminând necesitatea de a tasta comenzi lungi în terminal.

3\. `ui.py` (Dummy UI)



## 2. Funcționalități Cheie
-----------------------

1.  **Selecție Automată a Modelului:** Aplicația caută modelele în ordinea priorității:

    1.  `defect_detector_ult` (Model "Ultimate" - Optimizat)
    2.  `yolov8n.pt` (Fallback - Model standard Nano)

2.  **Configurare în Timp Real:**

    -   **Confidence Threshold:** Slider pentru ajustarea sensibilității detecției (0.0 - 1.0).
    -   **Visual Settings:** Modificarea grosimii liniilor și a dimensiunii textului pentru vizibilitate optimă.

3.  **Analytics Dashboard:**

    -   Afișează metricile finale (Acuratețe, F1 Score, Latență) în sidebar.
    -   Generează grafice interactive pentru evoluția pierderii (Loss) și acurateței (mAP) pe durata antrenamentului.


## 3. Cerințe de Sistem și Structură
---------------------------------

Pentru ca aplicația să funcționeze corect și să încarce datele, structura proiectului trebuie să includă următoarele directoare relative la `app.py`:

Plaintext

```
project_root/
├── app.py                     # Scriptul principal
├── models/                    # Folderul cu modele antrenate
│   ├── defect_detector_AUG/weights/best.pt
│   └── defect_detector_ult/weights/best.pt
└── results/                   # Folderul cu rezultatele antrenamentului
    ├── training_history.csv   # Log-uri CSV pentru grafice
    └── test_metrics.json      # Metrici finale pentru sidebar

```

## 4.Cum se Rulează Aplicația
---------------------------

Există două metode de a porni interfața:

     Metoda 1: Folosind Launcher-ul (Recomandat)

Dacă structura permite, rulați scriptul Python standard:

bash

```
python main.py

```

     Metoda 2: Comanda directă Streamlit

Din terminal, în directorul unde se află `app.py`:

bash

```
streamlit run app.py

```


## 5. Interpretarea Rezultatelor din Interfață
-------------------------------------------

-   **Input Image:** Imaginea originală încărcată de utilizator.
-   **Result Image:** Imaginea procesată, unde defectele sunt marcate cu chenare colorate (fiecare clasă are o culoare unică generată aleatoriu).
-   **Tabel Detecții:** Lista detaliată a defectelor, conținând Tipul (Clasa) și Scorul de Încredere (Confidence %).
-   **Training Analytics:**
    -   *Loss Curves:* Arată cât de bine a învățat modelul (valorile mai mici sunt mai bune).
    -   *Accuracy Curves:* Arată performanța mAP@50 (valorile mai mari sunt mai bune).