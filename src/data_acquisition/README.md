### Generare Augmentare și Împărțire Dataset (`generate_augmentation.py`)
=========================================================================

Acest script reprezintă **Etapa 1** în pipeline-ul de pregătire a datelor. Rolul său este de a prelua imaginile brute (RAW), de a le distribui în seturi de Antrenare, Validare și Testare, și de a **genera date sintetice** pentru a îmbogăți setul de date final.

## 1. Descriere Funcțională
------------------------

Scriptul realizează automat următoarele operațiuni:

1.  **Agregarea Datelor:** Colectează toate imaginile și adnotările XML din folderele `data/raw/train` și `data/raw/validation` într-un singur "pool" comun.
2.  **Împărțirea Aleatorie (Splitting):** Distribuie imaginile originale în trei seturi distincte, asigurându-se că datele nu se suprapun:
    -   **Test:** Număr fix de originale (ex: 30 imagini/clasă).
    -   **Validation:** Număr fix de originale (ex: 30 imagini/clasă).
    -   **Train:** Restul imaginilor disponibile.
3.  **Generare Sintetică (Augmentare):** Pentru fiecare set (Train/Val/Test), scriptul generează imagini noi artificiale până când acestea reprezintă **40%** din totalul datelor.
4.  **Gestionarea Adnotărilor:** Copiază fișierele XML originale și creează automat fișiere XML noi pentru imaginile generate, calculând noile coordonate (Bounding Boxes).

## 2. Tehnici de Augmentare Utilizate
----------------------------------

Spre deosebire de augmentarea clasică (doar rotiri/flip), acest script folosește **simularea procedurală a defectelor**. Scriptul desenează defecte direct pe imagini folosind biblioteca `PIL` și `numpy`.

| **Funcție** | **Descriere Tehnică** |
| --- | --- |
| `general_augmentations` | Ajustări aleatorii de luminozitate (0.8-1.2) și contrast (0.9-1.3). |
| `simulate_scratches` | Desenează linii aleatorii de culori variabile pentru a simula zgârieturi. |
| `simulate_pitted_surface` | Generează clustere de elipse mici pentru a imita suprafețele poroase/ciupite. |
| `simulate_patches` | Aplică suprapuneri (overlay) translucide pentru a simula pete de oxidare sau ulei. |
| `simulate_inclusion` | Desenează linii groase și închise la culoare pentru a imita incluziunile metalice. |
| `simulate_rolled_in_scale` | Generează forme neregulate difuze pentru a imita tunderul laminat. |
| `simulate_crazing` | Creează o rețea de linii fine interconectate (pânză de păianjen) pentru micro-fisuri. |

## 3. Configurare
--------------

Parametrii principali pot fi modificați direct în secțiunea `CONFIGURATION` a scriptului:

Python

```
# Numărul de imagini reale păstrate pentru validare și testare per clasă
TARGET_VAL_ORIGINALS  = 30
TARGET_TEST_ORIGINALS = 30

# Procentul țintă de date augmentate în setul final (0.40 = 40%)
AUG_TARGET_RATIO = 0.40

```

## 4. Utilizare
------------

Scriptul trebuie rulat din rădăcina proiectului sau asigurându-se că variabila `PROJECT_ROOT` este rezolvată corect.

Bash

```
python src/data_management/generate_augmentation.py

```

## 5. Structura Output
-------------------

După rulare, folderul `data/` va fi populat cu directoarele finale organizate, gata pentru conversia la formatul YOLO:

Plaintext

```
data/
├── train/
│   ├── images/      # Conține originale + imagini 'aug_*.jpg'
│   └── annotations/ # Conține XML-uri originale + XML-uri generate
├── validation/
│   ├── images/
│   └── annotations/
└── test/
    ├── images/
    └── annotations/

```

Notă Importantă
------------------

Acest script generează adnotări în format **Pascal VOC (.xml)**. Pentru a antrena modelul YOLOv8, este necesară rularea ulterioară a scriptului de conversie (`convert_xml_to_yolo.py`) care transformă aceste XML-uri în fișiere `.txt`.