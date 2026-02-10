### Scripturi utilitare pentru gestionarea datelor
==================================================

Acest modul conține o colecție de scripturi Python esențiale pentru organizarea, convertirea, curățarea și analiza setului de date NEU-DET. Acestea asigură tranziția de la datele brute (XML) la formatul necesar pentru antrenarea YOLO (TXT) și permit managementul datelor sintetice.

## 1. Cuprins
----------

1.  [`xml_to_txt.py`] - **Critic** (Pregătește datele pentru antrenare)
2.  [`augmentation_stats.py`] - Raportare și grafice.
3.  [`utility_delete.py`] - Resetare dataset.
4.  [`reorganize_structure.py`] - Setup inițial.
5.  [`mv.py`] - Izolare date generate.

* * * * *

1\. Conversie Format YOLO (`xml_to_txt.py`)
-------------------------------------------

Acesta este cel mai important script după augmentare. YOLOv8 nu poate citi fișiere `.xml` (Pascal VOC); are nevoie de fișiere `.txt` cu coordonate normalizate.

-   **Ce face:**
    -   Citește fișierele XML din folderele `annotations`.
    -   Convertește coordonatele (xmin, ymin, xmax, ymax) în format YOLO (x_center, y_center, width, height) normalizat [0-1].
    -   Generează folderul `labels` paralel cu `images`.
    -   Creează automat fișierul de configurare **`data.yaml`** necesar antrenării.

-   **Când se rulează:** După ce ai generat datele (augmentare) și înainte de antrenare.

-   **Comandă:**

    Bash

    ```
    python src/data_management/xml_to_txt.py

    ```

2\. Generare Statistici (`augmentation_stats.py`)
-------------------------------------------------

Oferă o privire de ansamblu asupra compoziției setului de date, permițând verificarea echilibrului dintre datele reale și cele sintetice.

-   **Ce face:**
    -   Scanează folderele `train`, `validation` și `test`.
    -   Numără imaginile reale vs. cele sintetice (identificate prin prefixul `aug_`).
    -   Generează un raport CSV: `docs/data_statistics.csv`.
    -   Generează un grafic tip bară: `docs/generated_vs_real.png`.

-   **Comandă:**

    Bash

    ```
    python src/data_management/augmentation_stats.py

    ```

3\. Curățare Date Sintetice (`utility_delete.py`)
-------------------------------------------------

Un script de "curățenie" generală. Este util dacă ai greșit parametrii de augmentare și vrei să o iei de la capăt, păstrând doar imaginile originale.

-   **Ce face:**

    -   Caută recursiv în tot folderul `data/`.
    -   Șterge **orice fișier** (imagine, xml, txt) care începe cu prefixul `aug_`.
    -   Include o confirmare de siguranță (y/n) pentru a preveni ștergerile accidentale.

-   **Comandă:**

    Bash

    ```
    python src/data_management/utility_delete.py

    ```

4\. Organizare Inițială (`reorganize_structure.py`)
---------------------------------------------------

Folosit la începutul proiectului pentru a aduce datele descărcate (care pot avea structuri imbricate complexe) în formatul standard `train/val`.

-   **Ce face:**

    -   Mută fișierele din structura brută (ex: `data/raw/NEU-DET/...`) în folderele curate `data/raw/train` și `data/raw/validation`.

-   **Comandă:**

    Bash

    ```
    python src/data_management/reorganize_structure.py

    ```

5\. Backup Date Augmentate (`mv.py`)
------------------------------------

Util dacă dorești să izolezi doar datele generate sintetic pentru a le inspecta vizual separat sau pentru a le arhiva, fără a le amesteca cu cele originale.

-   **Ce face:**

    -   Copiază toate fișierele care încep cu `aug_` (imagini, XML, TXT) din `data/train`.
    -   Le salvează într-un folder nou: `data/generated/`.

-   **Comandă:**

    Bash

    ```
    python src/data_management/mv.py

    ```

* * * * *

## 2. Flux de Lucru Recomandat (Pipeline)
--------------------------------------

Pentru a pregăti corect datele pentru antrenament, ordinea execuției este:

1.  **Organizare:** `reorganize_structure.py` (O singură dată, la început).
2.  **Augmentare:** `generate_augmentation.py` (Scriptul principal de generare).
3.  **Conversie:** `xml_to_txt.py` (Esențial pentru YOLO).
4.  **Verificare:** `augmentation_stats.py` (Pentru a valida distribuția în `docs/`).
5.  **Antrenare:** `train.py`.
