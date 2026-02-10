### NEU-DET 

## 1. Prezentare Generală
----------------------

Acest set de date reprezintă o versiune procesată a **NEU-DET (Northeastern University Surface Defect Database)**, optimizată pentru sarcini de detecție a obiectelor folosind **YOLOv8**. Conține imagini cu benzi de oțel laminate la cald, prezentând șase tipuri tipice de defecte de suprafață.

Setul de date a fost supus următoarelor procese:

1.  **Împărțire (Split):** Divizare în seturi de Antrenare (Train), Validare (Validation) și Testare (Test).

2.  **Conversie:** Transformare din formatul original XML (Pascal VOC) în format **TXT (YOLO)**.

3.  **Augmentare:** Generare de date sintetice pentru a crește robustețea modelului la variabilitatea din producție.

Structura Setului de Date
-----------------------------

Directorul este organizat conform structurii standard Ultralytics YOLO:

Plaintext

```
data/
├── data.yaml            # Fișier de configurare pentru antrenarea YOLO
├── train/               # Set de Antrenare (Învățare)
│   ├── images/          # Imagini .jpg / .png
│   └── labels/          # Fișiere de adnotare .txt
├── validation/          # Set de Validare (Reglare & Evaluare în timpul antrenării)
    ├── images/
    └── labels/

```

## 2. Clase de Defecte
-------------------

Setul de date include **6 clase** de defecte de suprafață. Indicii claselor din fișierele de etichetă corespund următoarelor tipuri:

| **ID** | **Nume Clasă** | **Descriere** |
| --- | --- | --- |
| `0` | **crazing** | Micro-fisuri complexe pe suprafață. |
| `1` | **inclusion** | Particule străine încorporate în metal. |
| `2` | **patches** | Zone discrete de iregularitate a suprafeței. |
| `3` | **pitted_surface** | Mici cratere sau găuri cauzate de laminare. |
| `4` | **rolled-in_scale** | Tunder (oxid de fier) presat în suprafață. |
| `5` | **scratches** | Abraziuni liniare (zgârieturi) cauzate de contact mecanic. |

## 3. Formatul Adnotărilor (YOLO)
------------------------------

Adnotările sunt stocate în fișiere `.txt` (unul pentru fiecare imagine). Fiecare linie reprezintă un obiect în formatul:

Plaintext

```
<class-id> <x_center> <y_center> <width> <height>

```

-   **class-id**: Număr întreg de la 0 la 5.

-   **coordinates**: Valori normalizate (între 0 și 1) relative la dimensiunile imaginii.


## 4. Statistici Dataset
---------------------

-   **Sursa Originală:** Northeastern University (NEU)

-   **Total Eșantioane:** ~1,800 imagini (Original) + Eșantioane Augmentate

-   **Rezoluție:** 200x200 px (Original) -> Redimensionat/Augmentat pentru antrenare (ex: 832px).

-   **Format:** Grayscale (convertit la RGB pentru compatibilitatea cu modelul).

## 5. Referințe Bibliografice
--------------------------


> **K. Song and Y. Yan**, "A noise robust method based on deep learning for defect detection of steel surface," in *IEEE Transactions on Instrumentation and Measurement*, 2013.

```