#!/bin/bash

# Script pentru corectarea structurii de foldere YOLO
# Muta imaginile din subfoldere (categorii) in folderul parinte images

# Verificam daca suntem in radacina proiectului
if [ ! -d "data/raw" ]; then
    echo "❌ EROARE: Nu gasesc folderul 'data/raw'."
    echo "Te rog ruleaza acest script din radacina proiectului (folderul 'alexandrasmechera')."
    exit 1
fi

echo "🚀 Incep repararea structurii folderelor..."

# --- FUNCTIE PENTRU APLATIZARE ---
flatten_directory() {
    TARGET_PATH="$1"
    
    if [ -d "$TARGET_PATH" ]; then
        echo "📂 Procesez: $TARGET_PATH"
        
        # Intram in director
        cd "$TARGET_PATH" || exit

        # Verificam daca exista subfoldere cu imagini jpg
        if ls */*.jpg 1> /dev/null 2>&1; then
            # Mutam toate imaginile .jpg din subfoldere in folderul curent
            mv */*.jpg .
            echo "   ✅ Imaginile au fost mutate."
            
            # Stergem folderele goale (crazing, patches, etc.)
            # 'rmdir *' va sterge doar folderele goale si va da eroare (ignorabila) pentru fisiere
            rmdir * 2>/dev/null
            echo "   ✅ Subfolderele goale au fost sterse."
        else
            echo "   ⚠️ Nu am gasit imagini in subfoldere sau structura e deja corecta."
        fi
        
        # Ne intoarcem la locatia initiala pentru urmatorul pas
        cd - > /dev/null || exit
    else
        echo "❌ EROARE: Calea $TARGET_PATH nu exista!"
    fi
    echo "---------------------------------------------------"
}

# --- EXECUTIE ---

# 1. Reparam TRAIN
flatten_directory "data/raw/train/images"

# 2. Reparam VALIDATION
flatten_directory "data/raw/validation/images"

echo "🎉 GATA! Structura a fost corectata."
echo "Acum poti rula din nou: python src/neural_network/train.py"
