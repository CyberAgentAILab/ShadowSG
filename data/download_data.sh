#!/bin/bash

pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1cn9Y0F1Chib4OQaB4lcypIfsmr0dRgG9" -O ./data

for file in $(find ./data/*/* -name "*.zip"); do
    unzip -d "$(dirname "$file")" "$file" && rm "$file"
done
