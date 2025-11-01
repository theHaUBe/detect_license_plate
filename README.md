# Wykrywanie i Rozpoznawanie Tablic Rejestracyjnych

Projekt służy do automatycznego wykrywania tablic rejestracyjnych na obrazach, ekstrakcji poszczególnych znaków oraz dopasowywania ich do wzorców w celu odczytu numeru tablicy. Wyniki są zapisywane w pliku JSON.

## Funkcjonalności

1. **Ładowanie szablonów znaków**
   - Funkcja `load_templates` wczytuje obrazy szablonów liter i cyfr (A-Z, 0-9) z folderu `templates`.
   - Szablony wykorzystywane są do dopasowania znaków wykrytych na tablicy rejestracyjnej.

2. **Wykrywanie tablic rejestracyjnych**
   - Funkcja `detect_license_plate` stosuje:
     - filtr bilateralny do wygładzenia obrazu bez utraty krawędzi,
     - konwersję do przestrzeni kolorów HSV,
     - maskę koloru białego i wyszukiwanie konturów.
   - Wybierane są prostokątne obszary o określonym zakresie powierzchni, które mogą odpowiadać tablicy.
   - Wykorzystuje transformację perspektywiczną, aby uzyskać prostokątny obraz tablicy.
   - Wykrywa krawędzie metodą Canny'ego i kontury poszczególnych znaków.

3. **Dopasowywanie znaków do wzorców**
   - Funkcja `match_template_on_contour` dopasowuje wykryty znak do szablonów za pomocą `cv2.matchTemplate`.
   - Wybiera najlepsze dopasowanie na podstawie współczynnika korelacji.

4. **Zapis wyników**
   - Rozpoznane tablice rejestracyjne są zapisywane w pliku JSON w formacie:
     ```json
     {
         "nazwa_pliku.jpg": "ROZPOZNANY_TEKST",
         ...
     }
     ```

## Użycie

Uruchom skrypt podając folder z obrazami do przetworzenia oraz nazwę pliku wyjściowego JSON:

```bash
python script.py <path_to_image_folder> <output_json_file>
```

Skrypt przetworzy wszystkie obrazy w folderze images i zapisze rozpoznane tablice rejestracyjne w results.json.
