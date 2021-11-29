Diplomski rad
---

#### take_photo.py
````
python take_photo.py -n osoba
````
Pokreće se python skripta s obaveznim parametrom **-n** ili **--name** iza ime fotografije.
1 fotografija će spremljena na trenutnoj putanji kao **osoba.jpg**

#### photobooth.py
````
python photobooth.py -n osoba 1
````
Pokreće se python skripta s obaveznim parametrom **-n** ili **--name** iza kojeg se upisuje ime osobe.
200 fotografija će okinuti i spremiti u odgovarajuće train/valid direktorije **osoba_1**


#### model_training.py
````
python model_training.py
````
Pokreće se python skripta za treniranje modela na skupu fotografija koje se nalaze u
**dataset** direktoriju. Po završetku treninga se kreira model.h5 datoteka


#### face_recognition.py
````
python face_recognition.py --lcd
python face_recognition.py --cli
````
Pokreće se python skripta za prepoznavanje lica. Kamera na računalu mora biti upaljena.
- Ako je postavljen parametar **lcd** obavezno zaslon mora biti spojen na RPi, ispis će biti na zaslonu
- Ako je postavljen parametar **cli** ispis će biti isključivo u naredbenom retku