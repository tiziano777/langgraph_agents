from pipelines.tender_ner_pipeline import run_pipeline
text="""> Splošna bolnišnica Celje

OBLAKOVA ULICA 5
3000 CELJE

tel: (03) 42 33000

fax: (0B) 42 33 757

Postopek: EN N 26-2022-KT/Mš

 

 

POVABILO K ODDAJ I PONUDBE

 

Vabimo vas k oddaji ponudbe za predmet:
OGREVANI VOZIČEK ZA GN POSODE bain marie

- Na premičnem podstavku

- Spodnja polica

- 2 kolesi z možnostjo fiksiranja

- Kapaciteta 4 X GN 1/1 20 cm

- Priključek: 230 V / 2,8kw

- Mere 65 x 170 x 90 cm (SxGxV)

Osnova za realizacijo dobave je prejem pisnega naročila strani naročnika. Ponudbe zbiramo
informativno za predložitev v presojo odobritve. V kolikor bomo prejeli odobritev s strani
pristojnih bomo realizacijo izvedli pri najugodnejšemu ponudniku.

Cena mora vsebovati vse stroške, popuste, rabate in davek na dodano vrednost. Izvajalec mora
zagotoviti dostavo blaga skladišče nabavne službe v delovnem času za dostavo oz. prevzem

blaga (od ponedeljka do petka od 6:00 do 14 ure).
Rok za plačilo je 60 dni od prejema računa.

Merilo za izbora: najnižja vrednost ponudbe v EUR z DDV ob izpolnjevanju tehničnih zahtev
naročnika.

Kontaktna oseba s strani naročnika ;
- Klavdija Trunkl, uni. dipl.ekon., tel.št. 03 423 35 35, faks št. 03 423 37 56, elektronska pošta
klavdija.trunkl@ sb-celje.si

Ponudbe pošljite na naslov elektronske pošte

Nabava2@ sb-celje.si

z oznako EN N 26 OGREVANI VOZIČEK ZA GN POSODE bain marie 2022 do
najkasneje 25.3.2022 do 12 ure

Vodja nabavne službe
Matjaž Štinek,univ.dipl.ekon.
"""

run_pipeline(text)