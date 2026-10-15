"""Both sides of the header question, written independently.

A rewrite that measured only the headerless side took the error rate on
ordinary headed files from 5% to 25% and shipped. The rewrite after it,
measured against the corpus below as it then stood, scored 34 of 34 — and an
independently written 62-file corpus put it at 19% wrong on headed files and
35% on headerless, worse on both sides than the version it replaced.

That is what a corpus fitted to the mechanism is worth. Every entry marked
`[independent]` was named by someone who did not write the code, and the
lists must not be trimmed to make a change pass: they are what the change is
measured against.

The rule the misses teach: a header can be told from a record only where the
two differ in a way that is not a coincidence. A column of dates, of numbers
of one width, of two-letter codes, of fixed-width identifiers — agreement
there means something. A column of four-letter Title-case words agrees with a
four-letter Title-case header by accident, and no amount of character
analysis recovers the difference. Where nothing informative is available the
answer is a guess, and the guess is "header", because most files have one.
The four in ACCEPTED_AS_HEADED are what that guess costs.
"""

#: Files that do have a header row.
WITH_HEADER = [
    ("zwei Woerter, Titelfall", "Name,City\nAnna,Berlin\nBeth,Madrid\nCara,Vienna\n"),
    ("zwei Woerter, Grossbuchstaben", "NAME,CITY\nAnna,Berlin\nBeth,Madrid\nCara,Vienna\n"),
    ("Geographie", "City,Country\nBerlin,Germany\nMadrid,Spain\nVienna,Austria\n"),
    ("Vor- und Nachname", "First,Last\nAnna,Meier\nBeth,Klein\nCara,Roth\n"),
    ("Artikel und Ort", "Item,Town\nWidget,Berlin\nGadget,Madrid\nBolt,Vienna\n"),
    ("klein geschrieben", "name,city\nAnna,Berlin\nBeth,Madrid\nCara,Vienna\n"),
    ("mit Zahlenspalte", "region,units,revenue\nNorth,120,4500\nSouth,90,3200\nEast,150,6100\n"),
    ("id voran", "id,name,city\n1,Ann,Berlin\n2,Bob,Paris\n3,Cid,Rome\n"),
    ("Datumsspalte", "date,amount,payee\n2026-01-03,-42.50,Rewe\n2026-01-04,-12.90,Bahn\n2026-01-05,1500.00,Salary\n"),
    ("pandas-Index", ",name,value\n0,a,10\n1,b,20\n2,c,30\n"),
    ("Jahre als Namen", "2020,2021,2022\n10,20,30\n40,50,60\n70,10,20\n"),
    ("doppelte Namen", "date,value,date,value\n2026-01-01,10,2026-02-01,20\n2026-01-02,11,2026-02-02,21\n2026-01-03,12,2026-02-03,22\n"),
    ("deutsche Kopfzeile", "Artikel;Preis;Menge\nSchraube;0,45;1200\nMutter;0,12;3400\nScheibe;0,08;5000\n"),
    ("snake_case", "first_name,last_name,email\nAnna,Meier,a@x\nBeth,Klein,b@x\nCara,Roth,c@x\n"),
    ("mit Einheiten", "Weight (kg),Height (cm)\n71.5,180\n64.2,165\n88.0,192\n"),
    ("boolesche Spalte", "user,active\nanna,true\nbeth,false\ncara,true\n"),
    ("eine Spalte", "email\na@x.example\nb@x.example\nc@x.example\n"),
    ("Grossbuchstaben-Codes", "CODE,LABEL\nDE,Germany\nFR,France\nIT,Italy\n"),
    ("langer Kopf, kurze Werte", "identifier,quantity\nA1,5\nB2,7\nC3,9\n"),
    ("kurzer Kopf, lange Werte", "id,description\n1,A very long descriptive sentence here\n2,Another equally long sentence\n3,And a third one for good measure\n"),
    ("R write.csv", '"x","y"\n"a",1,4\n"b",2,5\n"c",3,6\n'),
    ("Nachkommastelle", "sku,price\nA1,19.99\nB2,24.50\nC3,7.25\n"),
    # [independent] Eight languages of ordinary spreadsheet whose header
    # words are in no list sift owns, and whose columns hold nothing a shape
    # test can speak about. Every one lost its column names to a rule that
    # guessed instead of abstaining.
    ("spanisch", "Descripcion,Categoria\nTornillo,Ferreteria\nTuerca,Ferreteria\nArandela,Ferreteria\nMartillo,Herramientas\n"),
    ("spanisch, zweite", "Descripcion,Emisor\nFactura,Endesa\nRecibo,Iberdrola\nNota,Repsol\n"),
    ("franzoesisch", "Fournisseur,Departement\nRenault,Achats\nMichelin,Logistique\nAirbus,Production\n"),
    ("italienisch", "Reparto,Responsabile\nVendite,Rossi\nAcquisti,Bianchi\nTecnico,Verdi\n"),
    ("polnisch", "Nazwisko,Stanowisko\nKowalski,Inzynier\nNowak,Ksiegowa\nWojcik,Kierowca\n"),
    ("ungarisch", "Telepules,Megye\nSzeged,Csongrad\nPecs,Baranya\nGyor,Gyor\n"),
    ("deutsch, ungelistet", "Ansprechpartner,Zustaendigkeit\nMeier,Einkauf\nKlein,Vertrieb\nRoth,Technik\n"),
    ("Fachjargon", "Instrument,Venue\nBund,Eurex\nGilt,ICE\nOAT,Euronext\n"),
    ("Abkuerzungen", "dt,amt,qty_x\n2026-01-01,10.5,3\n2026-01-02,11.0,4\n2026-01-03,9.75,2\n"),
    ("Wertpapierkennungen", "cusip,isin,figi\n037833100,US0378331005,BBG000B9XRY4\n"
                            "594918104,US5949181045,BBG000BPH459\n"
                            "023135106,US0231351067,BBG000BVPV84\n"),
]

#: Files that do not. Every one of them loses a record if row one is
#: mistaken for column names.
WITHOUT_HEADER = [
    ("Personen", "Ann,Berlin,DE\nBob,Paris,FR\nCid,Rome,IT\nDee,Oslo,NO\nEve,Madrid,ES\n"),
    ("Artikelnummern", "SKU-1,widget,12\nSKU-2,gadget,7\nSKU-3,doohickey,3\nSKU-4,thing,19\n"),
    ("Bankauszug", "2026-01-03;-42,50;Rewe Markt\n2026-01-04;-12,90;Bahn\n2026-01-05;1500,00;Gehalt\n2026-01-06;-89,00;Stadtwerke\n2026-01-07;-7,20;Baecker\n"),
    ("Zahlenmatrix", "1,2,3\n4,5,6\n7,8,9\n10,11,12\n"),
    ("absteigend sortiert", "1000,2000\n1,2\n2,3\n3,4\n5,6\n"),
    ("Koordinaten", "52.5200,13.4050\n40.4168,-3.7038\n48.2082,16.3738\n41.9028,12.4964\n"),
    ("Zeitreihe", "2026-01-01,10\n2026-01-02,11\n2026-01-03,12\n2026-01-04,13\n"),
    ("Laendercodes", "DE,Germany\nFR,France\nIT,Italy\nES,Spain\nNL,Netherlands\n"),
    ("Namen und Alter", "Ann,31\nBob,42\nCid,25\nDee,38\nEve,29\n"),
    ("Protokollzeilen", "GET,/index.html,200\nPOST,/api/login,401\nGET,/static/a.css,200\nGET,/favicon.ico,404\n"),
    ("Stadt und Einwohner", "Berlin,3664088\nHamburg,1852478\nMuenchen,1488202\nKoeln,1073096\n"),
    ("Paarliste", "alpha,1\nbeta,2\ngamma,3\ndelta,4\nepsilon,5\n"),
    # [independent] Records whose first row happens to contain a word that
    # names a column somewhere else. A two-column file needs one such word
    # for a vocabulary rule to take its first record away.
    ("Dienste und Ports", "web,80\nssh,22\nftp,21\nsmtp,25\ndns,53\n"),
    ("Postfaecher", "mail,25\nnews,119\nchat,5222\nvoice,5060\n"),
    ("Tankstellen", "Total,1.899\nShell,1.929\nAral,1.879\nEsso,1.909\nJet,1.859\n"),
    ("Glossar", "no,nein\nyes,ja\nmaybe,vielleicht\nnever,nie\nalways,immer\n"),
    ("Schlagwortzaehlung", "web,120\nmobile,90\ncloud,75\nsecurity,60\ndata,45\n"),
    ("Flugtarife", "First,Class,1200,USD\nBusiness,Class,800,USD\nEconomy,Plus,400,USD\nEconomy,Basic,250,USD\n"),
    ("Kennungen und Zustand", "3f2b1c-aa,active\n8d4e2f-bb,inactive\n1a9c7e-cc,active\n5b3d8f-dd,pending\n"),
    ("Nachname als Wert", "Name,Hamburg\nSchmidt,Berlin\nMueller,Muenchen\nWeber,Koeln\n"),
]

#: [independent, round 15] Headed files whose header matches its column's
#: exact length and case class — which the third rewrite took as proof of
#: *headerlessness*. Uppercase codes, fixed-width logins and words of equal
#: length are exactly where a header agrees by construction, and the file
#: where it agrees is disproportionately a headed one.
WITH_HEADER += [
    ("Flughafenkennungen", "ICAO,Name,Country\nKLAX,Los Angeles,US\n"
                           "KSFO,San Francisco,US\nEGLL,Heathrow,GB\n"
                           "EDDF,Frankfurt,DE\n"),
    ("Waehrungen", "ISO,Currency,Minor\nUSD,Dollar,2\nEUR,Euro,2\n"
                   "GBP,Pound,2\nJPY,Yen,0\n"),
    ("Anmeldenamen", "userid,department,office\njsmith,finance,berlin\n"
                     "mjones,legal,munich\nbwhite,sales,hamburg\n"
                     "agreen,tech,cologne\n"),
    ("gleich lange Woerter", "Country,Capital\nGermany,Berlin\n"
                             "Austria,Vienna\nFinland,Helsinki\n"
                             "Denmark,Copenhagen\n"),
    ("schwedische Nachnamen", "Efternamn,Ort,Postnummer\n"
                              "Andersson,Malmoe,21120\n"
                              "Lindqvist,Uppsala,75236\n"
                              "Bergstrom,Goeteborg,41118\n"),
    ("Jahres-Pivot", "2022,2023,2024\n1388,1402,1455\n1390,1410,1440\n"
                     "1395,1420,1450\n"),
]

#: [independent, round 15] Headerless files of exactly three lines. A floor of
#: three body values meant the shape test could never speak on a 3-line table,
#: and the fallback favours "header" — 23 of 24 such files lost their first
#: record. A 3-line CSV is a filtered query result, a receipt, a config dump.
WITHOUT_HEADER += [
    ("drei Zeilen, Kurse", "AAPL,189.45\nMSFT,411.22\nNVDA,878.10\n"),
    ("drei Zeilen, Codes", "DE,Germany\nFR,France\nIT,Italy\n"),
    ("drei Zeilen, Datum", "2026-01-01,10\n2026-01-02,11\n2026-01-03,12\n"),
    ("Luecken in der Spalte", "Ann Meier,54000\nBob Klein,\nCid Roth,\n"
                              "Dee Fischer,72000\n"),
]

#: Headerless files sift knowingly reads as headed. Each is a two-column
#: file whose first record is, in shape, indistinguishable from the records
#: below it: no date, no number of fixed width, no code, no identifier — two
#: short words over two short words. The fallback in that situation favours
#: "header", and it is right to, because most CSV files have one. Recorded
#: rather than hidden: a heuristic with no admitted cost is one whose cost
#: has not been looked for.
ACCEPTED_AS_HEADED = {
    "Postfaecher",           # mail,25 over news,119 / chat,5222 / voice,5060
    "Glossar",               # no,nein over yes,ja / maybe,vielleicht
    "Schlagwortzaehlung",    # web,120 over mobile,90 / cloud,75
    "Nachname als Wert",     # Name,Hamburg over Schmidt,Berlin
    # And one where the evidence is missing rather than ambiguous: the only
    # column that could speak has one usable value, the rest being blank.
    "Luecken in der Spalte",
}

#: Files whose failure is not about the header row at all. Each is a
#: (name, content, checks) triple; `checks` names what must be true.
STRUCTURE = [
    # [independent, round 15]
    ("600 Spalten bleiben 600 oder werden benannt",
     ",".join(f"f{i:03d}" for i in range(600)) + "\n"
     + "\n".join(",".join(str(i) for i in range(600)) for _ in range(20)) + "\n",
     {"no_false_ragged": True}),
    ("Kommaspalte mit Einheiten",
     "patient,visits,dosage; route\nP001,3,50mg daily; oral\n"
     "P002,5,25mg twice; oral\nP003,2,100mg weekly; iv\n"
     "P004,4,10mg nightly; oral\n",
     {"delimiter": ",", "columns": 3}),
    ("deutsche Gewichte mit einem US-Wert",
     "Artikel;Gewicht;Menge\nSchraube;0,500;1200\nMutter;0,250;3400\n"
     "Scheibe;1,000;5000\nBolzen;1.234,50;900\n",
     {"column": 1, "min": 0.25, "max": 1234.5}),
    ("R-Matrix mit langen Zeilennamen",
     '"S1","S2","S3"\n"TP53",4.2,5.1,3.3\n"BRCA1",2.8,3.0,2.9\n'
     '"EGFR",7.7,8.1,7.4\n"MYC",1.2,1.5,1.1\n',
     {"header": True}),
    ("Komma am Ende der Kopfzeile",
     "id,name,city,,\n1,Ann,Berlin\n2,Bob,Paris\n3,Cid,Rome\n4,Dee,Oslo\n",
     {"columns": 3, "no_false_ragged": True}),
    ("Titelzeile ueber Messreihe",
     "Messreihe Februar 2026\n21,4;44,2\n21,6;43,8\n21,9;43,1\n",
     {"delimiter": ";", "columns": 2, "rows": 3}),
    ("Titelzeile mit zwei Feldern",
     "Monthly Sales,2026-01\nregion,units,revenue\nNorth,120,4500\n"
     "South,90,3200\nEast,150,6100\n",
     {"names": ["region", "units", "revenue"], "rows": 3}),
    ("Titelzeile mit einem Feld",
     "Employee Report\nAnn,Berlin,31\nBob,Madrid,42\nCid,Vienna,25\n"
     "Dee,Warsaw,38\n",
     {"rows": 4, "preamble": ["Employee Report"]}),
    ("kurzer erster Satz",
     "2026-01-01,GET,/a\n2026-01-02,GET,/b,200\n2026-01-03,GET,/c,200\n"
     "2026-01-04,GET,/d,200\n2026-01-05,GET,/e,200\n",
     {"rows": 5, "header": False}),
    ("Kopfzeile breiter als die Daten",
     "id,name,city,notes,extra\n1,Ann,Berlin\n2,Bob,Paris\n3,Cid,Rome\n"
     "4,Dee,Oslo\n",
     {"names": ["id", "name", "city", "notes", "extra"], "rows": 4}),
    ("zerrissene Zeile behaelt ihr Feld",
     "id,name,city\n1,Ann,Berlin\n2,Bob,Madrid\n3,Smith, John,Rome\n"
     "4,Cid,Vienna\n5,Dee,Oslo\n",
     {"rows": 5, "keeps": "Rome"}),
    ("Kommas im Textfeld gegen Semikolon",
     "Artikel, Typ, Norm, Werkstoff;Preis;Menge\n"
     "Schraube, verzinkt, M8, A2;0,45;1200\n"
     "Mutter, verzinkt, M8, DIN934;0,12;3400\n"
     "Scheibe, verzinkt, A2, DIN125;0,08;5000\n"
     "Bolzen, verzinkt, M10, DIN931;1,20;900\n",
     {"delimiter": ";", "columns": 3}),
    ("sep= mit einer zerrissenen Zeile",
     "sep=;\n2026-01-03;-42,50;Rewe\n2026-01-04;-12,90;Bahn\n"
     "2026-01-05;1500,00;Gehalt;extra\n2026-01-06;-89,00;Stadtwerke\n"
     "2026-01-07;-7,20;Baecker\n",
     {"delimiter": ";", "rows": 5}),
    ("sep= mit einer Spalte",
     "sep=,\nemail\na@x.example\nb@x.example\nc@x.example\n",
     {"delimiter": ",", "columns": 1}),
    ("amerikanische Tausender mit Cents",
     'sku,price\nA1,"1,234.50"\nB2,"999.00"\nC3,"12,000"\nD4,"8,336,817"\n',
     {"min": 999.0, "max": 8336817.0}),
]

#: Files that are not tables at all, whatever they declare.
NOT_TABLES = [
    ("Prosa mit sep=", "note.txt",
     "sep=,\nFirst, we ship the thing.\nSecond, we watch it closely.\n"
     "Third, we write the note.\nFourth, we ship again.\n"),
    ("Prosa mit sep=|", "note2.txt",
     "sep=|\nHello there, this is a note.\nSecond line of the note.\n"
     "Third line here.\nFourth line.\n"),
]
