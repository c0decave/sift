"""Passages for measuring detect_natural_language, in both directions.

Written for this corpus, in the shapes files actually take: an invoice line,
a mail, a README paragraph, a meeting note, a log message, a contract clause,
a recipe. GERMAN and ENGLISH are the two answers the function may give;
NEITHER is text it must decline — other languages (Dutch and Danish share
function words with German, French and Spanish with English), code, tables,
numbers — because a wrong language picks the wrong stemmer for the full-text
index, and no language picks the neutral one.

Each list has a HELD_OUT twin that was kept aside while the thresholds were
set and never used to set them. The numbers reported for the held-out lists
are the ones that say something about text the function was not tuned on.
"""

GERMAN = [
    "Sehr geehrte Damen und Herren, anbei erhalten Sie die Rechnung für den "
    "Monat März. Bitte überweisen Sie den Betrag bis zum 15. April auf das "
    "unten genannte Konto.",
    "Die Besprechung wurde auf Donnerstag verschoben, weil der Raum am "
    "Mittwoch nicht frei ist. Wir treffen uns um 10 Uhr im zweiten Stock.",
    "Das Programm liest die Datei zeilenweise ein und schreibt das Ergebnis "
    "in eine neue Tabelle. Fehlerhafte Zeilen werden übersprungen und am Ende "
    "gezählt.",
    "Hallo Anna, kannst du mir bis morgen die Zahlen für das dritte Quartal "
    "schicken? Ich brauche sie für die Präsentation beim Vorstand.",
    "Der Mieter ist verpflichtet, die Wohnung in einem ordnungsgemäßen Zustand "
    "zu halten und Schäden unverzüglich dem Vermieter zu melden.",
    "Zutaten: 200 g Mehl, zwei Eier, eine Prise Salz. Das Mehl mit den Eiern "
    "verrühren und den Teig eine halbe Stunde ruhen lassen.",
    "Protokoll der Sitzung vom 12. Mai: Es wurde beschlossen, dass die neue "
    "Software ab Juli eingesetzt wird. Die Schulung übernimmt die IT.",
    "Leider konnte die Lieferung nicht zugestellt werden, da niemand zu Hause "
    "war. Sie können das Paket in der nächsten Filiale abholen.",
    "Wir haben festgestellt, dass der Server in der Nacht mehrmals neu "
    "gestartet wurde. Die Ursache ist noch nicht bekannt.",
    "Die Installation ist einfach: Laden Sie das Archiv herunter, entpacken "
    "Sie es und führen Sie das Skript im Hauptverzeichnis aus.",
    "Ich wollte nur kurz Bescheid geben, dass ich heute etwas später komme, "
    "weil mein Zug ausgefallen ist.",
    "Gemäß § 5 des Vertrages kann jede Partei mit einer Frist von drei Monaten "
    "zum Ende des Kalenderjahres kündigen.",
    "Die Umsätze sind im Vergleich zum Vorjahr um zwölf Prozent gestiegen, "
    "vor allem wegen der starken Nachfrage im Ausland.",
    "Bitte beachten Sie, dass das Büro zwischen den Feiertagen geschlossen "
    "bleibt. In dringenden Fällen erreichen Sie uns per E-Mail.",
    "Fehler beim Öffnen der Datenbank: Die Datei ist gesperrt oder wird von "
    "einem anderen Prozess verwendet.",
    "Unsere Kunden erwarten, dass wir ihre Daten sicher aufbewahren und nur "
    "für den vereinbarten Zweck verwenden.",
    "Der Bericht fasst die Ergebnisse der Umfrage zusammen und gibt "
    "Empfehlungen für die weitere Planung.",
    "Nach dem Update startet die Anwendung nicht mehr. Im Log steht nur, dass "
    "eine Bibliothek nicht gefunden wurde.",
    "Liebe Kolleginnen und Kollegen, am Freitag findet unser Sommerfest statt. "
    "Für Essen und Getränke ist gesorgt.",
    "Die Messwerte wurden jede Stunde erfasst und anschließend mit den Daten "
    "der Wetterstation verglichen.",
    "Fuer die Abrechnung brauchen wir noch die Belege, die ueber das Portal "
    "eingereicht wurden, sonst koennen wir nichts erstatten.",
    "Es ist nicht klar, ob die Genehmigung rechtzeitig vorliegt. Wir sollten "
    "deshalb einen Plan für den Fall einer Verzögerung haben.",
    "Die Anlage besteht aus drei Teilen, die getrennt geliefert und vor Ort "
    "montiert werden.",
    "Kurz notiert: Termin mit dem Steuerberater am Dienstag, Unterlagen "
    "vorher kopieren und die Quittungen sortieren.",
]

ENGLISH = [
    "Dear customer, please find attached the invoice for March. The amount is "
    "due by April 15 and can be paid by bank transfer to the account below.",
    "The meeting has been moved to Thursday because the room is not available "
    "on Wednesday. We will meet at 10 am on the second floor.",
    "The program reads the file line by line and writes the result to a new "
    "table. Lines that cannot be parsed are skipped and counted at the end.",
    "Hi Anna, could you send me the numbers for the third quarter by "
    "tomorrow? I need them for the presentation to the board.",
    "The tenant shall keep the apartment in good condition and report any "
    "damage to the landlord without delay.",
    "Ingredients: 200 g flour, two eggs and a pinch of salt. Mix the flour "
    "with the eggs and let the dough rest for half an hour.",
    "Minutes of the meeting on May 12: it was decided that the new software "
    "will be used from July. IT will run the training.",
    "Unfortunately the parcel could not be delivered because nobody was at "
    "home. You can collect it from the nearest branch.",
    "We noticed that the server restarted several times during the night. "
    "The cause is not known yet.",
    "Installation is simple: download the archive, unpack it and run the "
    "script in the top-level directory.",
    "Just a quick note that I will be a little late today because my train "
    "was cancelled.",
    "Under section 5 of this agreement either party may terminate it with "
    "three months notice to the end of the calendar year.",
    "Revenue grew by twelve percent compared with the previous year, mainly "
    "because of strong demand from abroad.",
    "Please note that the office will be closed between the holidays. In "
    "urgent cases you can reach us by email.",
    "Error while opening the database: the file is locked or is being used "
    "by another process.",
    "Our customers expect us to keep their data safe and to use it only for "
    "the purpose they agreed to.",
    "The report summarizes the results of the survey and makes "
    "recommendations for further planning.",
    "After the update the application no longer starts. The log only says "
    "that a library could not be found.",
    "Dear colleagues, our summer party takes place on Friday. Food and drinks "
    "will be provided.",
    "The readings were taken every hour and then compared with the data from "
    "the weather station.",
    "It is not clear whether the permit will arrive in time, so we should "
    "have a plan for the case of a delay.",
    "The system consists of three parts which are delivered separately and "
    "assembled on site.",
    "Quick reminder: appointment with the accountant on Tuesday, copy the "
    "documents before and sort the receipts.",
    "This function returns the number of rows that were written, or raises "
    "an error if the connection was closed.",
]

NEITHER = [
    # Dutch: shares "er", "als", "in", "was", "die" with German.
    "Beste klant, in de bijlage vindt u de factuur voor maart. Het bedrag "
    "moet uiterlijk 15 april worden betaald op de onderstaande rekening.",
    "De vergadering is verplaatst naar donderdag omdat de zaal woensdag niet "
    "vrij is. Er zijn nog vragen over het budget.",
    # Danish: "der", "den", "er" are German words, "for" an English one.
    "Mødet er flyttet til torsdag, fordi lokalet ikke er ledigt om onsdagen. "
    "Der er stadig spørgsmål om budgettet for den nye afdeling.",
    # French and Spanish.
    "Madame, Monsieur, vous trouverez ci-joint la facture du mois de mars. "
    "Le montant est à régler avant le 15 avril sur le compte indiqué.",
    "La reunión se ha trasladado al jueves porque la sala no está libre el "
    "miércoles. Nos vemos a las diez en la segunda planta.",
    "Il programma legge il file riga per riga e scrive il risultato in una "
    "nuova tabella. Le righe non valide vengono saltate.",
    # Code, tables, numbers, identifiers.
    "def parse(path):\n    with open(path) as fh:\n        for line in fh:\n"
    "            yield line.split(',')\n",
    "id,name,score,active\n1,Alice,9.5,true\n2,Bob,7.0,false\n3,Cleo,8.25,true\n",
    "2026-01-01 00:00:00 INFO start\n2026-01-01 00:00:01 ERROR boom\n"
    "2026-01-01 00:00:02 WARNING careful\n",
    "SELECT id, name FROM users WHERE active = 1 ORDER BY name LIMIT 20;",
    '{"name": "svc", "port": 8080, "tags": ["a", "b"], "nested": {"x": 1}}',
    "4711 0815 1234 5678 9012 3456 7890 1111 2222 3333 4444 5555 6666",
    "aGVsbG8gd29ybGQgdGhpcyBpcyBiYXNlNjQgZW5jb2RlZCB0ZXh0IGFuZCBub3RoaW5nIGVsc2U=",
    # Too little to say: a greeting, a heading, one word.
    "Hallo",
    "Invoice",
    "OK",
]

GERMAN_HELD_OUT = [
    "Die Heizung im Erdgeschoss funktioniert seit gestern nicht mehr. Kann "
    "sich bitte jemand darum kümmern, bevor es am Wochenende kalt wird?",
    "Mit diesem Schreiben bestätigen wir den Eingang Ihrer Bewerbung. Wir "
    "melden uns, sobald die Auswahl abgeschlossen ist.",
    "Das Backup lief ohne Fehler durch, allerdings hat es fast doppelt so "
    "lange gedauert wie in der letzten Woche.",
    "Wenn die Temperatur über den Grenzwert steigt, schaltet sich das Gerät "
    "automatisch ab und sendet eine Meldung an die Zentrale.",
    "Ich habe den Entwurf durchgesehen und ein paar Kommentare eingefügt. Die "
    "meisten betreffen nur die Formulierung.",
    "Die Kosten für die Reparatur übernimmt die Versicherung, sofern der "
    "Schaden innerhalb von vierzehn Tagen gemeldet wird.",
    "Auf der Baustelle ist das Tragen eines Helms jederzeit Pflicht. Das gilt "
    "auch für Besucher und für die Mitarbeiter der Zulieferer.",
    "Unser Ziel ist es, die Wartezeiten im Kundenservice bis zum Jahresende "
    "auf unter fünf Minuten zu senken.",
    "Der Zugang zum Netzwerk wird nur nach einer Schulung freigeschaltet, "
    "damit alle die Regeln für Passwörter kennen.",
    "Vielen Dank für Ihre schnelle Antwort. Wir werden den Termin wie "
    "besprochen in den Kalender eintragen.",
    "Die Datei enthält alle Buchungen des letzten Jahres, sortiert nach Datum "
    "und nach Kostenstelle.",
    "Obwohl das Wetter schlecht war, sind mehr Besucher gekommen als im "
    "vergangenen Jahr.",
]

ENGLISH_HELD_OUT = [
    "The heating on the ground floor has not been working since yesterday. "
    "Could someone please take care of it before it gets cold at the weekend?",
    "With this letter we confirm that we have received your application. We "
    "will be in touch as soon as the selection is complete.",
    "The backup finished without errors, but it took almost twice as long as "
    "it did last week.",
    "If the temperature rises above the limit, the device switches itself off "
    "and sends a message to the control room.",
    "I have gone through the draft and added a few comments. Most of them are "
    "only about the wording.",
    "The insurance will cover the cost of the repair, provided that the "
    "damage is reported within fourteen days.",
    "Wearing a helmet on the building site is mandatory at all times. This "
    "also applies to visitors and to the staff of suppliers.",
    "Our goal is to bring waiting times in customer service down to under "
    "five minutes by the end of the year.",
    "Access to the network is only granted after a training session, so that "
    "everyone knows the rules for passwords.",
    "Thank you for your quick reply. We will put the appointment in the "
    "calendar as discussed.",
    "The file contains all of last year's bookings, sorted by date and by "
    "cost centre.",
    "Although the weather was bad, more visitors came than in the previous "
    "year.",
]

NEITHER_HELD_OUT = [
    "Het bestand bevat alle boekingen van vorig jaar, gesorteerd op datum en "
    "op kostenplaats. Er ontbreken nog een paar facturen.",
    "Filen indeholder alle bogføringer fra sidste år, sorteret efter dato. "
    "Der mangler stadig et par fakturaer for den sidste måned.",
    "Le fichier contient toutes les écritures de l'année dernière, triées par "
    "date et par centre de coûts.",
    "El archivo contiene todas las reservas del año pasado, ordenadas por "
    "fecha y por centro de coste.",
    "for (int i = 0; i < n; ++i) {\n    total += values[i];\n}\nreturn total;\n",
    "Datum;Betrag;Konto\n2026-01-03;120,50;4711\n2026-01-04;99,00;4712\n",
    "ERROR 2026-03-01T12:00:00Z pid=4411 code=E_CONN_RESET retry=3",
    "0x7f 0x45 0x4c 0x46 0x02 0x01 0x01 0x00 0x00 0x00 0x00 0x00",
    "Danke",
    "Thanks!",
]
