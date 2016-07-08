# Naive-Bayes

A Clojure(Script) naive Bayes classifier.

![status badge](https://circleci.com/gh/dhessing/naive-bayes.svg?style=shield&circle-token=50f12c15daf0b921511c064a2a8dcd0d23ba194f)

## Usage
```clojure
(require '[naive-bayes.core :as nb])
(let [data (nb/bag
             [{:class ["en"] :text (nb/string->words "English is a West Germanic language that was first spoken in early medieval England and is now a global lingua franca.[4][5] English is either the official language or an official language in almost 60 sovereign states. It is the most commonly spoken language in the United Kingdom, the United States, Canada, Australia, Ireland, and New Zealand, and is widely spoken in some areas of the Caribbean, Africa, and South Asia.[6] It is the third most common native language in the world, after Mandarin and Spanish.[7] It is the most widely learned second language and an official language of the United Nations, of the European Union, and of many other world and regional international organisations.\n\nEnglish has developed over the course of more than 1,400 years. The earliest forms of English, a set of Anglo-Frisian dialects brought to Great Britain by Anglo-Saxon settlers in the fifth century, are called Old English. Middle English began in the late 11th century with the Norman conquest of England.[8] Early Modern English began in the late 15th century with the introduction of the printing press to London and the King James Bible, and the start of the Great Vowel Shift.[9] Through the worldwide influence of the British Empire, modern English spread around the world from the 17th to mid-20th centuries. Through all types of printed and electronic media, as well as the emergence of the United States as a global superpower, English has become the leading language of international discourse and the lingua franca in many regions and in professional contexts such as science, navigation, and law.[10]\n\nModern English has little inflection compared with many other languages, and relies more on auxiliary verbs and word order for the expression of complex tenses, aspect and mood, as well as passive constructions, interrogatives and some negation. Despite noticeable variation among the accents and dialects of English used in different countries and regions – in terms of phonetics and phonology, and sometimes also vocabulary, grammar and spelling – English-speakers from around the world are able to communicate with one another with surprising ease.")}
              {:class ["en"] :text (nb/string->words "Machine learning is a subfield of computer science[1] (more particularly soft computing) that evolved from the study of pattern recognition and computational learning theory in artificial intelligence.[1] In 1959, Arthur Samuel defined machine learning as a \"Field of study that gives computers the ability to learn without being explicitly programmed\".[2] Machine learning explores the study and construction of algorithms that can learn from and make predictions on data.[3] Such algorithms operate by building a model from an example training set of input observations in order to make data-driven predictions or decisions expressed as outputs,[4]:2 rather than following strictly static program instructions.\n\nMachine learning is closely related to (and often overlaps with) computational statistics; a discipline which also focuses in prediction-making through the use of computers. It has strong ties to mathematical optimization, which delivers methods, theory and application domains to the field. Machine learning is employed in a range of computing tasks where designing and programming explicit algorithms is unfeasible. Example applications include spam filtering, optical character recognition (OCR),[5] search engines and computer vision. Machine learning is sometimes conflated with data mining,[6] where the latter sub-field focuses more on exploratory data analysis and is known as unsupervised learning.[4]:vii[7]\n\n")}
              {:class ["de"] :text (nb/string->words "Die englische Sprache (Eigenbezeichnung: English [ˈɪŋɡlɪʃ]) ist eine ursprünglich in England beheimatete germanische Sprache, die zum westgermanischen Zweig gehört. Sie entwickelte sich ab dem frühen Mittelalter durch Einwanderung nordseegermanischer Völker nach Britannien, darunter der Angeln – von denen der Name „Englisch“ sich herleitet – sowie der Sachsen. Die Frühformen der Sprache werden daher auch manchmal Angelsächsisch genannt.\n\nDie am nächsten verwandten lebenden Sprachen sind die friesischen Sprachen und das Niederdeutsche auf dem Festland, zu dem anfangs lange ein Dialektkontinuum bestand. Im Verlauf seiner Geschichte hat das Englische dann allerdings starke Sonderentwicklungen ausgebildet: Im Satzbau wechselte das Englische, im Gegensatz zu allen westgermanischen Verwandten auf dem Kontinent, in ein Subjekt-Verb-Objekt-Schema über und verlor die Verbzweiteigenschaft. Die Bildung von Wortformen (Flexion) bei Substantiven, Artikeln, Verben und Adjektiven wurde stark abgebaut. Im Wortschatz wurde das Englische in einer frühen Phase zunächst vom Sprachkontakt mit nordgermanischen Sprachen stark beeinflusst, der sich durch die zeitweilige Besetzung durch Dänen und Norweger im 9. Jahrhundert ergab. Später ergab sich nochmals eine starke Prägung durch den Kontakt mit dem Französischen aufgrund der normannischen Eroberung Englands 1066. Aufgrund der vielfältigen Einflüsse aus westgermanischen und nordgermanischen Sprachen, dem Französischen sowie den klassischen Sprachen besitzt das heutige Englisch einen außergewöhnlich umfangreichen Wortschatz.\n\nDie englische Sprache wird mit dem lateinischen Alphabet geschrieben. Eine wesentliche Fixierung der Rechtschreibung erfolgte mit Aufkommen des Buchdrucks im 15./16. Jahrhundert, trotz gleichzeitig fortlaufenden Lautwandels.[3] Die heutige Schreibung des Englischen stellt daher eine stark historische Orthographie dar, die von der Abbildung der tatsächlichen Lautgestalt vielfältig abweicht.\n\nAusgehend von seinem Entstehungsort England breitete sich das Englische über die gesamten Britischen Inseln aus und verdrängte allmählich die zuvor dort gesprochenen (v. a. keltischen) Sprachen. In seiner weiteren Geschichte ist das Englische vor allem infolge der Besiedlung Amerikas sowie der Kolonialpolitik Großbritanniens in Australien, Afrika und Indien zu einer Weltsprache geworden, die heute (global) weiter verbreitet ist als jede andere Sprache (die Sprache mit der größten Zahl an Muttersprachlern ist jedoch Mandarin-Chinesisch). Englischsprachige Länder und Gebiete bzw. ihre Bewohner werden auch anglophon (reformiert auch anglofon geschrieben) genannt.\n\nDas Englische wird in den Schulen vieler Länder als erste Fremdsprache gelehrt und ist offizielle Sprache der meisten internationalen Organisationen, wobei viele davon daneben noch andere offizielle Sprachen nutzen. In Westdeutschland verständigten sich die Länder 1955 im Düsseldorfer Abkommen darauf, an den Schulen Englisch generell als Pflichtfremdsprache einzuführen.")}
              {:class ["de"] :text (nb/string->words "Maschinelles Lernen ist ein Oberbegriff für die „künstliche“ Generierung von Wissen aus Erfahrung: Ein künstliches System lernt aus Beispielen und kann diese nach Beendigung der Lernphase verallgemeinern. Das heißt, es werden nicht einfach die Beispiele auswendig gelernt, sondern es „erkennt“ Muster und Gesetzmäßigkeiten in den Lerndaten. So kann das System auch unbekannte Daten beurteilen (Lerntransfer) oder aber am Lernen unbekannter Daten scheitern (Überanpassung).\n\nAus dem weiten Spektrum möglicher Anwendungen seien hier genannt automatisierte Diagnose\u00ADverfahren, Erkennung von Kreditkartenbetrug, Aktienmarkt\u00ADanalysen, Klassifikation von Nukleotidsequenzen, Sprach- und Texterkennung und autonome Systeme.\n\nDas Thema ist eng verwandt mit „Knowledge Discovery in Databases“ und „Data-Mining“, bei dem es jedoch vorwiegend um das Finden von neuen Mustern und Gesetzmäßigkeiten geht. Viele Algorithmen können für beide Ziele verwendet werden, und insbesondere kann „Knowledge Discovery in Databases“ verwendet werden, um Lerndaten für „maschinelles Lernen“ zu produzieren oder vorzuverarbeiten, und Algorithmen aus dem maschinellen Lernen finden beim Data-Mining Anwendung.")}
              {:class ["nl"] :text (nb/string->words "Het Engels (English) is een Indo-Europese taal, die vanwege de nauwe verwantschap met talen als het Fries, (Neder-)Duits en Nederlands tot de West-Germaanse talen wordt gerekend. De taal is ontstaan in Engeland in de tijd van de Angelsaksen en is nu de lingua franca in grote delen van de wereld, als resultaat van de militaire, economische, culturele, wetenschappelijke en politieke invloed van het Britse Rijk gedurende de 18e, 19e en begin 20e eeuw[2] en de invloed van de Verenigde Staten vanaf het begin van de 20e eeuw tot op heden. Het Engels wordt tegenwoordig op grote schaal gebruikt als tweede taal of officiële taal in de Gemenebestlanden en is de voertaal van vele internationale organisaties. Zo is het een van de zes officiële talen van de Verenigde Naties.\n\nDe gezamenlijke Engelstalige landen en hun gewoonten worden soms aangeduid als Angelsaksisch, om verwarring te voorkomen met de andere betekenis van 'Engels', nl. 'uit Engeland'.")}
              {:class ["nl"] :text (nb/string->words "Automatisch leren of Machinaal leren is een breed onderzoeksveld binnen kunstmatige intelligentie, dat zich bezighoudt met de ontwikkeling van algoritmes en technieken waarmee computers kunnen leren.\n\nDe methodes zijn te verdelen in twee ruwe categorieën: aanleidinggevend en deductief. Aanleidinggevende methodes creëren computerprogramma's door het vormen van regels of het extraheren van patronen uit data. Deductieve methoden hebben als resultaat een functie die net zo generiek is als de invoerdata.\n\nAutomatisch leren is sterk gerelateerd aan statistiek, aangezien beide velden de studie van data analyseren. Automatisch leren is meer gericht op de algoritmische complexiteit of de implementatie in programma's. Het is ook gerelateerd aan data mining, waarin op een geautomatiseerde manier patronen en relaties worden gezocht in grote hoeveelheden gegevens.\n\nVeel leerproblemen zijn NP-hard of moeilijker, dus een belangrijk onderdeel van dit vakgebied is algoritmes te ontwikkelen die de oplossing benaderen.")}])]
  (nb/classify-text data :text "What language is this?"))
=> [:class "en"]
```
## License

Copyright © 2016 FIXME

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
