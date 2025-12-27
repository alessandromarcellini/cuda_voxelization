un processo che legge i file e carica i punti nella socket (quando finisce un frame elemento speciale)
un processo che legge dalla socket, ed elabora i punti (lancia kernel)


PROTOCOLLO:
per ogni frame:
- invio intero numero punti
- invio un punto per volta
- passo al prossimo frame (file)