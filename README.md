# Tesi_magistrale

## Aggiungere modifiche locali su git:  
git status  
git add .  
git commit -m "Messaggio"  
git push origin main  

## Per rimuovere file solo da git  
 git rm --cached nomefile.py   
git commit -m "Rimosso nomefile.py dal tracking"  
git push origin main  

## Spiegazione moduli
- **`main.py`**: è lo script principale che esegue l'intero workflow:
  - caricamento e preprocessing dei dataset,  
  - anonimizzazione tramite modulo dedicato,  
  - addestramento e valutazione dei modelli,  
  - analisi di interpretabilità (SHAP e LIME) attraverso modulo dedicato,  
  - salvataggio dei risultati finali. 
- **`anonymize.py`**: implementa l’algoritmo di **l-diversity** utilizzato per l’anonimizzazione dei dataset.  
- **`explainability.py`** implementa le funzioni per l’analisi di interpretabilità dei modelli globale **SHAP** e locale  **LIME**.
- **`graphs.py`**  modulo di supporto utile per la generazione e salvataggio di grafici relativi a metriche e risultati.
- **`stats_dataset.py`** permette la generazione di tabelle, figure e immagini relative all'analisi statistica e descrittiva dei dataset


## Guida all'esecuzione

Per eseguire il progetto è sufficiente seguire questi passaggi:

1. **Eseguire `main.py`**  
   - Carica i dataset e applica l’anonimizzazione tramite il modulo `anonymize.py`.  
   - Addestra e valuta i modelli.
   - Crea e salva i valori di **SHAP** per l'interpretabilità globale grazie al modulo `explainability.py`
   - Crea e salva i grafici per l'analisi di interpretabilità locale **LIME**
   - Salva i dataset anonimizzati (`dataset_anon/`) e i risultati delle metriche (`risultati_classificatori.csv` e `.xlsx`).  

2. **Eseguire `graphs.py`**  
   - Genera i grafici riassuntivi e descrittivi dei risultati ottenuti.  

3. **Eseguire `explainability.py`**   
   - Genera i grafici sull’evoluzione del ranking delle feature al variare del livello di anonimizzazione basandosi sui valori di SHAP trovati in precedenza.  

### Opzionale

4. **Eseguire `stats_dataset.py`**
  - Genera i grafici descrittivi e riassuntivi delle statistiche dei dataset con lo scopo di conososcere meglio le caratteristsiche struttturali.
   


### All'interno della cartella [EXPLANATIONS](https://github.com/lucacena/Progetto_tesi/tree/main/explanations) è presente lo studio della variazione dell'importanza delle features prima e dopo l'anonimizzazione utilizzando due metodologie: SHAP e LIME.
#### Per ogni dataset, è presente una directory sia per il dataset orignale che anonimizzzato suddiviso ulteriormente per i 4 modelli studiati.
- *shap_beeswarm*: impatto delle features sulle decisioni prese dal modello
- *shap_importance*: salvataggio dell'importanza delle features in decisioni del modello 
- *rank_change*: come cambia il rank delle features passando da un dataset NON anonimizzato a uno anonimizzato, visualizzato come bar_plot
- *rank_evolution*: evoluzione importanza tutte features al crescere di k e l
