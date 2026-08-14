# Barrier Option Pricing — Interactive Thesis Dashboard

Dashboard interattiva multi-pagina per la tesi *"Barrier Option Pricing
via Path Integral Methods"*. Contiene 9 dashboard (Script 1, 2, 3, 4.4,
5.5, 6.6, 7, 9.9, 10), ciascuna una pagina di un'unica app Dash.

## Struttura del progetto

```
thesis_dashboard/
├── app.py                      ← entry point principale
├── pricing_bs.py                ← formule Black-Scholes/Reiner-Rubinstein
├── pricing_floating.py          ← MC Naive/BB/NPI, barriera flottante
├── pricing_heston.py            ← Heston MC + NPI 3D
├── pricing_adaptive.py          ← NPI adattivo/uniforme
├── requirements.txt
├── render.yaml                  ← configurazione per il deploy su Render
├── check_structure.py           ← verifica la struttura prima di caricare
├── .gitignore
└── pages/
    ├── 00_home.py                ← indice / homepage
    ├── 01_bs_barrier.py
    ├── 02_mc_naive.py
    ├── 03_brownian_bridge.py
    ├── 04_mc_naive_floating.py
    ├── 05_mc_bb_floating.py
    ├── 06_npi_floating.py
    ├── 07_comparison.py
    ├── 08_adaptive_npi.py
    └── 09_heston.py
```

**Importante:** tutti i file `pricing_*.py` e `app.py` vanno nella
cartella principale, allo stesso livello. Tutte le pagine numerate
vanno dentro la sottocartella `pages/` (nome esatto, richiesto da Dash).

Prima di caricare su GitHub, esegui sempre:

```bash
python check_structure.py
```

Controlla che ogni file sia al posto giusto e segnala eventuali nomi
con spazi invece di underscore (errore comune quando si rinominano i
file scaricati).

## 1. Provare in locale (facoltativo, ma consigliato)

```bash
pip install -r requirements.txt
python app.py
```

Poi apri http://127.0.0.1:8000 nel browser. Controlla che tutte e 10
le pagine si carichino e che i pulsanti "Run" funzionino, prima di
passare al deploy online.

## 2. Caricare su GitHub

Se non hai già un account GitHub, creane uno gratuito su
https://github.com/signup.

**Interfaccia web (senza terminale, più semplice):**

1. Vai su https://github.com/new
2. Dai un nome al repository (es. `thesis-barrier-dashboard`), lascialo
   **Public** (Render free richiede repo pubblico o l'autorizzazione
   esplicita a quello privato)
3. Clicca "Create repository"
4. Nella pagina del repo appena creato, clicca "uploading an existing
   file"
5. Trascina dentro **tutti i file della cartella principale**
   (`app.py`, i 4 `pricing_*.py`, `requirements.txt`, `render.yaml`,
   `.gitignore`, `check_structure.py`) — NON la cartella `pages/`
   ancora, GitHub la creerà da sé quando trascini i file di pagina con
   il path corretto
6. Per i file dentro `pages/`: puoi trascinarli nello stesso passaggio
   di upload — se il tuo file manager permette di trascinare
   l'intera cartella `pages/`, GitHub la ricrea automaticamente con lo
   stesso percorso interno
7. Scrivi un messaggio di commit (es. "Initial upload") e clicca
   "Commit changes"

**Da terminale (alternativa, se preferisci):**

```bash
cd thesis_dashboard
git init
git add .
git commit -m "Initial upload"
git branch -M main
git remote add origin https://github.com/TUO-USERNAME/thesis-barrier-dashboard.git
git push -u origin main
```

## 3. Collegare a Render

1. Vai su https://render.com e registrati (puoi accedere direttamente
   con il tuo account GitHub)
2. Nella dashboard di Render, clicca "New +" → "Web Service"
3. Seleziona "Build and deploy from a Git repository", poi collega il
   repository GitHub appena creato
4. Render dovrebbe rilevare automaticamente `render.yaml` e
   precompilare le impostazioni (build command, start command). Se non
   lo fa automaticamente, inserisci manualmente:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:server --bind 0.0.0.0:$PORT --timeout 120 --workers 1`
5. Seleziona il piano **Free**
6. Clicca "Create Web Service"

Il primo deploy richiede qualche minuto (installazione dipendenze,
compilazione Numba). Al termine, Render ti dà un URL pubblico del tipo:

```
https://thesis-barrier-dashboard.onrender.com
```

**Nota sul piano gratuito:** Render free "addormenta" il servizio dopo
un periodo di inattività; il primo caricamento dopo un periodo di
pausa può richiedere 30-60 secondi extra. Le richieste successive sono
normali.

## 4. Generare il QR code

Una volta ottenuto l'URL pubblico, genera il QR code con un servizio
gratuito, ad esempio https://www.qr-code-generator.com — incolla l'URL
di Render e scarica l'immagine PNG/SVG da inserire nella tesi.

## Note tecniche sulle singole pagine

- **Pagine 6 e 7** (NPI Floating, Method Comparison) usano una
  versione **vettorizzata** del metodo NPI, verificata numericamente
  identica alla versione originale a doppio ciclo Python ma 2-5 volte
  più veloce — necessaria per restare entro tempi ragionevoli su CPU
  condivisa gratuita.
- **Pagina 9** (Heston) include tutte le correzioni discusse: fix
  della scala del running-max, DI Call aggiunto, warning di stabilità
  numerica σᵥ/dv, pannello di sensitivity rispetto a θ, curva NPI
  ridotta (5 punti) nel grafico α.
- Tutte le formule Black-Scholes duplicate nelle dashboard originali
  sono state consolidate in `pricing_bs.py`, verificate numericamente
  equivalenti prima della sostituzione.
