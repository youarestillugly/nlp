# DAT401 IMDB Classical NLP Project

## Files

- `DAT401_IMDB_Final_NLP_Project.ipynb` — complete final notebook
- `app.py` — Streamlit deployment application
- `requirements.txt` — Python dependencies
- `model/imdb_sentiment_final_pipeline.joblib` — generated after running the notebook

## Run the notebook

Run the notebook from this project directory. The notebook downloads the official IMDB dataset if it is not already present.

The final model is selected using validation/CV results and is evaluated on the held-out test set only after selection.

## Run the Streamlit application

After running the notebook and creating the joblib model:

```bash
streamlit run app.py
```

The application accepts raw movie review text and uses the same cleaning, feature extraction, and classifier pipeline used during training.

## Important

Do not manually insert example final accuracy/F1 values into the report. Use the actual values produced by your final notebook run.
