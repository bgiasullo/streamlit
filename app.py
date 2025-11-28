import streamlit as st
import pandas as pd
import numpy as np
import csv
import unicodedata
import random
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Debug function for suspicious bytes (optional)
# -----------------------------
def debug_encoding(label, text):
    if not isinstance(text, str):
        return
    raw = text.encode("utf-8", errors="replace")
    suspicious = [b"\xc2", b"\xa0", b"\xef\xbb\xbf"]
    if any(s in raw for s in suspicious):
        print(f"ENCODING WARNING [{label}] repr={repr(text)} raw={raw}")

# -----------------------------
# Clean text function (preserve real newlines)
# -----------------------------
def clean_text(s):
    # Flatten lists, tuples, numpy arrays
    if isinstance(s, (list, tuple, np.ndarray)):
        s = " ".join(str(x) for x in s)

    # Flatten dicts
    if isinstance(s, dict):
        s = " ".join(f"{k}:{v}" for k, v in s.items())

    # Force string type
    s = "" if pd.isna(s) else str(s)

    # Replace literal backslash-n with real newline
    s = s.replace("\\n", "\n")

    # Normalize Unicode
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\ufeff", "")
    s = s.replace("\u00A0", " ")

    debug_encoding("after_clean", s)
    return s

# -----------------------------
# Replacement rules
# -----------------------------
replacements = {
    '[{"task":"T4","value":"': "",
    '","taskType":"textFromSubject"},{"task":"T1","task_type":"dropdown-simple","value":{"select_label":"Main Dropdown","option":true,"value":1,"label":"Page is blank"}}]': "",
    '","taskType":"textFromSubject"},{"task":"T1","task_type":"dropdown-simple","value":{"select_label":"Main Dropdown","option":true,"value":0,"label":"Corrections made"}}]': "",
    '","taskType":"textFromSubject"},{"task":"T1","task_type":"dropdown-simple","value":{"select_label":"Main Dropdown","option":true,"value":2,"label":"No corrections needed"}}]': "",
    '","taskType":"textFromSubject"},{"task":"T1","task_type":"dropdown-simple","value":{"select_label":"Main Dropdown","option":true,"value":3,"label":"Text is illegible"}}]': "",
    r"\u0026": "&",
    r"\u003e": ">",
    r"\u003c": "<",
    "♂": "[male]",
    "♀": "[female]",
    "⚥": "[intersex]",
    '\"': '"'
}

# -----------------------------
# Clean raw CSV text
# -----------------------------
def clean_raw_csv(raw_csv_text):
    input_io = StringIO(raw_csv_text)
    output_io = StringIO()
    reader = csv.reader(input_io)
    writer = csv.writer(output_io)

    def replace_all(text):
        debug_encoding("raw_cell", text)
        for find, replace in replacements.items():
            text = text.replace(find, replace)
        return text

    for row in reader:
        cleaned_row = [replace_all(cell) for cell in row]
        writer.writerow(cleaned_row)

    return output_io.getvalue()

# -----------------------------
# Select most similar annotation per subject
# -----------------------------
def select_most_similar_annotation(df):
    results = []
    df = df.copy()
    df["annotations"] = df["annotations"].apply(clean_text)

    # Ensure all annotations are strings and non-empty
    df["annotations"] = df["annotations"].astype(str)
    df = df[df["annotations"].str.strip() != ""]

    for subject_id, group in df.groupby("subject_ids"):
        annotations = group["annotations"].tolist()
        if len(annotations) == 1:
            selected = annotations[0]
        else:
            try:
                vectorizer = TfidfVectorizer(
                    strip_accents=None,
                    analyzer="word",
                    token_pattern=r"(?u)\b\w+\b"
                )
                tfidf_matrix = vectorizer.fit_transform(annotations)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                np.fill_diagonal(similarity_matrix, -1)

                max_sim = similarity_matrix.max()
                if not np.isscalar(max_sim):
                    max_sim = np.max(max_sim)

                if max_sim <= -1:
                    selected = random.choice(annotations)
                else:
                    pairs = np.argwhere(similarity_matrix == max_sim)
                    i, j = random.choice(pairs)
                    selected = random.choice([annotations[i], annotations[j]])
            except Exception:
                # Fallback if TF-IDF fails
                selected = random.choice(annotations)

        results.append({"subject_ids": subject_id, "annotations": selected})

    return pd.DataFrame(results)

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Zooniverse Annotation Cleaner + Excel-Safe CSV Output")

uploaded_file = st.file_uploader("Upload your raw Zooniverse CSV", type=["csv"])

if uploaded_file is not None:
    try:
        raw_csv_text = uploaded_file.read().decode("utf-8", errors="replace")
        cleaned_csv_text = clean_raw_csv(raw_csv_text)

        df = pd.read_csv(StringIO(cleaned_csv_text), dtype=str, encoding="utf-8-sig")
        result_df = select_most_similar_annotation(df)

        st.success("Processing complete! Unicode and newlines are preserved, Excel-safe CSV ready.")

        # -----------------------------
        # Excel-safe CSV with quotes to preserve multiline cells
        # -----------------------------
        output_buffer = StringIO()
        result_df.to_csv(
            output_buffer,
            index=False,
            sep=",",
            encoding="utf-8-sig",
            quoting=csv.QUOTE_ALL,  # Wrap all cells in quotes
            quotechar='"'
        )
        output_bytes = output_buffer.getvalue().encode("utf-8-sig")

        st.download_button(
            label="Download Final CSV (Excel-Safe UTF-8)",
            data=output_bytes,
            file_name="final_text.csv",
            mime="text/csv",
        )

        st.dataframe(result_df)

    except Exception as e:
        st.error(f"Error processing file: {e}")
