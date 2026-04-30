import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<br /><br />', ' ', text)   # remove HTML breaks
    text = re.sub(r'[^a-z\s]', '', text)        # keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()    # remove extra spaces
    return text