import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imapclient import IMAPClient
import pyzmail

# -------------------------
# LOAD DATASET
# -------------------------

df = pd.read_csv(
    "spam.csv",
    sep="\t",
    names=["label", "message"]
)

# Features and labels
X = df["message"]
y = df["label"]

# Convert text to numbers
vectorizer = CountVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# -------------------------
# GMAIL LOGIN
# -------------------------

EMAIL = "sandeepkella37@gmail.com"
PASSWORD = "fgpw gmyq neck mwiv"

imapObj = IMAPClient("imap.gmail.com", ssl=True)

imapObj.login(EMAIL, PASSWORD)

imapObj.select_folder("INBOX", readonly=True)

# Get all emails
UIDs = imapObj.search(["ALL"])

# Latest 5 emails
latest_emails = UIDs[-5:]

print("\nChecking latest emails...\n")

for uid in latest_emails:

    raw_message = imapObj.fetch([uid], ["BODY[]"])

    message = pyzmail.PyzMessage.factory(
        raw_message[uid][b"BODY[]"]
    )

    subject = message.get_subject()

    # Get body
    body = ""

    if message.text_part:
        try:
            body = message.text_part.get_payload().decode(
                message.text_part.charset
            )
        except:
            body = ""

    email_text = subject + " " + body

    # Transform email text
    email_vector = vectorizer.transform([email_text])

    # Predict
    prediction = model.predict(email_vector)

    print("Subject:", subject)
    print("Prediction:", prediction[0])
    print("-" * 50)

imapObj.logout()