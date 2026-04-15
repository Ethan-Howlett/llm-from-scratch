"""
CS-3309R Final Project — Dataset Generator
Support-ticket intent classification (n ≈ 1 500, 7 classes).

Run:  python generate_data.py
Produces: support_tickets_train.csv, support_tickets_dev.csv, support_tickets_test.csv
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd

SEED = 3309
TRAIN_FRAC = 0.70
DEV_FRAC = 0.15

# Fraction of rows that get at least one deliberate spelling slip (common or synthetic).
SPELLING_ERROR_FRAC = 0.14

TEMPLATES: dict[str, list[str]] = {
    "password_reset": [
        "I forgot my password and cannot log in",
        "Password reset link is not working",
        "How do I reset my account password",
        "My password expired and I need a new one",
        "I changed my password but it still does not work",
        "The reset email never arrived in my inbox",
        "I keep getting invalid password error",
        "Can someone help me recover my account password",
        "I locked myself out after too many wrong passwords",
        "Please send me a password recovery link",
        "I cannot remember my login credentials",
        "My temporary password is not being accepted",
        "Where do I go to change my password",
        "I need to update my password for security reasons",
        "The system says my password does not meet requirements",
        "I tried resetting but the page gives an error",
        "How long does the password reset link last",
        "I want to set up a stronger password",
        "Password reset keeps timing out",
        "Can you manually reset my password please",
    ],
    "billing_refund": [
        "I was charged twice this month",
        "Please help me request a refund for an accidental charge",
        "Billing error I need a refund",
        "You charged me after I cancelled my subscription",
        "I see an unknown charge on my statement",
        "I was overcharged for my last order",
        "How do I dispute a charge on my account",
        "I need a refund for a service I did not use",
        "The wrong amount was billed to my card",
        "I received a charge but never placed an order",
        "Can I get a refund for a duplicate payment",
        "My account shows a pending charge I did not authorize",
        "I was charged full price but had a coupon",
        "Please reverse the incorrect billing on my account",
        "How long does it take to process a refund",
        "I cancelled within the refund window but was still charged",
        "The promotional price was not applied to my bill",
        "I need a receipt and refund for my last transaction",
        "Your system double billed me again",
        "I want to know why I was charged this amount",
    ],
    "shipping_delay": [
        "My order still has not arrived",
        "Shipping is delayed and tracking has not updated",
        "Where is my package",
        "It has been two weeks and my order is not here yet",
        "The tracking number shows no movement for days",
        "My delivery was supposed to arrive yesterday",
        "Can you check the status of my shipment",
        "I ordered express shipping but it is taking too long",
        "The estimated delivery date has passed",
        "My package seems to be stuck in transit",
        "I need an update on my order delivery",
        "The carrier says they cannot find my package",
        "My shipment has been in the same location for a week",
        "When will my order finally be delivered",
        "I paid for fast shipping and it is still not here",
        "Is there a way to expedite my delayed order",
        "The delivery status says exception what does that mean",
        "My package was marked delivered but I never received it",
        "I need to change my delivery address for a delayed order",
        "Can you resend my order since shipping is taking so long",
    ],
    "account_locked": [
        "My account was locked after failed attempts",
        "I am seeing an account locked message",
        "Please unlock my account",
        "I cannot access my account it says temporarily locked",
        "How do I unlock my account after too many tries",
        "My account got locked and I cannot do anything",
        "The system locked me out for no reason",
        "I need help getting back into my locked account",
        "Why is my account showing a security lock",
        "I got locked out after entering the wrong password",
        "My account has been locked for hours now",
        "Can an admin please unlock my account",
        "I verified my identity but my account is still locked",
        "The lock on my account has not been removed yet",
        "How long will my account stay locked",
        "I cannot log in because of an account security hold",
        "Please remove the lock from my profile",
        "My account was locked due to suspicious activity",
        "I need urgent access to my locked account",
        "The account lockout is preventing me from working",
    ],
    "cancel_subscription": [
        "I want to cancel my monthly plan",
        "Please stop renewing my subscription",
        "How do I cancel my membership",
        "I no longer need this service please cancel it",
        "Cancel my subscription immediately",
        "I want to stop my recurring payments",
        "How do I end my subscription before it renews",
        "Please cancel my account and stop all charges",
        "I tried cancelling but the button does not work",
        "Can you confirm my subscription has been cancelled",
        "I want to downgrade and eventually cancel my plan",
        "Stop charging my card I want to cancel",
        "I do not want to be auto renewed next month",
        "Where is the option to cancel my subscription",
        "I submitted a cancellation but I was still charged",
        "Please process my cancellation request right away",
        "I want to opt out of my current plan",
        "I need to cancel before the next billing cycle",
        "The cancel button is not showing on my account page",
        "Can I cancel and get a prorated refund",
    ],
    "update_payment": [
        "I need to update my card on file",
        "My payment failed how can I add a new card",
        "Please change my billing method",
        "I got a new credit card and need to update my account",
        "The card on my account has expired",
        "How do I switch to a different payment method",
        "I want to pay with a different card",
        "My bank issued a replacement card I need to update",
        "Where can I change my stored payment information",
        "I need to remove my old card and add a new one",
        "Can I update my payment to use PayPal instead",
        "My payment keeps failing because the card is old",
        "Please help me add a backup payment method",
        "I want to switch from credit card to debit card",
        "The system is not accepting my new card number",
        "How do I update the expiration date on my card",
        "I want to change the billing address for my card",
        "Can I set a different default payment method",
        "My card was compromised and I need to replace it",
        "The saved payment method needs to be updated urgently",
    ],
    "technical_bug": [
        "The app crashes when I open settings",
        "I found a bug on the checkout page",
        "Feature is not working as expected",
        "The page keeps loading and never finishes",
        "I get an error message when I try to save my profile",
        "The search function returns no results for anything",
        "Images are not displaying on the product page",
        "The mobile app freezes when I scroll through my feed",
        "I cannot upload files the upload button does nothing",
        "The notification system is sending duplicate alerts",
        "There is a display glitch on the dashboard",
        "The form resets when I click submit",
        "Videos do not play they just show a black screen",
        "The dropdown menu is not showing any options",
        "I get a 500 error when I try to access my orders",
        "The calendar widget is showing wrong dates",
        "My data is not syncing between desktop and mobile",
        "The export function generates a corrupted file",
        "Auto save is not working and I lost my changes",
        "The dark mode toggle breaks the layout",
    ],
}

# Optional openings / closings (support tickets: mixed formality).
PREFIXES = [
    "Hi, ",
    "Hello, ",
    "Hey ",
    "Urgent - ",
    "Quick question: ",
    "Good morning, ",
    "Hi there, ",
    "Sorry to bother you but ",
    "Need help - ",
]

SUFFIXES = [
    " Thanks",
    " thanks",
    " Please advise",
    " asap",
    " ASAP please",
    " Any help appreciated",
    " Thanks in advance",
    " Let me know",
    ".",
    "?",
]

LABELS = sorted(TEMPLATES.keys())
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

# Common real-world misspellings (correct -> wrong). Match is case-insensitive; replacement uses the wrong spelling as-is.
SPELLING_PAIRS: list[tuple[str, str]] = [
    ("password", "passwrod"),
    ("receive", "recieve"),
    ("received", "recieved"),
    ("separate", "seperate"),
    ("occurred", "occured"),
    ("definitely", "definately"),
    ("accommodate", "accomodate"),
    ("until", "untill"),
    ("successful", "succesful"),
    ("environment", "enviroment"),
    ("beginning", "begining"),
    ("maintenance", "maintainance"),
    ("calendar", "calender"),
    ("embarrassed", "embarassed"),
    ("recommend", "reccomend"),
    ("privilege", "priviledge"),
    ("tomorrow", "tommorrow"),
    ("apparent", "apparant"),
    ("consistent", "consistant"),
    ("occasion", "occassion"),
    ("parallel", "parralel"),
    ("weird", "wierd"),
    ("believe", "beleive"),
    ("achieve", "acheive"),
    ("friend", "freind"),
    ("because", "becuase"),
    ("before", "befor"),
    ("which", "wich"),
    ("their", "thier"),
    ("through", "throught"),
    ("immediately", "immediatly"),
    ("necessary", "neccessary"),
    ("occurrence", "occurence"),
    ("question", "quesiton"),
    ("subscription", "subscribtion"),
    ("business", "buisness"),
    ("financial", "finacial"),
    ("guarantee", "garantee"),
    ("account", "accoutn"),
    ("payment", "payement"),
    ("charged", "chaged"),
    ("delivery", "delievery"),
    ("shipping", "shiping"),
    ("information", "infomation"),
    ("button", "buton"),
    ("working", "workign"),
    ("wrong", "worng"),
]


def inject_spelling_error(text: str, rng: random.Random) -> str:
    """Insert one plausible typo: prefer dictionary pairs, else adjacent-letter swap in a word."""
    rng.shuffle(pairs := list(SPELLING_PAIRS))
    for correct, wrong in pairs:
        if re.search(rf"\b{re.escape(correct)}\b", text, flags=re.IGNORECASE):
            return re.sub(rf"\b{re.escape(correct)}\b", wrong, text, count=1, flags=re.IGNORECASE)

    words = text.split()
    long_idx = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not long_idx:
        return text
    i = rng.choice(long_idx)
    w = list(words[i])
    pos = rng.randint(0, len(w) - 2)
    w[pos], w[pos + 1] = w[pos + 1], w[pos]
    words[i] = "".join(w)
    return " ".join(words)


def vary_phrasing(text: str, rng: random.Random) -> str:
    """Layer informal tone, optional prefix/suffix, light punctuation — keeps same intent."""
    t = text

    if rng.random() < 0.28:
        t = rng.choice(PREFIXES) + t

    if rng.random() < 0.20:
        t = t + rng.choice(SUFFIXES)

    # Informal typing (no apostrophe), occasional — real tickets.
    if rng.random() < 0.12:
        t = t.replace(" cannot ", " cant ")
        t = t.replace(" do not ", " dont ")
        t = t.replace(" does not ", " doesnt ")
        t = t.replace(" I am ", " im ")
    if rng.random() < 0.06:
        t = t.replace(" you ", " u ")
    if rng.random() < 0.05:
        t = t.replace(" please ", " pls ")

    # Drop final period sometimes; add extra "!!" rarely.
    if rng.random() < 0.15 and t.endswith("."):
        t = t[:-1]
    if rng.random() < 0.04:
        t = t.rstrip(".!?") + "!!"

    # Occasional lowercase right after "Hi," / "Hey," — common in quick mobile tickets.
    if rng.random() < 0.06:
        m = re.match(r"^(.+?,\s+)([A-Z])(.*)$", t)
        if m:
            t = m.group(1) + m.group(2).lower() + m.group(3)

    return t.strip()


def generate(n_per_class: int = 220) -> pd.DataFrame:
    rng = random.Random(SEED)
    rows: list[dict] = []
    idx = 1
    for label, templates in TEMPLATES.items():
        for _ in range(n_per_class):
            query = rng.choice(templates)
            query = vary_phrasing(query, rng)
            if rng.random() < SPELLING_ERROR_FRAC:
                query = inject_spelling_error(query, rng)
            rows.append({"example_id": f"EX-{idx:04d}", "text": query, "label": label})
            idx += 1
    return pd.DataFrame(rows)


def main() -> None:
    base = Path(__file__).resolve().parent
    df = generate()
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    n_train = int(len(df) * TRAIN_FRAC)
    n_dev = int(len(df) * DEV_FRAC)

    train = df.iloc[:n_train]
    dev = df.iloc[n_train : n_train + n_dev]
    test = df.iloc[n_train + n_dev :]

    train.to_csv(base / "support_tickets_train.csv", index=False)
    dev.to_csv(base / "support_tickets_dev.csv", index=False)
    test.to_csv(base / "support_tickets_test.csv", index=False)

    (base / "label_map.txt").write_text(
        "\n".join(f"{i}\t{l}" for l, i in sorted(LABEL2ID.items(), key=lambda x: x[1])),
        encoding="utf-8",
    )

    print(f"train: {len(train)}  dev: {len(dev)}  test: {len(test)}")
    print(f"labels: {LABELS}")
    print(f"saved to {base}")


if __name__ == "__main__":
    main()
