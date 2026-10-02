"""Generate a synthetic customer-support dataset for the SFT fundamentals demo.

Running example used across the whole fine-tuning series: an Amazon-style AI
shopping / customer-support assistant. Each example maps a customer's problem
to the support response the assistant should learn to produce.

Deterministic (seeded RNG) so the dataset is reproducible, inspectable, and
needs no external download or API calls.
"""

import json
import random
from pathlib import Path

SEED = 42
N_EXAMPLES = 500
OUTPUT_PATH = Path(__file__).parent / "data" / "support_tickets.json"

PRODUCTS = [
    "wireless earbuds", "coffee maker", "running shoes", "laptop stand",
    "desk lamp", "yoga mat", "bluetooth speaker", "phone case",
    "winter jacket", "kitchen blender", "backpack", "office chair",
    "water bottle", "gaming mouse", "electric toothbrush", "air fryer",
    "board game", "children's book", "garden hose", "car phone mount",
]

NAMES = ["Alex", "Jordan", "Priya", "Sam", "Taylor", "Morgan", "Chris",
         "Jamie", "Riley", "Casey", "Drew", "Avery"]


def order_number():
    return f"#{random.randint(100000, 999999)}"


def days(lo=1, hi=14):
    return random.randint(lo, hi)


CATEGORIES = {}


def register(name):
    def deco(fn):
        CATEGORIES[name] = fn
        return fn
    return deco


@register("order_status")
def order_status():
    product = random.choice(PRODUCTS)
    order = order_number()
    d = days(1, 10)
    problem = (
        f"I ordered a {product} with order number {order} {d} days ago and "
        f"I still don't see any shipping update. Can you tell me what's going on?"
    )
    response = (
        f"Thanks for reaching out about order {order}. I checked and your {product} "
        f"is currently being prepared at our fulfillment center and is on track to ship "
        f"within 1-2 business days. You'll receive a tracking email the moment it ships, "
        f"and estimated delivery is {days(2, 5)} days after that. I'll flag the order so "
        f"it's monitored closely."
    )
    return problem, response


@register("returns_refunds")
def returns_refunds():
    product = random.choice(PRODUCTS)
    order = order_number()
    problem = (
        f"I want to return the {product} from order {order} because it's not what I "
        f"expected. How do I get a refund?"
    )
    response = (
        f"No problem at all — I've started a return for the {product} in order {order}. "
        f"You'll get a prepaid return label by email within the hour. Once the item is "
        f"scanned by the carrier, your refund of the full purchase price will be issued "
        f"to your original payment method within 3-5 business days."
    )
    return problem, response


@register("damaged_item")
def damaged_item():
    product = random.choice(PRODUCTS)
    order = order_number()
    problem = (
        f"The {product} I received in order {order} arrived damaged — it looks like it "
        f"was crushed in shipping. What can you do?"
    )
    response = (
        f"I'm sorry the {product} arrived damaged, that's not the experience we want for "
        f"you. I've issued a free replacement for order {order} at no extra cost, and you "
        f"don't need to return the damaged item. The replacement will ship within 1 "
        f"business day with expedited shipping."
    )
    return problem, response


@register("payment_issue")
def payment_issue():
    order = order_number()
    amount = random.choice([19.99, 29.99, 49.50, 89.00, 124.75])
    problem = (
        f"I was charged ${amount} twice for order {order}. Can you check my payment and "
        f"fix this?"
    )
    response = (
        f"I checked order {order} and confirmed a duplicate charge of ${amount}. I've "
        f"submitted a refund for the extra charge, which will appear on your statement "
        f"within 3-5 business days. Only one charge of ${amount} will remain for your order."
    )
    return problem, response


@register("account_access")
def account_access():
    name = random.choice(NAMES)
    problem = (
        f"Hi, this is {name}. I can't log into my account, it keeps saying my password "
        f"is incorrect even after I reset it. Can you help?"
    )
    response = (
        f"Hi {name}, sorry for the trouble logging in. I've sent a fresh password reset "
        f"link to your account email — it will be valid for 30 minutes. If it still "
        f"doesn't work after that, it may be a browser cache issue, so try an incognito "
        f"window or clearing cookies for our site."
    )
    return problem, response


@register("shipping_delay")
def shipping_delay():
    product = random.choice(PRODUCTS)
    order = order_number()
    d = days(5, 14)
    problem = (
        f"My {product} from order {order} was supposed to arrive {d} days ago and it "
        f"still hasn't shown up. This is really frustrating."
    )
    response = (
        f"I completely understand the frustration, and I'm sorry for the delay on your "
        f"{product} in order {order}. I checked with the carrier and the package is "
        f"delayed in transit but still moving — updated delivery is within 2 days. As an "
        f"apology for the delay, I've credited your account with a $10 voucher for your "
        f"next order."
    )
    return problem, response


@register("cancellation")
def cancellation():
    product = random.choice(PRODUCTS)
    order = order_number()
    problem = (
        f"I need to cancel my order {order} for the {product} — I accidentally ordered "
        f"two of them. Is it too late?"
    )
    response = (
        f"Good news — order {order} hasn't shipped yet, so I was able to cancel it for "
        f"you right away. You will not be charged for this order, and any pending "
        f"authorization will drop off within a few business days. The other order for "
        f"the {product} is unaffected."
    )
    return problem, response


@register("product_question")
def product_question():
    product = random.choice(PRODUCTS)
    problem = (
        f"Before I buy the {product}, can you tell me if it comes with a warranty and "
        f"what the return window is?"
    )
    response = (
        f"Great question! The {product} comes with a 1-year manufacturer warranty "
        f"covering defects, and you can return it within 30 days of delivery for any "
        f"reason as long as it's in like-new condition. If anything goes wrong after the "
        f"return window, our support team can still help with the manufacturer warranty."
    )
    return problem, response


@register("subscription_issue")
def subscription_issue():
    problem = (
        "I signed up for a monthly subscription but I was charged again this month even "
        "though I thought I canceled it last week. Can you check?"
    )
    response = (
        "I looked into your subscription and I can confirm the cancellation request from "
        "last week did not fully process due to a billing sync issue on our end. I've now "
        "canceled the subscription effective immediately and refunded this month's charge "
        "in full — it will reflect within 3-5 business days."
    )
    return problem, response


@register("promo_discount")
def promo_discount():
    code = random.choice(["SAVE10", "WELCOME15", "SUMMER20", "FIRST5"])
    problem = (
        f"I tried to use the promo code {code} at checkout but it says invalid. The "
        f"order total didn't get discounted. Can you help?"
    )
    response = (
        f"Thanks for flagging this. I checked the promo code {code} and applied the "
        f"discount manually to your most recent order, so you'll see the adjusted total "
        f"reflected shortly. In the future, that code works best when applied before "
        f"adding a gift wrap option, which was conflicting with it at checkout."
    )
    return problem, response


def main():
    random.seed(SEED)
    category_names = list(CATEGORIES.keys())
    per_category = N_EXAMPLES // len(category_names)
    remainder = N_EXAMPLES - per_category * len(category_names)

    examples = []
    for i, name in enumerate(category_names):
        count = per_category + (1 if i < remainder else 0)
        generator = CATEGORIES[name]
        for _ in range(count):
            problem, response = generator()
            examples.append({
                "category": name,
                "problem": problem,
                "response": response,
            })

    random.shuffle(examples)
    assert len(examples) == N_EXAMPLES, f"expected {N_EXAMPLES}, got {len(examples)}"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(examples, f, indent=2)

    print(f"Wrote {len(examples)} examples to {OUTPUT_PATH}")
    counts = {}
    for ex in examples:
        counts[ex["category"]] = counts.get(ex["category"], 0) + 1
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
