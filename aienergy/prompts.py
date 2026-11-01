import re

_POLICY = ("Late submissions lose 5% per day unless an extension is approved by the course "
           "coordinator. Extensions require documented evidence such as a medical certificate. "
           "Students should apply through the portal before the deadline where possible. ")

PROMPTS = {
    "short": "Explain in two sentences what a library fine is.",
    "medium": ("Write a clear 150-word explanation of how a student can request an extension "
               "on a deadline, including who to contact and what evidence to provide."),
    "long_context": ("You are a campus assistant. Use only the policy text below.\n\n"
                     + _POLICY * 40 + "\n\nQuestion: A student was ill for three days. "
                     "What should they do and what penalty applies?"),
}


def parse_params_b(model: str):
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", model.lower().replace(":", " ").replace("-", " "))
    return float(m.group(1)) if m else None
