"""Plain-language feature names and the driver -> fix rules.

Shared by the pipeline (for reports) and the Streamlit app. No heavy imports here.
Fixes are ordered non-toxic first; poison is only ever the last resort.
"""
from __future__ import annotations

FEATURE_INFO: dict[str, tuple[str, str]] = {
    # feature: (plain label, driver category)
    "rodent": ("Rat requests this month", "colony"),
    "rodent_3": ("Rat requests, last 3 months", "colony"),
    "rodent_12": ("Rat requests, last 12 months", "colony"),
    "rodent_months_12": ("Months with rat requests (last 12)", "colony"),
    "months_since_rodent": ("Months since last rat request", "colony"),
    "food": ("Sanitation requests this month", "food"),
    "food_3": ("Sanitation requests, last 3 months", "food"),
    "food_12": ("Sanitation requests, last 12 months", "food"),
    "food_trend": ("Sanitation requests rising vs. usual", "food"),
    "shelter_3": ("Shelter requests (vacant lots, abandoned cars), 3 mo", "shelter"),
    "shelter_12": ("Shelter requests (vacant lots, abandoned cars), 12 mo", "shelter"),
    "containers_3": ("Rat-resistant can requests, last 3 months", "containers"),
    "containers_12": ("Rat-resistant can requests, last 12 months", "containers"),
    "vacant_n": ("Vacant or blighted buildings", "shelter"),
    "venues_n": ("Licensed bars and restaurants", "food_business"),
    "restaurants_n": ("Licensed restaurants", "food_business"),
    "grocery_n": ("Grocery stores", "food_business"),
    "pop_density": ("Population density", "density"),
    "log_pop": ("Population", "density"),
    "temp_c": ("Monthly temperature", "season"),
    "precip_mm": ("Monthly rainfall", "season"),
    "month_sin": ("Time of year", "season"),
    "month_cos": ("Time of year", "season"),
}

DRIVERS: dict[str, dict] = {
    "food": {
        "title": "Food source: trash and sanitation problems",
        "fixes": [
            ("Swap open or broken cans for rat-resistant lidded containers", "DPW (Rat Replacement Containers, Supercans)", "Non-toxic"),
            ("Alley wash-down and litter pickup on a fixed schedule", "BID / DPW", "Non-toxic"),
            ("Sanitation enforcement visit for overflowing dumpsters", "DPW Solid Waste Enforcement", "Non-toxic"),
        ],
    },
    "food_business": {
        "title": "Food businesses concentrated here",
        "fixes": [
            ("Dumpster-lid and grease-bin check with each restaurant", "Restaurants, with BID support", "Non-toxic"),
            ("Shared compacting trash point for the block", "BID (DSLBD compactor grants)", "Non-toxic"),
        ],
    },
    "shelter": {
        "title": "Shelter: vacant property, abandoned vehicles or overgrowth",
        "fixes": [
            ("Secure vacant-building openings and clear debris", "Property owner / Department of Buildings", "Non-toxic"),
            ("Tow abandoned vehicles; mow and clear overgrowth", "DPW / property owner", "Non-toxic"),
        ],
    },
    "density": {
        "title": "Dense housing with shared trash storage",
        "fixes": [
            ("Building trash-room and can-count check", "Property managers", "Non-toxic"),
        ],
    },
    "season": {
        "title": "Seasonal pressure",
        "fixes": [
            ("Pre-season clean-up push before peak months", "BID / ANC / residents", "Non-toxic"),
        ],
    },
    "containers": {
        "title": "Rat-resistant cans already requested",
        "fixes": [
            ("Confirm the cans were delivered and are being used", "DPW", "Non-toxic"),
        ],
    },
    "colony": {
        "title": "Established colony (rat requests keep coming back)",
        "fixes": [
            ("Seal gaps under fences and foundations with wire mesh", "Property owners", "Non-toxic"),
            ("Burrow treatment using dry ice or carbon monoxide first", "DC Health Rodent Control", "Low toxicity"),
            ("Rodenticide only if the above fail, in sealed stations", "DC Health Rodent Control", "Last resort"),
        ],
    },
}

TOX_ORDER = {"Non-toxic": 0, "Low toxicity": 1, "Last resort": 2}


def label(feature: str) -> str:
    return FEATURE_INFO.get(feature, (feature, "other"))[0]


def category(feature: str) -> str:
    return FEATURE_INFO.get(feature, (feature, "other"))[1]


def plan_for(drivers: list[dict], max_categories: int = 3) -> list[dict]:
    """Turn the top risk-raising drivers into an ordered fix list (non-toxic first)."""
    cats: list[str] = []
    for d in drivers:
        if d.get("contribution", 0) <= 0:
            continue
        c = category(d["feature"])
        if c in DRIVERS and c not in cats:
            cats.append(c)
        if len(cats) >= max_categories:
            break
    if "colony" not in cats:
        cats.append("colony")  # treatment stays available, but always after source fixes
    plan = []
    for c in cats:
        for action, owner, tox in DRIVERS[c]["fixes"]:
            plan.append({"driver": DRIVERS[c]["title"], "action": action, "owner": owner, "toxicity": tox})
    plan.sort(key=lambda r: TOX_ORDER[r["toxicity"]])
    return plan


def describe_driver(d: dict) -> str:
    v = d.get("value")
    lab = label(d["feature"])
    if d["feature"] in ("month_sin", "month_cos", "temp_c", "precip_mm"):
        return lab
    if d["feature"] == "pop_density" and v is not None:
        return f"{lab}: {v:,.0f} people per km²"
    if v is None:
        return lab
    return f"{lab}: {v:,.0f}" if float(v).is_integer() else f"{lab}: {v:,.1f}"


def notice_template(block_name: str, owner: str, action: str, reasons: list[str], lang: str = "en") -> str:
    reasons_txt = "; ".join(reasons[:3])
    if lang == "es":
        return (
            f"Aviso de prevención de ratas — {block_name}\n\n"
            f"Estimado/a responsable ({owner}):\n\n"
            f"Nuestros datos muestran un riesgo alto de actividad de ratas en esta zona en los próximos 90 días. "
            f"Motivos principales: {reasons_txt}.\n\n"
            f"Acción recomendada: {action}.\n\n"
            "Esta es una solicitud de ayuda, no una multa. Si necesita contenedores con tapa o apoyo, "
            "responda a este aviso y lo conectaremos con el programa correspondiente.\n\nGracias,\nEquipo RatGuard"
        )
    return (
        f"Rat prevention notice — {block_name}\n\n"
        f"Dear {owner},\n\n"
        f"Our data shows a high risk of rat activity on this block over the next 90 days. "
        f"Main reasons: {reasons_txt}.\n\n"
        f"Recommended action: {action}.\n\n"
        "This is a request for help, not a fine. If you need lidded containers or other support, "
        "reply to this notice and we will connect you with the right program.\n\nThank you,\nThe RatGuard team"
    )
