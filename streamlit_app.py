"""AutoValue: Streamlit interface for the existing used-car FastAPI service."""
import json
import math
import os
from datetime import datetime, timezone
from html import escape

import altair as alt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="AutoValue | Used-car estimates", page_icon="🚘", layout="wide")
API_URL = os.environ.get("CAR_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.markdown("""
<style>
.stApp {background:radial-gradient(ellipse at 88% 0%,#173541 0%,#0b111b 48%);}
.block-container {max-width:1360px;padding-top:2.2rem;padding-bottom:3rem;}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stSidebar"] {background:#0e1824;border-right:1px solid #233343;}
h1,h2,h3 {letter-spacing:-.035em;}
.av-brand {font-size:1.55rem;font-weight:800;letter-spacing:-.05em;color:#f3f8fb;}
.av-brand span {color:#64e2cc;}
.av-eyebrow {color:#64e2cc;font-size:.72rem;letter-spacing:.17em;font-weight:700;text-transform:uppercase;}
.av-hero {position:relative;overflow:hidden;border:1px solid #2c4653;border-radius:24px;
 background:linear-gradient(120deg,#162a3a,#10212d 62%,#194738);padding:34px 38px;margin:12px 0 22px;}
.av-hero h1 {font-size:clamp(2rem,4vw,3.4rem);line-height:1.12;margin:12px 0;color:#f5fafc;max-width:700px;}
.av-hero p {color:#b9cad4;max-width:650px;font-size:1.04rem;margin:0;}
.av-pills {display:flex;flex-wrap:wrap;gap:9px;margin-top:23px;}
.av-pill {border:1px solid #426071;border-radius:999px;padding:6px 12px;font-size:.77rem;color:#d2e2e9;}
.av-card {padding:22px 24px;border-radius:18px;background:#142331;border:1px solid #304453;margin-bottom:14px;}
.av-card.primary {background:linear-gradient(120deg,#17493f,#183039);border-color:#388374;}
.av-label {color:#adbfcd;font-size:.78rem;letter-spacing:.07em;text-transform:uppercase;}
.av-value {font-size:clamp(1.8rem,3vw,2.7rem);font-weight:750;letter-spacing:-.04em;color:#f5fafc;margin:7px 0;}
.av-detail {color:#bacbd6;font-size:.85rem;line-height:1.6;}
.av-empty {padding:34px 25px;border:1px dashed #3e586b;border-radius:18px;background:#111f2d;}
.av-empty h3 {color:#eff8fb;margin:10px 0;}
.av-empty p {color:#a9bdca;line-height:1.7;}
.av-step {padding:11px 0;border-bottom:1px solid #263747;color:#b9cbd8;font-size:.86rem;}
.av-step b {color:#64e2cc;margin-right:10px;}
.av-footer {border-top:1px solid #2a3b4b;margin-top:32px;padding-top:18px;color:#8fa6b9;font-size:.8rem;}
[data-testid="stForm"] {border:1px solid #2f4456;border-radius:18px;background:#101d2b;padding:22px;}
[data-baseweb="tab-list"] {gap:18px;border-bottom:1px solid #2a3b4b;}
[data-baseweb="tab"] {padding:13px 8px;font-size:.96rem;}
[data-testid="stFormSubmitButton"] button {border-radius:10px;font-weight:700;min-height:46px;}
[data-testid="stMetric"] {background:#142331;border:1px solid #304453;border-radius:15px;padding:18px;}
@media(max-width:640px){.av-hero{padding:24px 20px;}.block-container{padding-left:1rem;padding-right:1rem;}}
</style>
""", unsafe_allow_html=True)


def api_request(path, payload=None):
    try:
        if payload is None:
            response = requests.get(API_URL + path, timeout=30)
        else:
            response = requests.post(API_URL + path, json=payload, timeout=120)
        if not response.ok:
            st.error("The prediction service could not accept these details.")
            with st.expander("See error details"):
                st.text(response.text)
            st.stop()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        st.error("The prediction service is unavailable. Start the servers with Cell 60, then refresh.")
        with st.expander("See connection details"):
            st.text(str(exc))
        st.stop()


@st.cache_data(ttl=60, show_spinner=False)
def load_reference_data(api_url):
    # Include the API URL in the cache key when switching environments.
    return api_request("/metadata"), api_request("/feature-importance")


meta, importance_records = load_reference_data(API_URL)
stats = meta["numeric_training_summary"]
options = meta["categorical_options"]
reference_year = int(meta["age_reference_year"])
coverage = float(meta["uncertainty"]["validation_coverage_percent"])


def card(label, value, detail="", primary=False):
    css = "av-card primary" if primary else "av-card"
    st.markdown(
        f'<div class="{css}"><div class="av-label">{escape(str(label))}</div>'
        f'<div class="av-value">{escape(str(value))}</div>'
        f'<div class="av-detail">{escape(str(detail))}</div></div>',
        unsafe_allow_html=True,
    )


def show_result(result, prefix):
    card("Estimated selling price", f"₹{result['prediction_lakh']:.2f} lakh",
         f"₹{result['prediction_inr']:,.0f} · model estimate", primary=True)
    card("Estimated model-uncertainty range",
         f"₹{result['low_lakh']:.2f}–{result['high_lakh']:.2f} L",
         f"Low to high · {result['mc_samples']} MC-dropout draws")
    st.caption(
        "This range is uncalibrated, not a guaranteed 90% interval for sale prices. "
        f"Observed validation coverage: {result['validation_coverage_percent']:.2f}%."
    )
    points = pd.DataFrame([
        {"Estimate": "Price", "Low": result["low_lakh"],
         "High": result["high_lakh"], "Point": result["prediction_lakh"]}
    ])
    base = alt.Chart(points)
    interval = base.mark_rule(strokeWidth=8, color="#3c796e").encode(
        x=alt.X("Low:Q", title="Price (₹ lakh)", scale=alt.Scale(zero=False)),
        x2="High:Q", y=alt.Y("Estimate:N", title=None, axis=None),
        tooltip=[alt.Tooltip("Low:Q", format=".2f"), alt.Tooltip("High:Q", format=".2f")],
    )
    point = base.mark_point(filled=True, size=180, color="#70eed3").encode(
        x="Point:Q", y="Estimate:N", tooltip=[alt.Tooltip("Point:Q", format=".2f")]
    )
    st.altair_chart((interval + point).properties(height=65), width="stretch")
    st.caption("Dot: standard prediction. Bar: sampled model-uncertainty range.")
    for note in result.get("notes", []):
        st.warning(note)


with st.sidebar:
    st.markdown('<div class="av-brand">Auto<span>Value</span> ↗</div>', unsafe_allow_html=True)
    st.caption("USED-CAR PRICE EXPLORER")
    st.divider()
    st.markdown("**From details to a decision**")
    st.markdown('''<div class="av-step"><b>01</b> Enter the car's details</div>
    <div class="av-step"><b>02</b> Explore the price estimate</div>
    <div class="av-step"><b>03</b> Compare age and kilometres</div>''', unsafe_allow_html=True)
    st.divider()
    st.markdown("**Dataset reference**")
    st.write(f"Age reference year: **{reference_year}**")
    st.caption("Estimates reflect the training dataset. They are not live market quotations.")
    st.caption("Submitted estimates are recorded in the project prediction log.")

st.markdown('''<div class="av-hero"><div class="av-eyebrow">Indian used-car market · Price explorer</div>
<h1>Know the car.<br>Explore its value.</h1>
<p>Turn vehicle details into a price estimate, explore uncertainty, and see how age and usage change the result.</p>
<div class="av-pills"><span class="av-pill">₹ Lakh estimates</span><span class="av-pill">What-if comparisons</span>
<span class="av-pill">Feature insights</span></div></div>''', unsafe_allow_html=True)

estimate_tab, compare_tab, insights_tab = st.tabs(["01  Price estimate", "02  What-if studio", "03  Model insights"])

with estimate_tab:
    form_col, result_col = st.columns([1.3, 1], gap="large")
    with form_col:
        st.subheader("Tell us about the car")
        st.caption(f"Manufacturing year must be {reference_year} or earlier to match this model's age definition.")
        with st.form("car_details"):
            st.markdown("**Vehicle identity**")
            a, b = st.columns(2)
            brand = a.selectbox("Brand", meta["brand_options"], key="brand")
            year = b.number_input("Manufacturing year", min_value=1900, max_value=reference_year,
                value=max(1900, min(reference_year, reference_year-int(stats["car_age"]["median"]))), step=1, key="year")
            a, b = st.columns(2)
            fuel = a.selectbox("Fuel", options["fuel"], key="fuel")
            transmission = b.selectbox("Transmission", options["transmission"], key="transmission")
            st.markdown("**Usage & ownership**")
            a, b = st.columns(2)
            km = a.number_input("Kilometres driven", min_value=0, value=max(0,int(stats["km_driven"]["median"])), step=1000, key="km")
            owner = b.selectbox("Ownership", options["owner"], key="owner")
            a, b = st.columns(2)
            seller = a.selectbox("Seller type", options["seller_type"], key="seller")
            seats = b.number_input("Seats", min_value=1, value=max(1,int(stats["seats"]["median"])), step=1, key="seats")
            st.markdown("**Engine & efficiency**")
            a, b = st.columns(2)
            engine = a.number_input("Engine capacity (cc)", min_value=1, value=max(1,int(stats["engine"]["median"])), step=50, key="engine")
            power = b.number_input("Maximum power (bhp)", min_value=0.1, value=max(0.1,float(stats["max_power"]["median"])), step=1.0, key="power")
            a, b = st.columns(2)
            mileage = a.number_input("Mileage value", min_value=0.1, value=max(0.1,float(stats["mileage_value"]["median"])), step=0.1, key="mileage")
            unit = b.selectbox("Mileage unit", options["mileage_unit"], key="unit")
            st.caption("Use the mileage unit stated for the vehicle. Prefilled numbers are training-data medians; replace them with the actual details.")
            submitted = st.form_submit_button("Estimate my car's price →", type="primary", width="stretch")

        if submitted:
            for name in ("base_inputs", "base_result", "scenario", "estimated_at"):
                st.session_state.pop(name, None)
            payload = dict(brand=brand, car_age=float(reference_year-year), km_driven=float(km),
                mileage_value=float(mileage), engine=float(engine), max_power=float(power), seats=int(seats),
                fuel=fuel, seller_type=seller, transmission=transmission, owner=owner, mileage_unit=unit)
            with st.spinner("Calculating your estimate and uncertainty range…"):
                result = api_request("/predict", payload)
            st.session_state.update(base_inputs=payload, base_result=result,
                scenario_age=int(payload["car_age"]), scenario_km=int(payload["km_driven"]),
                estimated_at=datetime.now(timezone.utc).isoformat())

    with result_col:
        st.subheader("Your valuation")
        if "base_result" not in st.session_state:
            st.markdown('''<div class="av-empty"><div class="av-eyebrow">Ready when you are</div>
            <h3>A clearer view of your car's value.</h3><p>Complete the form and select <b>Estimate my car's price</b>.
            Your price, model-uncertainty range, and downloadable summary will appear here.</p></div>''', unsafe_allow_html=True)
        else:
            used = st.session_state["base_inputs"]
            st.caption(f"{used['brand']} · {reference_year-int(used['car_age'])} · {used['fuel']} · {used['km_driven']:,.0f} km")
            show_result(st.session_state["base_result"], "base")
            with st.expander("View the details used"):
                st.dataframe(pd.DataFrame({"Detail": list(used), "Value": [str(x) for x in used.values()]}), hide_index=True, width="stretch")
            report = {"created_at_utc": st.session_state["estimated_at"], "inputs": used,
                "result": st.session_state["base_result"],
                "interpretation": "Uncalibrated model-uncertainty range; not a guaranteed 90% sale-price interval."}
            st.download_button("Download estimate · JSON", json.dumps(report, indent=2, ensure_ascii=False),
                file_name="car_price_estimate.json", mime="application/json", width="stretch")

with compare_tab:
    st.subheader("One car. Different scenarios.")
    st.write("Change age and kilometres while keeping the other vehicle details fixed.")
    if "base_result" not in st.session_state:
        st.info("Create an estimate in the first tab to unlock the what-if sliders.")
    else:
        original = st.session_state["base_inputs"]
        left, right = st.columns([1, 1.15], gap="large")
        with left:
            with st.form("scenario_form"):
                st.markdown("**Build your scenario**")
                age_max = max(1, math.ceil(stats["car_age"]["max"]), int(original["car_age"]))
                km_max = max(1000, math.ceil(stats["km_driven"]["max"]), int(original["km_driven"]))
                new_age = st.slider(f"Age at reference year {reference_year} (years)", 0, age_max, step=1, key="scenario_age")
                new_km = st.slider("Kilometres driven", 0, km_max, step=1, key="scenario_km")
                compare = st.form_submit_button("Compare scenario →", type="primary", width="stretch")
            st.caption("A model sensitivity comparison, not a guaranteed depreciation forecast.")
            if compare:
                st.session_state.pop("scenario", None)
                scenario_inputs = {**original, "car_age": float(new_age), "km_driven": float(new_km)}
                with st.spinner("Calculating the comparison…"):
                    scenario_result = api_request("/predict", scenario_inputs)
                st.session_state["scenario"] = {"inputs": scenario_inputs, "result": scenario_result}
        with right:
            if "scenario" in st.session_state:
                scenario = st.session_state["scenario"]
                sr = scenario["result"]
                baseline = st.session_state["base_result"]["prediction_lakh"]
                delta = sr["prediction_lakh"] - baseline
                m1, m2, m3 = st.columns(3)
                m1.metric("Original", f"₹{baseline:.2f} L")
                m2.metric("Scenario", f"₹{sr['prediction_lakh']:.2f} L")
                m3.metric("Difference", f"₹{delta:+.2f} L")
                si = scenario["inputs"]
                st.caption(f"Evaluated scenario: {si['car_age']:.0f} years · {si['km_driven']:,.0f} km")
                comparison = pd.DataFrame([
                    {"Version": "Original", "Age (years)": original["car_age"], "Kilometres": original["km_driven"], "Price (₹ lakh)": baseline},
                    {"Version": "Scenario", "Age (years)": si["car_age"], "Kilometres": si["km_driven"], "Price (₹ lakh)": sr["prediction_lakh"]}])
                st.dataframe(comparison, hide_index=True, width="stretch")
                show_result(sr, "scenario")
            else:
                st.info("Move the sliders and select Compare scenario to see the price difference.")

with insights_tab:
    st.subheader("What does the model rely on?")
    st.write("Explore how prediction error changes when each feature is shuffled on validation data.")
    importance = pd.DataFrame(importance_records)
    if not importance.empty:
        chart_data = importance.rename(columns={"Feature": "Feature name", "Mean MAE increase (₹)": "Importance", "SD across repeats (₹)": "SD"}).copy()
        chart_data["Feature name"] = chart_data["Feature name"].str.replace("_", " ", regex=False).str.title()
        chart_data["Importance"] /= 1000
        chart_data["SD"] /= 1000
        chart_data["Lower"] = chart_data["Importance"] - chart_data["SD"]
        chart_data["Upper"] = chart_data["Importance"] + chart_data["SD"]
        base = alt.Chart(chart_data).encode(y=alt.Y("Feature name:N", sort="-x", title=None))
        bars = base.mark_bar(cornerRadiusEnd=5, color="#64dac3", size=18).encode(
            x=alt.X("Importance:Q", title="Increase in validation MAE (₹ thousand)"),
            tooltip=["Feature name:N", alt.Tooltip("Importance:Q", title="MAE increase (₹ thousand)", format=".2f"), alt.Tooltip("SD:Q", title="Repeat SD (₹ thousand)", format=".2f")])
        errors = base.mark_rule(color="#e3f1f7", strokeWidth=2).encode(x="Lower:Q", x2="Upper:Q")
        zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color="#637686").encode(x="zero:Q")
        st.altair_chart((bars + errors + zero).properties(height=max(260,len(chart_data)*33)), width="stretch")
        st.caption("Hover for values. Error bars show one standard deviation across repeated shuffles. Negative importance values are retained.")
        with st.expander("View or download the importance data"):
            st.dataframe(importance, hide_index=True, width="stretch")
            st.download_button("Download importance · CSV", importance.to_csv(index=False).encode("utf-8-sig"),
                "feature_importance.csv", "text/csv")
    with st.expander("How to interpret these results"):
        st.write("A larger increase means the model relies more on that feature. Correlated features may share importance. These are global patterns, not explanations of one car's price or proof of causation.")
        st.write(f"The uncertainty range uses {meta['uncertainty']['samples']} MC-dropout draws and the 5th–95th percentiles. It is uncalibrated: observed validation coverage was {coverage:.2f}%.")
        st.write(f"Vehicle age = {reference_year} − manufacturing year. This reference stays fixed to match training.")

st.markdown('<div class="av-footer">AutoValue · Used-car price estimation project · Built with Streamlit & FastAPI</div>', unsafe_allow_html=True)
