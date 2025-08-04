import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI
from fpdf import FPDF
from io import BytesIO


class ContractPDF(FPDF):
    def header(self):
        self.set_font('Times', 'B', 14)
        self.cell(0, 10, 'Battery Warranty and Usage Agreement', 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, num, label):
        self.set_font('Times', 'B', 12)
        self.cell(0, 10, f'{num}. {label}', 0, 1)
        self.ln(2)

    def chapter_body(self, text):
        self.set_font('Times', '', 11)
        self.multi_cell(0, 8, text)
        self.ln()

def generate_contract_pdf(
    oem_name="BatteryCo Ltd.",
    optimiser_name="Green Energy Optimisers Inc.",
    battery_capacity=4.0,    # MWh
    battery_power=1.0,       # MW
    degradation_model="Cycle-Based Model (degradation linked to cycles and usage depth)",
    degradation_factors=None,  # dict e.g. {"cycles": 1.0, "dod": 1.0}
    warranty_years=10,
    warranty_cycles=4000,
    max_degradation_pct=20,
    max_cycles_per_day=4,
    arbitration_body="[Arbitration Body]",
    arbitration_location="[Location]",
    jurisdiction="[Jurisdiction]",
    filename='battery_warranty_contract.pdf'
):
    pdf = ContractPDF()
    pdf.add_page()

    # Introduction
    pdf.chapter_title('1', 'Introduction')
    intro_text = (
        "This Contract outlines the terms and conditions governing the warranty coverage, operational parameters, "
        "and financial arrangements related to the battery system described herein. The Parties agree to cooperate "
        "to ensure optimal performance, maintenance, and compliance with the agreed warranty standards."
    )
    pdf.chapter_body(intro_text)

    # Battery Description
    pdf.chapter_title('2', 'Battery Description')
    battery_text = (
        f"The battery system subject to this Agreement is specified as follows:\n\n"
        f"- Capacity: {battery_capacity} MWh\n"
        f"- Power: {battery_power} MW\n"
        f"- Degradation Model: {degradation_model}\n"
        f"- Relevant Degradation Factors:\n"
    )
    if degradation_factors:
        for factor, value in degradation_factors.items():
            battery_text += f"  - {factor.capitalize()}: {value}\n"
    else:
        battery_text += "  - Cycles: 1.0\n  - Depth of Discharge (DoD): 1.0\n"
    pdf.chapter_body(battery_text)

    # Warranty Terms
    pdf.chapter_title('3', 'Warranty Terms')
    warranty_text = (
        f"- Warranty Coverage Duration: This warranty shall remain in effect for a period of {warranty_years} years or {warranty_cycles} cycles, whichever occurs first.\n\n"
        f"- Warranty Limits and Conditions:\n"
        f"  - Maximum capacity degradation allowed under this warranty shall not exceed {max_degradation_pct}%.\n"
        f"  - The battery shall not be operated beyond {max_cycles_per_day} cycles per day to maintain warranty validity.\n"
        f"  - The Parties shall monitor compliance through the ReWarrant platform, which shall provide real-time tracking of battery usage and degradation.\n\n"
        f"- Servicing and Maintenance:\n"
        f"  The OEM shall be responsible for routine servicing and maintenance as specified in the service schedule to ensure battery performance. "
        f"The Optimiser/Integrator shall cooperate in providing operational data and access to the battery system for maintenance activities."
    )
    pdf.chapter_body(warranty_text)

    # Market Participation and Usage Profile
    pdf.chapter_title('4', 'Market Participation and Usage Profile')
    market_text = (
        "The Parties acknowledge that no specific market participation has been designated under this Agreement at this time. "
        "Accordingly, the operational profile shall adhere strictly to the warranty conditions set forth herein, and any future market participation "
        "affecting usage patterns shall be subject to additional agreement amendments."
    )
    pdf.chapter_body(market_text)

    # Financial Arrangements
    pdf.chapter_title('5', 'Financial Arrangements')
    financial_text = (
        "There are no additional servicing costs or fees agreed upon at this stage beyond those standard to routine maintenance. "
        "The ReWarrant fee schedule and any Optimiser premium shall be mutually determined in subsequent negotiations as market participation parameters become defined."
    )
    pdf.chapter_body(financial_text)

    # Obligations and Responsibilities
    pdf.chapter_title('6', 'Obligations and Responsibilities')
    obligations_text = (
        "- OEM Obligations: To provide a battery system conforming to the specifications herein and to uphold warranty commitments. "
        "To perform necessary maintenance and ensure transparency in reporting.\n\n"
        "- Optimiser/Integrator Obligations: To operate the battery within agreed parameters, report operational data accurately, and cooperate in maintenance activities.\n\n"
        "- Breach of Terms: Failure by either party to comply with these obligations may result in suspension or termination of warranty coverage, subject to remedial provisions and dispute resolution procedures."
    )
    pdf.chapter_body(obligations_text)

    # Termination and Dispute Resolution
    pdf.chapter_title('7', 'Termination and Dispute Resolution')
    termination_text = (
        f"This Agreement may be terminated by mutual consent or upon material breach by either party following written notice and opportunity to cure. "
        f"All disputes arising under this Agreement shall be resolved through good faith negotiations, and if unresolved, through arbitration under the rules of {arbitration_body}, "
        f"with jurisdiction in {arbitration_location}."
    )
    pdf.chapter_body(termination_text)

    # Miscellaneous
    pdf.chapter_title('8', 'Miscellaneous')
    misc_text = (
        f"- Confidentiality: Both Parties agree to maintain the confidentiality of all proprietary and operational information exchanged under this Agreement.\n\n"
        f"- Governing Law: This Agreement shall be governed by and construed in accordance with the laws of {jurisdiction}.\n\n"
        f"- Contacts: Official communications shall be directed to the authorized representatives of each party as designated in writing."
    )
    pdf.chapter_body(misc_text)

    # Signatures
    pdf.chapter_title('9', 'Signatures')
    signature_text = (
        f"IN WITNESS WHEREOF, the Parties hereto have executed this Agreement as of the dates written below.\n\n"
        f"{oem_name} (OEM)                      {optimiser_name} (Optimiser/Integrator)\n\n"
        f"Signature: ________________________      Signature: ________________________\n"
        f"Name: _____________________________      Name: _____________________________\n"
        f"Title: ____________________________      Title: ____________________________\n"
        f"Date: _____________________________      Date: _____________________________"
    )
    pdf.chapter_body(signature_text)

    pdf_bytes = pdf.output(dest='S').encode('latin1')  # get PDF as byte string
    return pdf_bytes

# Float input helper
def float_input(label, key, default=1.0, min_val=0.01, max_val=10.0):
    val_str = st.text_input(label, value=str(default), key=key)
    try:
        val = float(val_str)
        if val < min_val or val > max_val:
            st.warning(f"Value must be between {min_val} and {max_val}.")
            return None
        return val
    except ValueError:
        st.warning("Please enter a valid number.")
        return None

# Mock degradation curve plot
def plot_mock_degradation():
    years = np.arange(0, 11)
    previous_warranty = 2 * np.sqrt(years)
    new_warranty = 2.5 * np.sqrt(years)

    plt.figure(figsize=(6, 3))
    plt.plot(years, previous_warranty, label="Previous Warranty", marker='o')
    plt.plot(years, new_warranty, label="Proposed Warranty", marker='o')
    plt.xlabel("Years")
    plt.ylabel("Capacity Degradation (%)")
    plt.title("Modeled Battery Degradation Over Time")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Proposal output section
def show_proposal():

    years = np.arange(1, 11)

    # Degradation-driven additional servicing cost, compared to baseline
    degradation = 2.5 * np.sqrt(years) / 100
    servicing_base = 15000
    baseline_cost = servicing_base
    servicing_costs = servicing_base * (1 + 2 * degradation)  # higher than baseline in early years

    # Years 9 and 10 — reduced costs due to no warranty (lower than baseline)
    servicing_costs[-2:] = servicing_base * 0.3

    # Compute increased servicing cost (compared to baseline)
    increased_servicing = servicing_costs - baseline_cost

    # ReWarrant Fee only applies to years 1–8
    rewarrant_fee = 1000
    rewarrant_fees = np.array([rewarrant_fee if year <= 8 else 0 for year in years])

    # Total cost to OEM
    total_costs = increased_servicing + rewarrant_fees
    total_additional_cost = np.sum(increased_servicing + rewarrant_fees)
    optimiser_premium = total_additional_cost / 8
    optimiser_premiums = np.array([optimiser_premium if year <= 8 else 0 for year in years])

    # Total = all cost components
    total_costs = increased_servicing + rewarrant_fees + optimiser_premiums

    # Build table
    data = {
        "Year": years,
        "Increased Servicing Cost (€)": increased_servicing.round(0).astype(int),
        "ReWarrant Fee (€)": rewarrant_fees.astype(int),
        "Optimiser Premium (€)": optimiser_premiums.round(0).astype(int),
        "Total Additional Cost (€)": total_costs.round(0).astype(int),
    }

    df = pd.DataFrame(data)

    # Add cumulative row
    totals = {
        "Year": "Total (Cumulative)",
        "Increased Servicing Cost (€)": df["Increased Servicing Cost (€)"].sum(),
        "ReWarrant Fee (€)": df["ReWarrant Fee (€)"].sum(),
        "Optimiser Premium (€)": df["Optimiser Premium (€)"].sum(),
        "Total Additional Cost (€)": df["Total Additional Cost (€)"].sum()
    }

    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

    # Render with better formatting, no index
    st.dataframe(df.style.format({
        "Increased Servicing Cost (€)": "{:,.0f}",
        "ReWarrant Fee (€)": "{:,.0f}",
        "Optimiser Premium (€)": "{:,.0f}",
        "Total Additional Cost (€)": "{:,.0f}",
    }), use_container_width=True, hide_index=True)

# OEM Tab logic
def oem_tab():
    with st.expander("1 Define Degradation Model Variables"):
        model_choice = st.radio(
            "Select base degradation model:",
            [
                "Cycle-Based Model (degradation linked to cycles and usage depth)",
                "Stress Accumulation Model (cumulative stress from multiple usage factors)"
            ],
            index=0
        )
        st.session_state["degradation_model"] = model_choice

        st.markdown("Select variables to include and specify their positive float influence factors.")

        variables = {
            "Number of Cycles (per year)": "cycles",
            "Depth of Discharge (DoD)": "dod",
            "Ambient Temperature": "temp",
            "C-rate (Charge/Discharge Speed)": "c_rate",
            "Rest Time / Idle Duration": "rest"
        }

        influence_factors = {}

        for label, key in variables.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                include = st.checkbox(label, value=(key in ["cycles", "dod"]), key=f"{key}_include")
            if include:
                with col2:
                    val = float_input("", key=f"{key}_factor", default=1.0)
                influence_factors[key] = val
            else:
                influence_factors[key] = None

        st.session_state["degradation_factors"] = influence_factors

    with st.expander("2 Proposed warranty contract structure"):
   
        st.markdown("""
                    The following warranty terms are proposed after consulting wioth the optimiser:
        - Up to 4 cycles per day allowed under new warranty  
        - Warranty coverage for 10 years or 4000 cycles, whichever comes first  
        - OEM guarantees max capacity degradation below 20% within warranty period  
        - Compliance monitored via ReWarrant platform 
        - Optimiser premium based on increased servicing costs
        - Future market participation may trigger additional agreement amendments, to be agreed upon by both parties 
        """)

    with st.expander("3 Impact on degradation"):

        plot_mock_degradation()
    with st.expander("4 Financial impact"):
        show_proposal()

    # with st.expander("3. Contract"):

    # Optional name inputs if not set earlier
    st.text_input("OEM Company Name", key="oem_name", value="BatteryCo Ltd.")
    st.text_input("Optimiser Company Name", key="optimiser_name", value="Green Energy Optimisers Inc.")

    pdf_bytes = None

    if st.button("Generate Contract"):
        with st.spinner("Generating contract..."):
            pdf_bytes = generate_contract_pdf(
                oem_name=st.session_state.get("oem_name", "BatteryCo Ltd."),
                optimiser_name=st.session_state.get("optimiser_name", "Green Energy Optimisers Inc."),
                battery_capacity=st.session_state.get("battery_capacity", 4.0),
                battery_power=st.session_state.get("battery_power", 1.0),
                degradation_model=st.session_state.get("degradation_model", "Cycle-Based Model (degradation linked to cycles and usage depth)"),
                degradation_factors=st.session_state.get("degradation_factors", {"cycles":1.0, "dod":1.0}),
                warranty_years=10,
                warranty_cycles=4000,
                max_degradation_pct=20,
                max_cycles_per_day=4,
                arbitration_body="[Arbitration Body]",
                arbitration_location="[Location]",
                jurisdiction="[Jurisdiction]"
            )


    if pdf_bytes:
        st.download_button(
            label="Download Contract PDF",
            data=pdf_bytes,
            file_name="battery_warranty_contract.pdf",
            mime="application/pdf"
        )




def optimiser_tab():
    st.header("For optimisers")

    with st.expander("1 Battery Parameters (provided by OEM)"):
        st.markdown("""
        These are battery details provided by the OEM and the degradation model you will use.
        """)

        # Simulate imported OEM battery parameters (in real app, get from session_state or shared data)
        # For demo, fallback to placeholders
        battery_capacity = st.session_state.get("battery_capacity", 4.0)  # in MWh or hours
        battery_power = st.session_state.get("battery_power", 1.0)        # in MW
        degradation_model = st.session_state.get("degradation_model", "Cycle-Based Model")
        degradation_factors = st.session_state.get("degradation_factors", {})

        st.markdown(f"- **Battery Capacity:** {battery_capacity} MWh")
        st.markdown(f"- **Battery Power:** {battery_power} MW")
        st.markdown(f"- **Degradation Model:** {degradation_model}")
        st.markdown(f"- **Degradation Factors:** {degradation_factors}")

    with st.expander("2 Project financing and incentives"):
        st.markdown("""
        Specify the financing and incentives for the battery project.
        This will help in calculating the financial impact of the warranty proposal.
        """)
        project_horizon = st.number_input(
            "Project Horizon (years)", 
            min_value=1, max_value=20, value=10, step=1,
            key="project_horizon"
        )
        financing_rate = float_input(
            "Financing Rate (%)", 
            key="financing_rate", 
            default=5.0, min_val=0.01, max_val=20.0
        )
        project_goals = st.radio(
            "Project Goals",
            ["Maximize total Revenues", "Guarantee project lifespan", ""],
            index=0,
            key="project_goals"
        )

    with st.expander("3 Desired Market Participation and Usage"):
        st.markdown("Select markets you are active in and specify desired daily cycles and expected revenues.")

        markets = {
            "FCR (Frequency Containment Reserve)": {},
            "aFRR (automatic Frequency Restoration Reserve)": {},
            "mFRR (manual Frequency Restoration Reserve)": {},
            "ID (Intraday Arbitrage)": {}
        }

        # Header row for the "table"
        cols = st.columns([3, 1, 2, 3])
        cols[0].markdown("**Market**")
        cols[1].markdown("**Active?**")
        cols[2].markdown("**Cycles per day**")
        cols[3].markdown("**Expected daily revenue (€)**")

        for market in markets.keys():
            cols = st.columns([3, 1, 2, 3])
            cols[0].markdown(f"**{market}**")
            active = cols[1].checkbox("", key=f"{market}_active")

            if active:
                cycles = cols[2].number_input(
                    "", 
                    min_value=0.0, max_value=4.0, value=1.0, step=0.1,
                    key=f"{market}_cycles",
                    format="%.1f"
                )
                revenue = cols[3].number_input(
                    "", 
                    min_value=0.0, value=100.0, step=10.0,
                    key=f"{market}_revenue"
                )
                markets[market]["active"] = True
                markets[market]["cycles"] = cycles
                markets[market]["revenue"] = revenue
            else:
                markets[market]["active"] = False
                markets[market]["cycles"] = 0.0
                markets[market]["revenue"] = 0.0

        st.session_state["optimiser_markets"] = markets

    with st.expander("4 Warranty Restructuring Optimization"):
        st.write("Placeholder. This section will outline the suggested new warranty structure.")

    with st.expander(" 5 Financial Impact"):
        st.write("Placeholder. This section will show expected revenues and warranty costs based on the proposed structure.")


# Placeholder tab for integrators
def integrator_tab():
    st.header("For integrators")
    st.write("We are working on a warranty contract with the optimiser and the OEM. This tab is work in progress. Hang tight...")

# Tab layout
tabs = st.tabs(["For OEMs", "For Optimisers", "For Integrators"])

with tabs[0]:
    oem_tab()

with tabs[1]:
    optimiser_tab()

with tabs[2]:
    integrator_tab()
