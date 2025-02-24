import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
import plotly.express as pxk 
import numpy as np


# Page config
st.set_page_config(page_title="EcoGPT - Carbon Aware LLM Router", layout="wide")

MODEL_METRICS = {
    "gpt-4o-mini": {
        "accuracy": 0.94,
        "completion_tokens": 300,
        "inference_time": 1.2,
        "co2_emissions": 0.02,
        "cost_per_1000_tokens": 0.03
    },
    "claude-3-5-sonnet": {
        "accuracy": 0.91,
        "completion_tokens": 280,
        "inference_time": 1.0,
        "co2_emissions": 0.018,
        "cost_per_1000_tokens": 0.003
    },
    "grok-2": {
        "accuracy": 0.85,
        "completion_tokens": 200,
        "inference_time": 0.8,
        "co2_emissions": 0.015,
        "cost_per_1000_tokens": 0.001
    }
}

MODEL_METRICS = {
    "gpt-4o-mini": {
        "accuracy": 0.94,
        "completion_tokens": 4096,
        "inference_time": 2.5,
        "co2_emissions": 0.03,
        "cost_per_1000_tokens": 0.03
    },
    "claude-3-5-sonnet": {
        "accuracy": 0.91,
        "completion_tokens": 4096,
        "inference_time": 1.8,
        "co2_emissions": 0.025,
        "cost_per_1000_tokens": 0.003
    },
    "grok-2": {
        "accuracy": 0.82,
        "completion_tokens": 8192,
        "inference_time": 3.0,
        "co2_emissions": 0.04,
        "cost_per_1000_tokens": 0.001
    }
}

# Custom CSS
st.markdown("""
    <style>
    .stTextInput > div > div > input {
        border: 2px solid #4CAF50;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        height: 3em;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Description
st.title("🌍 EcoLLM: Sustainable AI Routing")
st.markdown("Optimizing AI for performance and sustainability")

# Query Interface
prompt = st.text_area(
    "Enter your query:",
    height=100,
    help="Your query will be analyzed and routed to the most efficient model based on type and requirements"
)

def plot_pareto_matrix(model_metrics, selected_model=None, selected_model_data=None):
    df = pd.DataFrame.from_dict(model_metrics, orient="index")
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Model"}, inplace=True)

    # Normalize cost, CO2, and time for better visualization
    df["Normalized Cost"] = df["cost_per_1000_tokens"] * 500
    df["Normalized CO₂"] = df["co2_emissions"] * 1000
    df["Normalized Time"] = df["inference_time"] * 100

    # Initialize the figure
    fig = go.Figure()

    # Highlight the selected model within the dataset (no extra point)
    for i, row in df.iterrows():
        is_selected = row["Model"] == selected_model  # ✅ Check if it's the selected model
        
        fig.add_trace(go.Scatter3d(
            x=[row["Normalized Cost"]],
            y=[row["Normalized CO₂"]],
            z=[row["Normalized Time"]],
            mode="markers",
            marker=dict(
                size=14 if is_selected else 10,  # ✅ Bigger size for selected model
                color="red" if is_selected else f"rgb({255 * (i / len(df))}, 150, {255 - 255 * (i / len(df))})",
                opacity=1 if is_selected else 0.8,
                symbol="diamond" if is_selected else "circle",  # ✅ Diamond shape for selected model
                line=dict(width=3, color="white") if is_selected else None  # ✅ Outline for selected
            ),
            name=f"🔥 Selected: {row['Model']}" if is_selected else row["Model"]
        ))

    # Update Layout with Dark Theme
    fig.update_layout(
        scene=dict(
            xaxis_title="Normalized Cost",
            yaxis_title="Normalized CO₂",
            zaxis_title="Normalized Time",
            xaxis=dict(color="white"),  # ✅ White axis labels
            yaxis=dict(color="white"),
            zaxis=dict(color="white")
        ),
        legend=dict(
            x=1.05,  # ✅ Move legend to the right
            y=0.5,
            font=dict(color="white")  # ✅ White legend text
        ),
        paper_bgcolor="black",  # ✅ Dark background for contrast
        plot_bgcolor="black"
    )

    return fig



if st.button("🚀 Process Query", use_container_width=True):
    if prompt:
        with st.spinner('Analyzing query and routing to optimal model...'):
            response = requests.post(
                "http://127.0.0.1:5000/process_prompt",
                json={"prompt": prompt}
            )
            data = response.json()
            
            # Show Query Analysis
            st.markdown("### 🔍 Query Analysis")
            analysis_col1, analysis_col2 = st.columns(2)
            with analysis_col1:
                st.info(f"**Query Type:** {data['query_analysis']['type']}")
                st.info(f"**Required Accuracy:** {data['query_analysis']['required_accuracy']}")
                st.info(f"**Cost Sensitivity:** {data['query_analysis']['cost_sensitivity']}")

            with analysis_col2:
                st.info(f"**Selected Model:** {data['model_selection']['selected_model']}")
                st.info(f"**Power Consumption Factor:** {data['model_selection']['power_consumption_factor']}")
                st.info(f"**Base Cost per Token:** ${data['model_selection']['base_cost_per_token']}")



            # Performance Metrics
            print("performance data is here------------------------------------------",data)
            st.markdown("### 📊 Performance Metrics & Environmental Impact")
          
            home_electricity = data["real_world_impact"].get("🏡 Home Electricity Saved (hrs)", 0)
            bike_rides = data["real_world_impact"].get("🚲 Bike Rides Instead of Car Trips", 0)
            plastic_bottles = data["real_world_impact"].get("🌊 Plastic Bottles Not Produced", 0)
            emails_offset = data["real_world_impact"].get("📧 Emails Offset", 0)

            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("🏡 Home Electricity Saved (hrs)", f"{home_electricity:.4f}")
            with metric_col2:
                st.metric("🚲 Bike Rides Instead of Cars", f"{bike_rides:.4f}")

            metric_col3, metric_col4 = st.columns(2)
            with metric_col3:
                st.metric("🌊 Plastic Bottles Not Produced", f"{plastic_bottles:.4f}")
            with metric_col4:
                st.metric("📧 Emails Offset", f"{emails_offset:.0f}")

            # Model Response
            st.markdown("### 🤖 Model Response")
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                {data['performance']['response']}
            </div>
            """, unsafe_allow_html=True)
            
            
            st.markdown("### 📊 Pareto Model Selection Matrix")

            if "model_selection" in data and "selected_model" in data["model_selection"]:
                selected_model = data["model_selection"]["selected_model"]

                selected_model_data = {
                    "accuracy": data["model_selection"].get("accuracy", 0),
                    "cost_per_1000_tokens": data["model_selection"].get("cost", 0),
                    "co2_emissions": data["model_selection"].get("co2_emissions", 0),
                    "inference_time": data["model_selection"].get("inference_time", 0)
                }

                pareto_fig = plot_pareto_matrix(MODEL_METRICS, selected_model, selected_model_data)
                st.plotly_chart(pareto_fig)

                reasons = []
                ## fUTURE SCOPE
                # st.markdown(f"###  Why was **{selected_model}** chosen?")

# Dynamically explain the reasoning based on its actual selection criteria

            # Check what made this model the best choice
            # if round(data["model_selection"].get("cost_per_1000_tokens", float("inf")), 5) == round(
            #     min(model["cost_per_1000_tokens"] for model in MODEL_METRICS.values()), 5
            # ):
            #     reasons.append("💰 **Cost Efficient:** This model was chosen because it has the lowest token cost.")

            # if round(data["model_selection"].get("inference_time", float("inf")), 5) == round(
            #     min(model["inference_time"] for model in MODEL_METRICS.values()), 5
            # ):
            #     reasons.append("⚡ **Fastest Inference:** This model delivers responses quicker than others.")

            # if round(data["model_selection"].get("accuracy", 0), 5) == round(
            #     max(model["accuracy"] for model in MODEL_METRICS.values()), 5
            # ):
            #     reasons.append("🎯 **Highest Accuracy:** This model was chosen because it provides the best responses.")

            # if round(data["model_selection"].get("co2_emissions", float("inf")), 5) == round(
            #     min(model["co2_emissions"] for model in MODEL_METRICS.values()), 5
            # ):
            #     reasons.append("🌱 **Eco-Friendly:** This model produces the lowest CO₂ emissions.")

            # # Display all relevant reasons
            # if reasons:
            #     for reason in reasons:
            #         st.markdown(f"- {reason}")
            # else:
            #     st.markdown("🤷 This model was chosen based on a balance of all factors.")
                
            # st.write("🔍 Debugging Selection Criteria:")
            # st.write("Selected Model:", data["model_selection"]["selected_model"])
            # st.write("Selected Model Cost:", data["model_selection"].get("cost_per_1000_tokens", "N/A"))
            # st.write("Min Model Cost:", min(model["cost_per_1000_tokens"] for model in MODEL_METRICS.values()))
            # st.write("Selected Model Inference Time:", data["model_selection"].get("inference_time", "N/A"))
            # st.write("Min Model Inference Time:", min(model["inference_time"] for model in MODEL_METRICS.values()))
            # st.write("Selected Model Accuracy:", data["model_selection"].get("accuracy", "N/A"))
            # st.write("Max Model Accuracy:", max(model["accuracy"] for model in MODEL_METRICS.values()))
            # st.write("Selected Model CO₂ Emissions:", data["model_selection"].get("co2_emissions", "N/A"))
            # st.write("Min Model CO₂ Emissions:", min(model["co2_emissions"] for model in MODEL_METRICS.values()))


                # if selected_model_data["accuracy"] >= 0.9:
                #     reasons.append("✅ **High Accuracy:** This model offers excellent precision, making it suitable for complex queries.")
                # if selected_model_data["cost_per_1000_tokens"] < 0.002:
                #     reasons.append("💰 **Low Cost:** It was selected for its budget-friendly cost per 1000 tokens.")
                # if selected_model_data["co2_emissions"] < 0.02:
                #     reasons.append("🌿 **Eco-Friendly:** The model minimizes carbon emissions, aligning with sustainability goals.")
                # if selected_model_data["inference_time"] < 1.5:
                #     reasons.append("⚡ **Fast Inference:** It responds quickly, reducing wait times.")

                # # Display key reasons dynamically
                # for reason in reasons:
                #     st.markdown(reason)
                
                # # If no strong reason is found (fallback)
                # if not reasons:
                #     st.markdown("🤷 The selection was made based on overall optimization of cost, accuracy, and sustainability.")

    else:
        st.error("⚠️ No model selection data available.")
            

st.markdown("---")  # Divider for clarity
st.markdown(
    "📜 **Sources for Sustainability Metrics:** "
    "[US EPA CO₂ Equivalencies](https://www.epa.gov/rad/environmental-benefits-calculator-calculations-and-references), "
    "[UK Gov Transport CO₂ Data](https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2021), "
    "[UNEP Plastic Pollution Report](https://www.unep.org/resources/report/plastics-and-climate-change), "
    "[IEA Digital Carbon Footprint](https://www.iea.org/reports/the-carbon-footprint-of-streaming)"
)



# Footer
st.markdown("---")
st.markdown("Made with ❤️ for AAAI 2025 Hackathon")