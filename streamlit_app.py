import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time

# Page config
st.set_page_config(page_title="EcoGPT - Carbon Aware LLM Router", layout="wide")

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
st.title("🌍 EcoGPT: Carbon-Aware LLM Router")
st.markdown("Optimizing AI for performance and sustainability")

# Query Interface
prompt = st.text_area(
    "Enter your query:",
    height=100,
    help="Your query will be analyzed and routed to the most efficient model based on type and requirements"
)

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
                st.info(f"""
                **Query Type:** {data['query_analysis']['type']}
                **Required Accuracy:** {data['query_analysis']['required_accuracy']}
                **Cost Sensitivity:** {data['query_analysis']['cost_sensitivity']}
                """)
            with analysis_col2:
                st.info(f"""
                **Selected Model:** {data['model_selection']['selected_model']}
                **Power Consumption Factor:** {data['model_selection']['power_consumption_factor']}
                **Base Cost per Token:** ${data['model_selection']['base_cost_per_token']}
                """)

            # Performance Metrics
            print("performance data is here------------------------------------------",data)
            st.markdown("### 📊 Performance Metrics & Environmental Impact")
            # metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            # with metric_col1:
            #     st.metric("Inference Time", f"{data['performance']['inference_time_seconds']:.2f}s")
            # with metric_col2:
            #     st.metric("CO₂ Emissions", f"{data['performance']['carbon_emissions']:.6f} kg")
            # with metric_col3:
            #     st.metric("Trees Saved 🌳", f"{data['real_world_impact']['🌳 Trees Saved']:.2f}")
            # with metric_col4:
            #     st.metric("Gaming PC Usage Reduced 🎮", f"{data['real_world_impact']['🎮 Gaming PC Usage Reduced (hrs)']:.2f}")

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


    # Show Real-World Impact

            # Carbon Offset 
            # st.markdown("### 💡 Fun Fact:")
            # st.success(f"Your query saved enough CO₂ to offset {data['real_world_impact']['📧 Emails Offset']:.0f} emails!")

            # Model Response
            st.markdown("### 🤖 Model Response")
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                {data['performance']['response']}
            </div>
            """, unsafe_allow_html=True)
            
            # # Historical Metrics
            # if st.button("📈 Show Historical Metrics"):
            #     metrics_response = requests.get("http://127.0.0.1:5000/metrics")
            #     historical_data = metrics_response.json()
                
            #     st.markdown("### Historical Performance")
                
            #     # Create DataFrame for visualization
            #     stats = historical_data['current_stats']
            #     df = pd.DataFrame(stats).T
                
            #     # Plot metrics
            #     fig = go.Figure()
            #     fig.add_trace(go.Bar(name='Avg CO₂', x=df.index, y=df['avg_co2']))
            #     fig.add_trace(go.Bar(name='Avg Inference Time', x=df.index, y=df['avg_inference_time']))
            #     fig.update_layout(barmode='group', title='Average Performance by Model')
            #     st.plotly_chart(fig)
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