import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import os

# --- Configuration ---
CSV_DIR = "../results/" # Path to the folder containing CSV files
csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
PLOT_DIR = "../plots/"

# Easily tweakable subplot widths (must sum to 1.0)
SUBPLOT_1_WIDTH = 1/3
SUBPLOT_2_WIDTH = 2/3

# Define color mappings
COLOR_MAP = {
    "Matches Golden FV": "#2ca02c",      # Green (Pass)
    "One or more faults": "#d62728",     # Red
    "False Perfect FV": "#ff7f0e",       # Orange
    "No result obtained": "#7f7f7f",     # Grey
    "Hardware Exception": "#9467bd",     # Purple
    "Invalid UART Payload": "#8c564b"    # Brown
}

TEST_RESULT_COLORS = {
    "Pass": "#2ca02c",
    "Fail": "#d62728"
}

for file in csv_files:
    basename = os.path.basename(file)
    print(f"[*] Processing {basename}...")
    
    # --- Parse Basename for Subheading ---
    lower_base = basename.lower()
    
    # 1. Check Phase
    if "ph1" in lower_base:
        phase_desc = ".BIT corruption using .LL"
    elif "ph2" in lower_base:
        phase_desc = ".BIT corruption using .EBD"
    elif "ph3" in lower_base:
        phase_desc = "Random corruption in .BIT"
    else:
        phase_desc = "Unknown Phase"
        
    # 2. Check Design
    if "neo_extbus_tpu" in lower_base:
        design_desc = "TinyTPU connected via AXI bus to NEORV32"
    elif "neo_cfs_tpu" in lower_base:
        design_desc = "TinyTPU integrated as CFS to NEORV32"
    else:
        design_desc = "Unknown Design"
        
    subtitle = f"{phase_desc} | {design_desc}"

    # Read the CSV
    df = pd.read_csv(file)
    
    # --- Data Prep: Subplot 1 (Test_Result) ---
    tr_counts = df['Test_Result'].value_counts().reset_index()
    tr_counts.columns = ['Test_Result', 'Count']
    tr_total = tr_counts['Count'].sum()
    
    tr_counts['Text'] = tr_counts['Count'].apply(
        lambda x: f"<b>{(x/tr_total)*100:.1f}%</b><br><span style='font-size: 10px'>({x})</span>"
    )
    
    # --- Data Prep: Subplot 2 (Result_Info for Failures Only) ---
    df_fail = df[df['Test_Result'] == 'Fail']
    ri_counts = df_fail['Result_Info'].value_counts().reset_index()
    ri_counts.columns = ['Result_Info', 'Count']
    
    # Sort Failures by occurrences (highest to lowest)
    ri_counts = ri_counts.sort_values(by='Count', ascending=False)
    ri_total = ri_counts['Count'].sum()
    
    if ri_total > 0:
        ri_counts['Text'] = ri_counts['Count'].apply(
            lambda x: f"<b>{(x/ri_total)*100:.1f}%</b><br><span style='font-size: 10px'>({x})</span>"
        )
    else:
        ri_counts['Text'] = []

    # --- Create the Subplot Grid ---
    fig = make_subplots(
        rows=1, cols=2, 
        column_widths=[SUBPLOT_1_WIDTH, SUBPLOT_2_WIDTH],
        subplot_titles=("Test_results", "Result_info for Fail")
    )
    
    # --- Populate Subplot 1 ---
    for _, row in tr_counts.iterrows():
        fig.add_trace(
            go.Bar(
                x=[row['Test_Result']],
                y=[row['Count']],
                name=row['Test_Result'],
                text=[row['Text']],
                textposition='auto',
                marker_color=TEST_RESULT_COLORS.get(row['Test_Result'], "#333333"),
                showlegend=False
            ),
            row=1, col=1
        )
        
    # --- Populate Subplot 2 ---
    # Iterates over the pre-sorted dataframe
    for _, row_data in ri_counts.iterrows():
        cat = row_data['Result_Info']
        fig.add_trace(
            go.Bar(
                x=[cat],
                y=[row_data['Count']],
                name=cat,
                text=[row_data['Text']],
                textposition='auto',
                marker_color=COLOR_MAP.get(cat, "#333333"),
                showlegend=False
            ),
            row=1, col=2
        )

    # --- Tweak Aesthetics & Layout ---
    # Use HTML tags to inject the parsed subtitle under the main title
    fig.update_layout(
            title=dict(
                text=(
                    f"<b>Fault Injection Campaign Results: {basename}</b>"
                    f"<br>"
                    f"<span style='font-size: 15px; color: #2E3A4B;'>{subtitle}</span>"
                ),
                x=0.5,
                xanchor='center'
            ),
            template="plotly_white",
            margin=dict(l=50, r=50, t=130, b=50)  # t=130 reserves space for the taller title
        )
        
    fig.update_yaxes(title_text="Occurrences", row=1, col=1)
    fig.update_xaxes(title_text="Result", row=1, col=1)
    
    fig.update_yaxes(title_text="Occurrences", row=1, col=2)
    fig.update_xaxes(title_text="Fault Category", row=1, col=2)
    
    # Show the interactive graph
    # fig.show()
    
    # --- Save the Output ---
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig.write_image(
        os.path.join(PLOT_DIR, f"plot_{os.path.splitext(basename)[0]}.png"),
        width=1280,
        height=720,
        scale=2
    )
    # fig.write_image(f"{PLOT_DIR}/plot_{os.path.splitext(basename)[0]}.png")
    print(f"  -> Saved plot_{os.path.splitext(basename)[0]}.png")