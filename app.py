from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np
import os
import ollama
import threading
import concurrent.futures
from fpdf import FPDF
import uuid
import time
import matplotlib
matplotlib.use('Agg')  # For servers without a display
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

app = Flask(__name__)

# Configuration
REPORTS_FOLDER = "reports"
ANALYSIS_CACHE = {}
os.makedirs(REPORTS_FOLDER, exist_ok=True)
app.config["REPORTS_FOLDER"] = REPORTS_FOLDER

# Database connection details
# IMPORTANT: update these to your correct credentials
DATABASE_URL = "postgresql://postgres:root@localhost:5432/RCA"
engine = create_engine(DATABASE_URL)

# Keywords that typically need root cause analysis
RCA_COLUMNS = {'slack', 'delay', 'violation', 'timing', 'setup', 'hold', 'clock_skew'}

# -----------------------------------------------------
# Utilities
# -----------------------------------------------------

def list_db_tables():
    query = text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.fetchall()
    return [row[0] for row in rows]

def load_table_as_dataframe(table_name):
    """
    Load an entire table from the DB into a Pandas DataFrame
    using the SQLAlchemy engine.
    """
    query = f'SELECT * FROM "{table_name}"'
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def is_relevant_column(col_name):
    """
    Check if the column is relevant for root cause analysis 
    by matching known keywords in its name.
    """
    return any(keyword in col_name.lower() for keyword in RCA_COLUMNS)

def analyze_column(col_name, delta_series):
    """
    Perform a more robust, structured root cause analysis on the delta data.
    """
    try:
        stats = delta_series.describe().to_dict()
        mean_val = stats.get("mean", 0)
        std_val = stats.get("std", 0)
        min_val = stats.get("min", 0)
        max_val = stats.get("max", 0)
        q1_val = stats.get('25%', 0)
        q2_val = stats.get('50%', 0)
        q3_val = stats.get('75%', 0)

        system_instructions = (
            "You are a highly advanced expert in timing analysis for digital IC design. "
            "Your goal is to identify any potential issues from these numeric deltas and "
            "explain the root cause thoroughly but concisely. Provide actionable solutions. "
            "Take into account that large positive or negative values might indicate design or timing shifts."
        )

        user_prompt = f"""
Column: {col_name}
Delta stats (File2 - File1):
  Mean: {mean_val:.3f}
  Std: {std_val:.3f}
  Min: {min_val:.3f}
  25%: {q1_val:.3f}
  50%: {q2_val:.3f}
  75%: {q3_val:.3f}
  Max: {max_val:.3f}

Provide a concise root cause analysis:
  - Problem: [short description or "None"]
  - Cause: [likely reason or multiple possibilities]
  - Solution: [recommended fixes or approaches]
  - Prevention: [how to avoid in future designs]
"""

        response = ollama.generate(
            model="mistral",
            prompt=f"[SYSTEM]\n{system_instructions}\n\n[USER]\n{user_prompt}",
            options={
                "temperature": 0.3,
                "max_tokens": 300
            }
        )

        return {
            'stats': stats,
            'analysis': response['response']
        }
    except Exception as e:
        return {'error': str(e)}

def generate_clustered_correlation(delta_df, relevant_cols):
    corr = delta_df[relevant_cols].corr()
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0)

    if corr.shape[0] < 2 or corr.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "Insufficient columns for clustering", 
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf

    g = sns.clustermap(
        corr, 
        method='ward',
        metric='euclidean',
        cmap='Spectral', 
        annot=True, 
        fmt=".2f"
    )
    g.fig.suptitle("Correlation Cluster Map (Delta Data)", y=1.02)
    plt.tight_layout()

    buf = BytesIO()
    g.fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(g.fig)
    return buf

def generate_distribution_plot(delta_series, col_name):
    """
    Generate a simple histogram plot for the delta values of a column.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(delta_series, kde=True, color='blue', ax=ax)
    ax.set_title(f"Delta Distribution for {col_name}")
    ax.set_xlabel("Delta (File2 - File1)")
    ax.set_ylabel("Count")
    ax.axvline(0, color='red', linestyle='--', alpha=0.8)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

def generate_report(affected_results, report_path, delta_df, relevant_cols, table1_name, table2_name):
    """
    Generate a PDF containing:
      - Page 1: Title + clustered correlation map
      - Subsequent pages: each affected column’s stats, histogram, & analysis
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Page 1: Title + cluster map
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Root Cause Analysis Report: {table2_name} vs {table1_name}", ln=True, align='C')
    pdf.ln(5)

    cluster_buf = generate_clustered_correlation(delta_df, relevant_cols)
    cluster_path = os.path.join(os.path.dirname(report_path), f"temp_cluster_{uuid.uuid4().hex}.png")
    with open(cluster_path, 'wb') as f:
        f.write(cluster_buf.read())
    cluster_buf.seek(0)
    pdf.image(cluster_path, x=15, w=180)
    pdf.ln(80)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Above is a clustered correlation map for the computed delta data ({table2_name} - {table1_name}).")
    if os.path.exists(cluster_path):
        os.remove(cluster_path)
    
    # Next pages: details for each affected column
    if not affected_results:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.multi_cell(0, 10, "No columns appear to require root cause analysis.")
    else:
        for col, data in affected_results.items():
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, txt=f"Column: {col} (Delta)", ln=True)
            pdf.set_font("Arial", size=12)

            if 'error' not in data:
                dist_buf = generate_distribution_plot(delta_df[col], col)
                dist_path = os.path.join(os.path.dirname(report_path), f"temp_dist_{uuid.uuid4().hex}.png")
                with open(dist_path, 'wb') as f:
                    f.write(dist_buf.read())
                dist_buf.seek(0)
                pdf.image(dist_path, x=30, w=150)
                pdf.ln(60)
                if os.path.exists(dist_path):
                    os.remove(dist_path)

            pdf.ln(5)
            if 'error' in data:
                pdf.multi_cell(0, 10, txt=f"Error: {data['error']}")
            else:
                stats = data['stats']
                analysis = data['analysis']
                pdf.cell(0, 10, txt="Key Delta Statistics:", ln=True)
                pdf.cell(0, 10, txt=f"Mean: {stats.get('mean', 'N/A'):.4f}", ln=True)
                pdf.cell(0, 10, txt=f"Std: {stats.get('std', 'N/A'):.4f}", ln=True)
                pdf.cell(0, 10, txt=f"Min: {stats.get('min', 'N/A'):.4f}", ln=True)
                pdf.cell(0, 10, txt=f"25%: {stats.get('25%', 'N/A'):.4f}", ln=True)
                pdf.cell(0, 10, txt=f"50%: {stats.get('50%', 'N/A'):.4f}", ln=True)
                pdf.cell(0, 10, txt=f"75%: {stats.get('75%', 'N/A'):.4f}", ln=True)
                pdf.cell(0, 10, txt=f"Max: {stats.get('max', 'N/A'):.4f}", ln=True)
                
                pdf.ln(5)
                pdf.multi_cell(0, 10, txt=f"Root Cause Analysis:\n{analysis}")

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.multi_cell(0, 10, txt="Thank you for using the Timing Data Delta Analysis!")
    
    pdf.output(report_path)

# -----------------------------------------------------
# Flask Routes
# -----------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/list_tables")
def list_tables_route():
    tables = list_db_tables()
    return jsonify({"tables": tables})

@app.route("/analyze", methods=["POST"])
def analyze_files():
    data = request.form
    table1_name = data.get("table1")
    table2_name = data.get("table2")

    if not table1_name or not table2_name:
        return jsonify({"error": "Two table names are required."})

    session_id = str(uuid.uuid4())
    pdf_filename = f"report_{table1_name}_vs_{table2_name}.pdf"
    report_path = os.path.join(app.config["REPORTS_FOLDER"], pdf_filename)

    try:
        df1 = load_table_as_dataframe(table1_name)
        df2 = load_table_as_dataframe(table2_name)

        common_cols = set(df1.columns).intersection(set(df2.columns))
        relevant_cols = []
        for col in common_cols:
            if is_relevant_column(col):
                if (pd.api.types.is_numeric_dtype(df1[col]) and
                    pd.api.types.is_numeric_dtype(df2[col])):
                    relevant_cols.append(col)

        if not relevant_cols:
            return jsonify({"error": "No common relevant numeric columns found for analysis."})

        delta_df = df2[relevant_cols] - df1[relevant_cols]

        ANALYSIS_CACHE[session_id] = {
            'status': 'processing',
            'total': len(relevant_cols),
            'completed': 0,
            'results': {},
            'affected': {},
            'relevant_cols': relevant_cols,
            'report_path': report_path,
            'table1_name': table1_name,
            'table2_name': table2_name
        }

        def process_columns():
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(analyze_column, col, delta_df[col]): col
                           for col in relevant_cols}
                for future in concurrent.futures.as_completed(futures):
                    col = futures[future]
                    try:
                        result = future.result(timeout=60)
                        ANALYSIS_CACHE[session_id]['results'][col] = result
                    except Exception as e:
                        ANALYSIS_CACHE[session_id]['results'][col] = {'error': str(e)}
                    ANALYSIS_CACHE[session_id]['completed'] += 1
            
            final_affected = {}
            for col, data in ANALYSIS_CACHE[session_id]['results'].items():
                if 'analysis' in data and not data.get('error'):
                    lines = data['analysis'].split('\n')
                    problem_line = next((ln for ln in lines if ln.lower().startswith("problem:")), "")
                    if "none" in problem_line.lower():
                        continue
                    final_affected[col] = data
                else:
                    final_affected[col] = data

            ANALYSIS_CACHE[session_id]['affected'] = final_affected
            generate_report(
                final_affected,
                ANALYSIS_CACHE[session_id]['report_path'],
                delta_df,
                relevant_cols,
                table1_name=ANALYSIS_CACHE[session_id]['table1_name'],
                table2_name=ANALYSIS_CACHE[session_id]['table2_name']
            )
            ANALYSIS_CACHE[session_id]['status'] = 'complete'

        threading.Thread(target=process_columns).start()
        
        return jsonify({
            "session_id": session_id,
            "total_columns": len(relevant_cols),
            "message": "Analysis started successfully."
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/status/<session_id>")
def get_status(session_id):
    status = ANALYSIS_CACHE.get(session_id, {"error": "Invalid session ID"})
    if 'error' in status:
        return jsonify({"error": status['error']})

    out = {
        "status": status.get('status', 'unknown'),
        "total_columns": status.get('total', 0),
        "completed": status.get('completed', 0),
        "results": []
    }
    
    all_results = status.get('results', {})
    
    for col, data in all_results.items():
        if 'analysis' in data and not data.get('error'):
            # Include stats + entire analysis
            stats = data['stats']
            analysis_text = data['analysis']
            
            # Construct a multiline string that mirrors the PDF text
            col_text = (
                f"Column: {col}\n"
                f"Mean: {stats.get('mean', 'N/A'):.4f}\n"
                f"Std: {stats.get('std', 'N/A'):.4f}\n"
                f"Min: {stats.get('min', 'N/A'):.4f}\n"
                f"25%: {stats.get('25%', 'N/A'):.4f}\n"
                f"50%: {stats.get('50%', 'N/A'):.4f}\n"
                f"75%: {stats.get('75%', 'N/A'):.4f}\n"
                f"Max: {stats.get('max', 'N/A'):.4f}\n\n"
                f"Root Cause Analysis:\n{analysis_text}"
            )
            
            out["results"].append(col_text)
        
        elif 'error' in data:
            out["results"].append(f"{col}: Error - {data['error']}")
        else:
            out["results"].append(f"{col}: Processing...")
    
    return jsonify(out)

@app.route("/report/<session_id>")
def get_report(session_id):
    data = ANALYSIS_CACHE.get(session_id)
    if not data:
        return jsonify({"status": "Invalid session ID"})
    
    report_path = data.get('report_path')
    if report_path and os.path.exists(report_path):
        return send_file(report_path, as_attachment=True)
    return jsonify({"status": "analysis ongoing"})

if __name__ == "__main__":
    app.run(debug=False, threaded=True)