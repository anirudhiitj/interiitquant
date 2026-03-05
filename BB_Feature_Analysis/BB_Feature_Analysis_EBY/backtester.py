import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "/data/quant14/EBY"
OUTPUT_DIR = "bb_metrics_html"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_daily_data(file_path):
    """Load a single day's data from CSV"""
    try:
        df = pd.read_csv(file_path)
        
        # Column 1 is Time, Column 2 is Price
        if len(df.columns) < 2:
            print(f"Not enough columns in {file_path}")
            return None
        
        # Get column names
        time_col = df.columns[0]
        price_col = df.columns[1]
        
        # Rename for consistency
        df.rename(columns={time_col: 'Time', price_col: 'Price'}, inplace=True)
        
        # Handle time column
        try:
            df['Time'] = pd.to_timedelta(df['Time'])
            df['TimeStr'] = df['Time'].apply(lambda x: str(x).split(' ')[-1] if pd.notna(x) else '')
        except:
            try:
                df['Time'] = pd.to_datetime(df['Time'])
                df['TimeStr'] = df['Time'].dt.strftime('%H:%M:%S')
            except:
                # Create sequential time
                df['TimeStr'] = [f"{i//3600:02d}:{(i%3600)//60:02d}:{i%60:02d}" for i in range(len(df))]
        
        return df
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_bb_columns(df):
    """Get all columns starting with 'BB'"""
    bb_cols = [col for col in df.columns if col.startswith('BB') and col not in ['Time', 'Price', 'TimeStr']]
    return sorted(bb_cols)

def create_interactive_html_plot(df, day_num, bb_columns, output_path, auto_open=True):
    """
    Create interactive HTML plot with dual y-axes and dropdown to switch metrics:
    - Primary Y-axis: Price
    - Secondary Y-axis: Selected BB Metric
    """
    
    # Drop rows with NaN prices
    valid_idx = df['Price'].notna()
    df_clean = df[valid_idx].copy()
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add Price trace (primary y-axis) - always visible
    fig.add_trace(
        go.Scatter(
            x=df_clean['TimeStr'],
            y=df_clean['Price'],
            name='Price',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>Price</b><br>Time: %{x}<br>Value: $%{y:.2f}<extra></extra>',
            visible=True
        ),
        secondary_y=False
    )
    
    # Color palette for BB metrics
    colors = [
        '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8',
        '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
        '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5', '#ff9896'
    ]
    
    # Add BB metric traces (secondary y-axis) - initially only first visible
    for idx, bb_col in enumerate(bb_columns):
        if bb_col in df_clean.columns:
            color = colors[idx % len(colors)]
            
            # Filter non-NaN values for this metric
            metric_valid = df_clean[bb_col].notna()
            
            if metric_valid.sum() > 0:
                fig.add_trace(
                    go.Scatter(
                        x=df_clean.loc[metric_valid, 'TimeStr'],
                        y=df_clean.loc[metric_valid, bb_col],
                        name=bb_col,
                        line=dict(color=color, width=1.5),
                        opacity=0.8,
                        hovertemplate=f'<b>{bb_col}</b><br>Time: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>',
                        visible=(idx == 0)  # Only first metric visible initially
                    ),
                    secondary_y=True
                )
    
    # Create dropdown buttons for metric selection
    buttons = []
    for idx, bb_col in enumerate(bb_columns):
        # Create visibility list: Price is always True, only one BB metric is True
        visibility = [True]  # Price trace
        for i in range(len(bb_columns)):
            visibility.append(i == idx)
        
        button = dict(
            label=bb_col,
            method="update",
            args=[
                {"visible": visibility},
                {"yaxis2.title.text": bb_col}
            ]
        )
        buttons.append(button)
    
    # Update layout with dropdown menu
    fig.update_layout(
        title=dict(
            text=f'<b>Day {day_num}: Price and BB Metrics Analysis</b><br><sub>Use dropdown to switch between metrics</sub>',
            font=dict(size=20, color='black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text='Time (HH:MM:SS)', font=dict(size=14)),
            tickfont=dict(size=11),
            gridcolor='lightgray',
            gridwidth=0.5,
            tickmode='auto',
            nticks=20
        ),
        yaxis=dict(
            title=dict(text='Price ($)', font=dict(size=14, color='#1f77b4')),
            tickfont=dict(size=12, color='#1f77b4'),
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        yaxis2=dict(
            title=dict(text=bb_columns[0] if bb_columns else 'BB Metric Values', 
                      font=dict(size=14, color='#ff7f0e')),
            tickfont=dict(size=12, color='#ff7f0e')
        ),
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.15,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="black",
                borderwidth=2,
                font=dict(size=11)
            )
        ],
        hovermode='x unified',
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.98,
            xanchor='left',
            x=1.15,
            font=dict(size=10),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1600,
        height=800,
        margin=dict(l=80, r=300, t=120, b=80)
    )
    
    # Add statistics annotation
    stats_text = (
        f'<b>Statistics for Day {day_num}</b><br>'
        f'Data Points: {len(df_clean):,}<br>'
        f'Initial Price: ${df_clean["Price"].iloc[0]:.2f}<br>'
        f'Final Price: ${df_clean["Price"].iloc[-1]:.2f}<br>'
        f'Min Price: ${df_clean["Price"].min():.2f}<br>'
        f'Max Price: ${df_clean["Price"].max():.2f}<br>'
        f'Price Range: ${df_clean["Price"].max() - df_clean["Price"].min():.2f}<br>'
        f'BB Metrics Available: {len(bb_columns)}'
    )
    
    fig.add_annotation(
        text=stats_text,
        xref='paper', yref='paper',
        x=0.02, y=0.98,
        xanchor='left', yanchor='top',
        showarrow=False,
        bgcolor='rgba(255, 248, 220, 0.95)',
        bordercolor='black',
        borderwidth=2,
        borderpad=10,
        font=dict(size=11, family='monospace', color='black')
    )
    
    # Save to HTML
    fig.write_html(
        output_path,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'day{day_num}_bb_metrics',
                'height': 800,
                'width': 1600,
                'scale': 2
            }
        }
    )
    
    # Auto-open in browser (only if not in SSH session)
    if auto_open:
        try:
            import webbrowser
            # Check if we're in SSH or headless environment
            import os
            if os.environ.get('SSH_CONNECTION') or os.environ.get('SSH_CLIENT'):
                # Skip auto-open in SSH
                pass
            else:
                webbrowser.open('file://' + os.path.abspath(output_path))
        except:
            # If any error, just skip opening
            pass

# ============================================================================
# MAIN ANALYSIS LOOP
# ============================================================================

def run_bb_metrics_analysis():
    """Generate interactive HTML plots for all days (day0.csv to day278.csv)"""
    
    print("\n" + "="*100)
    print("STARTING BB METRICS VISUALIZATION - INTERACTIVE HTML WITH DROPDOWN")
    print("="*100)
    print(f"Data Path: {DATA_PATH}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("="*100 + "\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all data files (day0.csv to day278.csv)
    all_files = []
    for day in range(279):
        file_path = os.path.join(DATA_PATH, f"day{day}.csv")
        if os.path.exists(file_path):
            all_files.append((day, file_path))
        else:
            print(f"WARNING: {file_path} not found")
    
    if len(all_files) == 0:
        print(f"ERROR: No data files found in {DATA_PATH}")
        print(f"Expected files: day0.csv, day1.csv, ..., day278.csv")
        return
    
    print(f"Found {len(all_files)}/279 data files")
    print(f"First file: day{all_files[0][0]}.csv")
    print(f"Last file: day{all_files[-1][0]}.csv")
    print()
    
    # Process each day
    summary_data = []
    
    for day_num, file_path in all_files:
        file_name = os.path.basename(file_path)
        
        print(f"[Day {day_num:3d}/278] Processing: {file_name} ", end='', flush=True)
        
        # Load data
        df = load_daily_data(file_path)
        
        if df is None or len(df) == 0:
            print(f"❌ SKIPPED: Failed to load data")
            continue
        
        # Get BB columns
        bb_columns = get_bb_columns(df)
        
        if len(bb_columns) == 0:
            print(f"❌ SKIPPED: No BB columns found")
            continue
        
        print(f"✓")
        print(f"      📊 Data points: {len(df):,}")
        print(f"      💰 Price: ${df['Price'].min():.2f} - ${df['Price'].max():.2f}")
        print(f"      📈 BB metrics: {len(bb_columns)}")
        
        # Show progress bar
        progress = (len(summary_data) + 1) / len(all_files) * 100
        bar_length = 40
        filled = int(bar_length * progress / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"      Progress: [{bar}] {progress:.1f}%")
        
        # Create interactive plot
        output_path = os.path.join(OUTPUT_DIR, f"day{day_num}.html")
        
        try:
            create_interactive_html_plot(df, day_num, bb_columns, output_path, auto_open=False)
            print(f"      ✅ HTML saved: {os.path.basename(output_path)}")
            
            # Store summary
            summary_data.append({
                'Day': day_num,
                'File': file_name,
                'Data_Points': len(df),
                'BB_Metrics': len(bb_columns),
                'Min_Price': df['Price'].min(),
                'Max_Price': df['Price'].max(),
                'Price_Range': df['Price'].max() - df['Price'].min(),
                'HTML_File': f"day{day_num}.html"
            })
            
        except Exception as e:
            print(f"      ❌ ERROR: {e}")
        
        print()
    
    # Create summary CSV
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(OUTPUT_DIR, 'bb_metrics_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n📊 Summary saved to: {summary_path}")
    
    # Create index HTML file
    create_index_html(summary_data)
    
    # Create combined view HTML
    create_combined_view_html(summary_data)
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print(f"✓ Total HTML files generated: {len(summary_data)}")
    print(f"✓ All files saved in: {OUTPUT_DIR}/")
    print(f"✓ Open 'index.html' to navigate all 279 days")
    print(f"✓ Open 'combined_view.html' to see all days in one page")
    print("="*100 + "\n")

def create_index_html(summary_data):
    """Create an index.html file to navigate all plots"""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BB Metrics Analysis - All 279 Days</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                text-align: center;
                color: #666;
                font-size: 1.1em;
                margin-bottom: 30px;
            }
            .stats {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
            }
            .stat-box {
                background-color: rgba(255, 255, 255, 0.1);
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }
            .stat-box h3 {
                margin: 0 0 10px 0;
                font-size: 1em;
                font-weight: normal;
                opacity: 0.9;
            }
            .stat-box .value {
                font-size: 2em;
                font-weight: bold;
            }
            .search-box {
                margin-bottom: 20px;
                text-align: center;
            }
            .search-box input {
                padding: 12px 20px;
                width: 300px;
                border: 2px solid #667eea;
                border-radius: 25px;
                font-size: 16px;
                outline: none;
            }
            .search-box input:focus {
                border-color: #764ba2;
                box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            th, td {
                padding: 15px;
                text-align: left;
                border-bottom: 1px solid #e0e0e0;
            }
            th {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-weight: bold;
                position: sticky;
                top: 0;
                z-index: 10;
            }
            tr:hover {
                background-color: #f8f9ff;
                transform: scale(1.01);
                transition: all 0.2s;
            }
            a {
                color: #667eea;
                text-decoration: none;
                font-weight: bold;
                padding: 8px 15px;
                border-radius: 5px;
                background-color: #f0f0ff;
                display: inline-block;
                transition: all 0.3s;
            }
            a:hover {
                background-color: #667eea;
                color: white;
                transform: translateX(5px);
            }
            .badge {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 12px;
                background-color: #e8f4f8;
                color: #1f77b4;
                font-size: 0.9em;
                font-weight: bold;
            }
        </style>
        <script>
            function searchTable() {
                var input = document.getElementById("searchInput");
                var filter = input.value.toUpperCase();
                var table = document.getElementById("dataTable");
                var tr = table.getElementsByTagName("tr");
                
                for (var i = 1; i < tr.length; i++) {
                    var td = tr[i].getElementsByTagName("td")[0];
                    if (td) {
                        var txtValue = td.textContent || td.innerText;
                        if (txtValue.toUpperCase().indexOf(filter) > -1) {
                            tr[i].style.display = "";
                        } else {
                            tr[i].style.display = "none";
                        }
                    }
                }
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>📊 BB Metrics Analysis Dashboard</h1>
            <div class="subtitle">Interactive visualization of 279 trading days with switchable BB metrics</div>
            
            <div class="stats">
                <div class="stat-box">
                    <h3>Total Days</h3>
                    <div class="value">{total_days}</div>
                </div>
                <div class="stat-box">
                    <h3>Total Data Points</h3>
                    <div class="value">{total_points:,}</div>
                </div>
                <div class="stat-box">
                    <h3>Avg BB Metrics/Day</h3>
                    <div class="value">{avg_metrics:.1f}</div>
                </div>
                <div class="stat-box">
                    <h3>Total HTML Files</h3>
                    <div class="value">{total_days}</div>
                </div>
            </div>
            
            <div class="search-box">
                <input type="text" id="searchInput" onkeyup="searchTable()" placeholder="Search by Day number...">
            </div>
            
            <table id="dataTable">
                <thead>
                    <tr>
                        <th>Day</th>
                        <th>Data Points</th>
                        <th>BB Metrics</th>
                        <th>Price Range</th>
                        <th>Interactive Plot</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Calculate summary stats
    total_points = sum(d['Data_Points'] for d in summary_data)
    avg_metrics = sum(d['BB_Metrics'] for d in summary_data) / len(summary_data) if summary_data else 0
    
    # Add rows for each day
    for data in summary_data:
        html_content += f"""
                    <tr>
                        <td><strong>Day {data['Day']}</strong></td>
                        <td>{data['Data_Points']:,}</td>
                        <td><span class="badge">{data['BB_Metrics']} metrics</span></td>
                        <td>${data['Min_Price']:.2f} - ${data['Max_Price']:.2f}</td>
                        <td><a href="{data['HTML_File']}" target="_blank">📈 View Chart</a></td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    
    # Format with summary stats
    html_content = html_content.format(
        total_days=len(summary_data),
        total_points=total_points,
        avg_metrics=avg_metrics
    )
    
    # Save index file
    index_path = os.path.join(OUTPUT_DIR, 'index.html')
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    print(f"Index file created: {index_path}")

def create_combined_view_html(summary_data):
    """Create a combined view showing all 279 days with iframe grid"""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BB Metrics - Combined View (All 279 Days)</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .container {
                max-width: 98%;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                text-align: center;
                color: #666;
                font-size: 1.1em;
                margin-bottom: 30px;
            }
            .controls {
                text-align: center;
                margin-bottom: 20px;
                padding: 20px;
                background-color: #f8f9ff;
                border-radius: 10px;
            }
            .controls label {
                margin: 0 10px;
                font-weight: bold;
            }
            .controls select, .controls input {
                padding: 8px 15px;
                border: 2px solid #667eea;
                border-radius: 5px;
                font-size: 14px;
                margin: 0 10px;
            }
            .controls button {
                padding: 10px 25px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                cursor: pointer;
                margin: 0 5px;
            }
            .controls button:hover {
                transform: scale(1.05);
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            }
            .grid-container {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(600px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            .chart-frame {
                border: 3px solid #667eea;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                background-color: white;
            }
            .chart-title {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 10px;
                text-align: center;
                font-weight: bold;
                font-size: 1.1em;
            }
            iframe {
                width: 100%;
                height: 500px;
                border: none;
            }
            .loading {
                text-align: center;
                padding: 50px;
                font-size: 1.2em;
                color: #667eea;
            }
        </style>
        <script>
            function changeColumns() {
                var cols = document.getElementById('columnsSelect').value;
                var container = document.getElementById('gridContainer');
                container.style.gridTemplateColumns = `repeat(auto-fill, minmax(${600/cols}px, 1fr))`;
            }
            
            function filterDays() {
                var startDay = parseInt(document.getElementById('startDay').value) || 0;
                var endDay = parseInt(document.getElementById('endDay').value) || 278;
                
                var frames = document.getElementsByClassName('chart-frame');
                for (var i = 0; i < frames.length; i++) {
                    var dayNum = parseInt(frames[i].getAttribute('data-day'));
                    if (dayNum >= startDay && dayNum <= endDay) {
                        frames[i].style.display = 'block';
                    } else {
                        frames[i].style.display = 'none';
                    }
                }
            }
            
            function showAll() {
                var frames = document.getElementsByClassName('chart-frame');
                for (var i = 0; i < frames.length; i++) {
                    frames[i].style.display = 'block';
                }
                document.getElementById('startDay').value = 0;
                document.getElementById('endDay').value = 278;
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>📊 BB Metrics - Combined View (All 279 Days)</h1>
            <div class="subtitle">View multiple days simultaneously - scroll to see all charts</div>
            
            <div class="controls">
                <label>Columns:</label>
                <select id="columnsSelect" onchange="changeColumns()">
                    <option value="1">1 Column</option>
                    <option value="2" selected>2 Columns</option>
                    <option value="3">3 Columns</option>
                    <option value="4">4 Columns</option>
                </select>
                
                <label>Show Days:</label>
                <input type="number" id="startDay" placeholder="From" min="0" max="278" value="0" style="width: 80px;">
                <input type="number" id="endDay" placeholder="To" min="0" max="278" value="278" style="width: 80px;">
                <button onclick="filterDays()">Apply Filter</button>
                <button onclick="showAll()">Show All</button>
            </div>
            
            <div class="grid-container" id="gridContainer">
    """
    
    # Add iframe for each day
    for data in summary_data:
        html_content += f"""
                <div class="chart-frame" data-day="{data['Day']}">
                    <div class="chart-title">Day {data['Day']} - {data['BB_Metrics']} BB Metrics</div>
                    <iframe src="{data['HTML_File']}" loading="lazy"></iframe>
                </div>
        """
    
    html_content += """
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save combined view file
    combined_path = os.path.join(OUTPUT_DIR, 'combined_view.html')
    with open(combined_path, 'w') as f:
        f.write(html_content)
    
    print(f"Combined view created: {combined_path}")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_bb_metrics_analysis()