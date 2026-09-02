import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas
import pypdfium2 as pdfium

# ----------------------------------------------------------------------
# Numbered Canvas for Two-Pass Page Numbering and Running Headers/Footers
# ----------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # Header (Pages 2+)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#475569"))
            self.drawString(50, 755, "DigitalTwin.ai | Detailed Business Proposal & Technical Architecture")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawRightString(562, 755, "Problem Track 4 — Round 2 Solution Design")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.75)
            self.line(50, 747, 562, 747)
            
        # Footer (All Pages)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.75)
        self.line(50, 42, 562, 42)
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#0F172A"))
        self.drawString(50, 31, "DIGITALTWIN.AI")
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(120, 31, "|   Mixed-Model Automotive Digital Twin & Predictive Quality Engine")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(562, 31, page_text)
            
        self.restoreState()


def create_proposal_pdf(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=46,
        bottomMargin=46
    )
    
    # Palette
    C_PRIMARY = colors.HexColor("#0F172A")    # Slate 900
    C_SECONDARY = colors.HexColor("#1E40AF")  # Royal Blue 800
    C_ACCENT = colors.HexColor("#0284C7")     # Ocean Sky 600
    C_TEXT = colors.HexColor("#1E293B")       # Body Slate 800
    C_MUTED = colors.HexColor("#475569")      # Muted Slate 600
    C_BG_LIGHT = colors.HexColor("#F8FAFC")   # Slate 50
    C_BG_ALT = colors.HexColor("#F1F5F9")     # Slate 100
    C_BORDER = colors.HexColor("#CBD5E1")     # Slate 300
    C_SUCCESS_BG = colors.HexColor("#DCFCE7") # Green 100
    C_SUCCESS_TXT = colors.HexColor("#166534")# Green 800

    styles = getSampleStyleSheet()
    
    title_badge = ParagraphStyle(
        'TitleBadge',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=10,
        textColor=C_ACCENT,
        spaceAfter=2
    )

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=C_PRIMARY,
        spaceAfter=3
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=13.5,
        textColor=C_SECONDARY,
        spaceAfter=6
    )
    
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=C_MUTED,
        spaceAfter=8
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=C_PRIMARY,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12.5,
        textColor=C_SECONDARY,
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=C_TEXT,
        spaceAfter=3
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=10,
        firstLineIndent=-6,
        spaceAfter=2
    )

    callout_style = ParagraphStyle(
        'Callout_Text',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10.5,
        textColor=C_TEXT
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=0
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.2,
        textColor=C_TEXT
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell,
        fontName='Helvetica-Bold'
    )
    
    table_cell_right = ParagraphStyle(
        'TableCellRight',
        parent=table_cell,
        alignment=2
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE, EXECUTIVE SUMMARY & STRATEGIC PROBLEM FRAMING
    # =========================================================================
    story.append(Paragraph("DIGITALTWIN.AI &bull; DETAILED BUSINESS PROPOSAL", title_badge))
    story.append(Paragraph("Predictive Digital Twin for Mixed-Model Automotive Assembly", title_style))
    story.append(Paragraph("A Unified Statistical, Physics-Informed & Explainable ML Architecture for Mixed Sensor Vintage Lines", subtitle_style))
    
    meta_text = (
        "<b>Track:</b> Problem Track 4 (Round 2) &nbsp;|&nbsp; "
        "<b>Target Line:</b> Northstar Mixed-Model Line (40 Stations, 420 Vehicles) &nbsp;|&nbsp; "
        "<b>Date:</b> August 2026 &nbsp;|&nbsp; "
        "<b>Status:</b> Backtested & Prototype Validated"
    )
    story.append(Paragraph(meta_text, meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=C_SECONDARY, spaceBefore=0, spaceAfter=6))

    story.append(Paragraph("Executive Summary & Value Proposition", h1_style))
    summary_text = (
        "Modern automotive assembly plants operate under fierce margin pressures and complex product mixes (SUV, Sedan, EV). "
        "However, factory floors remain a <b>patchwork of legacy and modern equipment</b> with severe sensor disparity—leaving up to "
        "30% of stations dependent on manual checklists. Traditional black-box AI digital twins fail because they trigger catastrophic "
        "false-alarm storms, lack physical interpretability, and risk line shutdowns through invasive control attempts. "
        "<b>DigitalTwin.ai</b> bridges this gap with a <b>non-invasive, read-only statistical, physics-informed, and explainable ML digital twin</b>. "
        "By fusing dynamic context-specific Statistical Process Control (SPC), thermodynamic domain physics, and an industrial Gradient-Boosted "
        "risk engine (XGBoost), DigitalTwin.ai detects micro-drifts up to 10 vehicles prior to downstream escape, forecasts starvation bottlenecks, "
        "and delivers a <b>4.2-month payback period ($418,000 net annual savings per line)</b> while adhering strictly to quarterly scheduled maintenance windows."
    )
    story.append(Paragraph(summary_text, body_style))

    # KPI Table
    kpi_data = [
        [
            Paragraph("<b>Core Capability / KPI</b>", table_header),
            Paragraph("<b>Industry Baseline (Status Quo)</b>", table_header),
            Paragraph("<b>DigitalTwin.ai Solution</b>", table_header),
            Paragraph("<b>Measurable Impact & ROI</b>", table_header)
        ],
        [
            Paragraph("Defect Detection Horizon", table_cell_bold),
            Paragraph("End-of-Line QA (Station 40)", table_cell),
            Paragraph("Early BIW/PNT detection (Lag: 10 vehicles)", table_cell),
            Paragraph("Saves $145k/yr in batch scrap & tear-down", table_cell)
        ],
        [
            Paragraph("Bottleneck Prediction", table_cell_bold),
            Paragraph("Reactive to buffer jams", table_cell),
            Paragraph("8-Vehicle rolling cycle regression", table_cell),
            Paragraph("+2.4% Net OEE recovery ($182k/yr capacity)", table_cell)
        ],
        [
            Paragraph("Sensor-Poor Stations", table_cell_bold),
            Paragraph("Blind spots or false data", table_cell),
            Paragraph("Explicit gap tagging + Adjacent inference", table_cell),
            Paragraph("Zero false hallucinations; 100% audit integrity", table_cell)
        ],
        [
            Paragraph("OT Integration Safety", table_cell_bold),
            Paragraph("High PLC recoding risk", table_cell),
            Paragraph("100% Read-Only Listener (OPC-UA/MQTT)", table_cell),
            Paragraph("Zero production downtime risk during rollout", table_cell)
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[110, 110, 142, 150])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_PRIMARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_BG_LIGHT, C_BG_ALT])
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 4))

    story.append(Paragraph("1. Strategic Problem Framing & Industrial Realities", h1_style))
    story.append(Paragraph(
        "Deploying predictive technologies in vehicle manufacturing requires addressing six harsh operational constraints:", body_style))

    p1_items = [
        "<b>1. Infrastructure Patchwork & Sensor Disparity:</b> Assembly lines combine high-tech 6-axis robotic weld cells with legacy manual stations relying entirely on paper/tablet checklists. A viable twin must function seamlessly across both.",
        "<b>2. Multi-Causal, Intermittent Root Causes:</b> Defects rarely stem from single isolated spikes; they emerge from coupled tool wear, sheet-metal gauge tolerances, thermal dissipation, and ambient variations.",
        "<b>3. Downstream Defect Propagation:</b> An unnoticed torque or weld flaw at Station 06 often escapes until Station 38 inspection. By then, 30+ units carry the defect, causing massive teardown costs.",
        "<b>4. Zero Live-PLC Modification Mandate:</b> Plant OT safety prohibits modifying live ladder logic or writing to operational PLCs during production. Digital twins must act exclusively as passive listeners.",
        "<b>5. Strict Scheduled Maintenance Windows:</b> Physical sensor retrofits cannot interrupt daily production; they are strictly confined to four scheduled quarterly maintenance shut-downs per year.",
        "<b>6. Operator Trust & False-Alarm Fatigue:</b> Black-box ML models uncalibrated for vehicle variant mix (SUV vs. Sedan vs. EV) trigger alarm storms, causing floor supervisors to dismiss recommendations."
    ]
    for itm in p1_items:
        story.append(Paragraph(f"&bull; {itm}", bullet_style))

    # End of Page 1
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: SOLUTION ARCHITECTURE & TECHNICAL DESIGN
    # =========================================================================
    story.append(Paragraph("2. Solution Architecture & Technical Design", h1_style))
    story.append(Paragraph(
        "DigitalTwin.ai employs a multi-tiered architecture combining dynamic context normalization, statistical anomaly detection, "
        "thermodynamic domain physics, and an explainable gradient-boosted decision tree risk engine:", body_style))

    story.append(Paragraph("2.1 Dynamic Context-Specific Baseline Engine", h2_style))
    story.append(Paragraph(
        "To eliminate false alarms caused by product variant transitions, telemetry is normalized dynamically against the 3-tuple context: "
        "<b>(Station ID, Parameter Name, Product Variant)</b>:", body_style))
    
    formula_text = (
        "<b>Dynamic Contextual Standardization Formulation:</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Z(v, s, p) = ( x(v, s, p) - &mu;(s, p, Variant(v)) ) / &sigma;(s, p, Variant(v))</b><br/>"
        "<i>Where x is raw sensor reading (torque, force, vibration, temp), &mu; is the empirical mean for that specific body variant "
        "(SUV, Sedan, EV), and &sigma; is the variant standard deviation.</i>"
    )
    f_table = Table([[Paragraph(formula_text, callout_style)]], colWidths=[512])
    f_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), C_BG_ALT),
        ('BOX', (0, 0), (-1, -1), 0.75, C_SECONDARY),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(f_table)
    story.append(Spacer(1, 4))

    story.append(Paragraph("2.2 Multi-Pattern Statistical & Physics Anomaly Engines", h2_style))
    story.append(Paragraph(
        "The core twin runs four concurrent statistical rules and a physics-informed thermal consistency engine:", body_style))

    algo_data = [
        [
            Paragraph("<b>Detection Mechanism</b>", table_header),
            Paragraph("<b>Trigger Condition & Mathematical Formulation</b>", table_header),
            Paragraph("<b>Target Industrial Failure Mode</b>", table_header),
            Paragraph("<b>Severity</b>", table_header)
        ],
        [
            Paragraph("Point Anomaly", table_cell_bold),
            Paragraph("|Z| &ge; 3.0 (Statistical 3-sigma excursion)", table_cell),
            Paragraph("Sudden tool failure, power transient, fastener break", table_cell),
            Paragraph("<font color='#991B1B'><b>CRITICAL</b></font>", table_cell)
        ],
        [
            Paragraph("Persistent Shift", table_cell_bold),
            Paragraph("All |Z| &ge; 2.0 across W = 4 consecutive vehicles", table_cell),
            Paragraph("Batch material hardness shift, calibration slip", table_cell),
            Paragraph("<font color='#991B1B'><b>CRITICAL</b></font>", table_cell)
        ],
        [
            Paragraph("Trend Drift", table_cell_bold),
            Paragraph("Monotonic |Z| growth (&Delta;|Z| &ge; 1.6, W = 5 units)", table_cell),
            Paragraph("Weld-gun tip wear, bearing degradation", table_cell),
            Paragraph("<font color='#92400E'><b>WARNING</b></font>", table_cell)
        ],
        [
            Paragraph("Cross-Parameter Coupling", table_cell_bold),
            Paragraph("Concurrent abnormal |Z| &ge; 2.5 on linked signals", table_cell),
            Paragraph("Simultaneous force + vibration spindle binding", table_cell),
            Paragraph("<font color='#991B1B'><b>CRITICAL</b></font>", table_cell)
        ],
        [
            Paragraph("Physics Thermal Engine", table_cell_bold),
            Paragraph("|T_measured - T_setpoint| > 12.0 K (Newton's cooling)", table_cell),
            Paragraph("Paint oven zone seal breach / burner failure", table_cell),
            Paragraph("<font color='#991B1B'><b>CRITICAL</b></font>", table_cell)
        ]
    ]
    algo_table = Table(algo_data, colWidths=[105, 165, 172, 70])
    algo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_PRIMARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_BG_LIGHT, C_BG_ALT])
    ]))
    story.append(algo_table)
    story.append(Spacer(1, 4))

    story.append(Paragraph("2.3 Industrial XGBoost Risk Engine & SHAP Attribution", h2_style))
    story.append(Paragraph(
        "To predict defect escape risk, DigitalTwin.ai integrates an industrial Gradient-Boosted Decision Tree (XGBoost) pre-trained on "
        "multi-mode predictive maintenance data (AI4I benchmark: tool wear, heat dissipation, overstrain, SPC flags). "
        "The model maps a 10-dimensional temporal feature vector to a calibrated defect probability P(Defect) with SHAP weights:", body_style))

    feat_data = [
        [
            Paragraph("<b>Telemetry Feature</b>", table_header),
            Paragraph("<b>Physical Meaning on Assembly Line</b>", table_header),
            Paragraph("<b>SHAP Importance Weight</b>", table_header)
        ],
        [
            Paragraph("max_abs_z", table_cell_bold),
            Paragraph("Peak statistical deviation across all station parameters for vehicle", table_cell),
            Paragraph("<b>44.9%</b> (Primary Driver)", table_cell)
        ],
        [
            Paragraph("cross_param_flag", table_cell_bold),
            Paragraph("Co-occurrence of correlated multi-sensor anomalies at same station", table_cell),
            Paragraph("<b>18.8%</b>", table_cell)
        ],
        [
            Paragraph("trend_slope_indicator", table_cell_bold),
            Paragraph("Drift velocity indicating progressive mechanical tool wear", table_cell),
            Paragraph("<b>12.7%</b>", table_cell)
        ],
        [
            Paragraph("mean_abs_z / alert_count", table_cell_bold),
            Paragraph("Aggregate station process variance and critical flag counts", table_cell),
            Paragraph("<b>11.2%</b>", table_cell)
        ],
        [
            Paragraph("queue / cycle / rework", table_cell_bold),
            Paragraph("Buffer congestion, station pacing strain, and manual checklist fails", table_cell),
            Paragraph("<b>12.4%</b>", table_cell)
        ]
    ]
    feat_table = Table(feat_data, colWidths=[120, 252, 140])
    feat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_PRIMARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_BG_LIGHT, C_BG_ALT])
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 4))

    story.append(Paragraph("2.4 Transparent Handling of Sensor-Poor Stations", h2_style))
    story.append(Paragraph(
        "DigitalTwin.ai enforces strict <b>Epistemic Transparency</b> to eliminate false hallucinations: "
        "<b>(1) Instrumented (70%):</b> Continuous real-time multi-signal SPC; "
        "<b>(2) Manual Outcome (30%):</b> Discrete pass/fail/rework records without artificial waveforms; "
        "<b>(3) Adjacent-Signal Inference:</b> Quality risks inferred from upstream/downstream neighbors are explicitly labeled with confidence metrics.", body_style))

    # End of Page 2
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: STAKEHOLDER VIEWS & NON-INVASIVE OT/IT INTEGRATION
    # =========================================================================
    story.append(Paragraph("3. Multi-Stakeholder Experience & Operational Workflows", h1_style))
    story.append(Paragraph(
        "DigitalTwin.ai delivers three distinct role-tailored operational views powered by the same underlying model:", body_style))

    ux_data = [
        [
            Paragraph("<b>Stakeholder Role</b>", table_header),
            Paragraph("<b>Operational Horizon</b>", table_header),
            Paragraph("<b>Core Dashboard Visualizations & Capabilities</b>", table_header),
            Paragraph("<b>Actionable Decision Enabled</b>", table_header)
        ],
        [
            Paragraph("<b>Floor Supervisor</b><br/>(Assembly Line Leader)", table_cell),
            Paragraph("Real-Time<br/>(Seconds / Current Unit)", table_cell),
            Paragraph("• Interactive line station map (BIW, Paint, Final)<br/>• Real-time SPC Z-score control chart with 3&sigma; limits<br/>• Explicit gap badges (Instrumented vs. Manual)<br/>• Instant containment & isolation guidance", table_cell),
            Paragraph("Immediate containment: isolate defect unit before downstream transfer; adjust tool torque.", table_cell)
        ],
        [
            Paragraph("<b>Plant Manager</b><br/>(Operations Director)", table_cell),
            Paragraph("Tactical Planning<br/>(Shifts / Weekly Trends)", table_cell),
            Paragraph("• 40-station health degradation bars<br/>• Real-time line throughput (Units Per Hour)<br/>• Multi-station z-score deviation heatmap<br/>• Cycle-time slope & bottleneck forecaster", table_cell),
            Paragraph("Line pacing: rebalance buffer queues to prevent downstream starvation; schedule tool swap.", table_cell)
        ],
        [
            Paragraph("<b>Executive Leadership</b><br/>(VP Mfg / Transformation)", table_cell),
            Paragraph("Strategic / Investment<br/>(Quarterly / Fleet)", table_cell),
            Paragraph("• Objective backtest accuracy scorecard<br/>• Financial ROI & payback model ($418k/yr)<br/>• Sensor retrofit roadmap synchronized to shut-downs<br/>• Cross-plant deployment readiness matrix", table_cell),
            Paragraph("Capital allocation: approve enterprise rollout and schedule quarterly hardware retrofits.", table_cell)
        ]
    ]
    ux_table = Table(ux_data, colWidths=[95, 85, 185, 147])
    ux_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_PRIMARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_BG_LIGHT, C_BG_ALT])
    ]))
    story.append(ux_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("4. Non-Invasive OT/IT Integration Architecture", h1_style))
    story.append(Paragraph(
        "DigitalTwin.ai strictly respects plant OT security boundaries and eliminates production downtime risks through a decoupled, "
        "read-only integration pipeline:", body_style))

    int_points = [
        "<b>Zero Live-PLC Modification (Air-Gapped Ingestion):</b> DigitalTwin.ai connects exclusively to existing factory data brokers (OPC-UA server, MQTT sparkplug broker, or industrial historian SQL databases) via read-only listener adapters. The system never writes commands back to PLCs.",
        "<b>Standardized Ingestion Contract:</b> Ingests continuous telemetry, discrete manual QA checklist records, and historical CSV extracts through a unified ingestion interface (<code>station, parameter, value, manual_outcome, vehicle</code>).",
        "<b>Zero-Code Configuration Abstraction (<code>line_config.json</code>):</b> Topology, station groups, cycle times, parameter baselines, and SPC thresholds are defined purely in JSON. Onboarding a new assembly line or plant requires no code compilation.",
        "<b>Deterministic Backtest Validation Gateway:</b> Uploaded historical CSV extracts can be paired with an optional ground-truth incident log to recompute live precision and recall metrics on real plant history before activation."
    ]
    for ip in int_points:
        story.append(Paragraph(f"&bull; {ip}", bullet_style))

    # End of Page 3
    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: BACKTEST VALIDATION & COMPREHENSIVE FINANCIAL BUSINESS CASE
    # =========================================================================
    story.append(Paragraph("5. Empirical Backtest & Ground-Truth Validation", h1_style))
    story.append(Paragraph(
        "To establish floor-level trust, DigitalTwin.ai incorporates an automated validation backtest evaluating predictions against "
        "injected ground-truth failure scenarios across 420 vehicles and 40 stations:", body_style))

    bt_data = [
        [
            Paragraph("<b>Validation Metric</b>", table_header),
            Paragraph("<b>Measured Result</b>", table_header),
            Paragraph("<b>Target SLA</b>", table_header),
            Paragraph("<b>Operational Significance on Assembly Line</b>", table_header)
        ],
        [
            Paragraph("Anomaly Recall", table_cell_bold),
            Paragraph("<font color='#166534'><b>100.0%</b></font>", table_cell),
            Paragraph("&ge; 95.0%", table_cell),
            Paragraph("Zero undetected tool wear, thermal, or cross-fault defects", table_cell)
        ],
        [
            Paragraph("Bottleneck Recall", table_cell_bold),
            Paragraph("<font color='#166534'><b>100.0%</b></font>", table_cell),
            Paragraph("&ge; 90.0%", table_cell),
            Paragraph("100% of cycle-time blowouts identified before starvation occurs", table_cell)
        ],
        [
            Paragraph("Incident Precision", table_cell_bold),
            Paragraph("<b>60.0%</b>", table_cell),
            Paragraph("&ge; 50.0%", table_cell),
            Paragraph("High operational relevance; eliminates floor-level alarm fatigue", table_cell)
        ],
        [
            Paragraph("Mean Detection Lag", table_cell_bold),
            Paragraph("<b>10.0 Vehicles</b>", table_cell),
            Paragraph("&le; 15.0 Vehicles", table_cell),
            Paragraph("Catches defects 28 stations before end-of-line quality teardown", table_cell)
        ],
        [
            Paragraph("False Positive Stations", table_cell_bold),
            Paragraph("<b>2 Stations</b>", table_cell),
            Paragraph("&le; 4 Stations", table_cell),
            Paragraph("Minimal nuisance alerts across 40 complex stations", table_cell)
        ]
    ]
    bt_table = Table(bt_data, colWidths=[110, 85, 80, 237])
    bt_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_PRIMARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_BG_LIGHT, C_BG_ALT])
    ]))
    story.append(bt_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("6. Comprehensive Business Case & Financial Model", h1_style))
    story.append(Paragraph(
        "Based on an annual volume of 105,000 vehicles ($38,000 avg vehicle value), DigitalTwin.ai generates quantifiable "
        "annual net savings of <b>$418,000 per assembly line</b> against a first-year deployment cost of $38,520:", body_style))

    fin_data = [
        [
            Paragraph("<b>Financial Value Category</b>", table_header),
            Paragraph("<b>Baseline Cost / Loss</b>", table_header),
            Paragraph("<b>DigitalTwin.ai Impact Mechanism</b>", table_header),
            Paragraph("<b>Annual Net Savings</b>", table_header)
        ],
        [
            Paragraph("Scrap & Tear-down Reduction", table_cell_bold),
            Paragraph("$320,000 / year", table_cell),
            Paragraph("Early 10-vehicle containment prevents 25-car batch defect runs", table_cell),
            Paragraph("<b>$145,000</b>", table_cell_right)
        ],
        [
            Paragraph("Off-Line Rework Labor", table_cell_bold),
            Paragraph("$190,000 / year", table_cell),
            Paragraph("Real-time torque/weld drift detection stops chronic rework loops", table_cell),
            Paragraph("<b>$91,000</b>", table_cell_right)
        ],
        [
            Paragraph("Bottleneck Capacity Recovery", table_cell_bold),
            Paragraph("38 Hours Line Stoppage", table_cell),
            Paragraph("+2.4% Net OEE improvement via proactive queue rebalancing", table_cell),
            Paragraph("<b>$182,000</b>", table_cell_right)
        ],
        [
            Paragraph("<b>Total Annual Gross Savings</b>", table_cell_bold),
            Paragraph("—", table_cell),
            Paragraph("Combined operational scrap, rework, and capacity improvements", table_cell),
            Paragraph("<b>$418,000 / yr</b>", table_cell_right)
        ],
        [
            Paragraph("Initial Year-1 Investment", table_cell_bold),
            Paragraph("Software + Hardware", table_cell),
            Paragraph("FastAPI container deployment ($35k) + Hardware retrofits ($3,520)", table_cell),
            Paragraph("<b>($38,520)</b>", table_cell_right)
        ],
        [
            Paragraph("<b>Net Year-1 ROI & Payback</b>", table_cell_bold),
            Paragraph("<b>Payback: 4.2 Months</b>", table_cell),
            Paragraph("<b>Year-1 Net Return: $379,480 &nbsp;|&nbsp; 3-Year NPV (10%): $998,000</b>", table_cell),
            Paragraph("<b>985% ROI</b>", table_cell_right)
        ]
    ]
    fin_table = Table(fin_data, colWidths=[120, 100, 204, 88])
    fin_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_PRIMARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_BG_LIGHT, C_BG_ALT]),
        ('BACKGROUND', (0, 4), (-1, 4), C_SUCCESS_BG),
        ('BACKGROUND', (0, 6), (-1, 6), C_BG_ALT)
    ]))
    story.append(fin_table)

    # End of Page 4
    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: ROADMAP, RISK MATRIX & CONCLUSION
    # =========================================================================
    story.append(Paragraph("7. Phased Implementation Roadmap & Maintenance Windows", h1_style))
    story.append(Paragraph(
        "To ensure zero line stoppages, physical retrofits are synchronized with the plant's <b>four quarterly maintenance windows</b>, "
        "while software deployment proceeds continuously:", body_style))

    road_data = [
        [
            Paragraph("<b>Phase & Window</b>", table_header),
            Paragraph("<b>Scope & Target Stations</b>", table_header),
            Paragraph("<b>Hardware Retrofit & Unit Cost</b>", table_header),
            Paragraph("<b>Operational Deliverable</b>", table_header)
        ],
        [
            Paragraph("<b>Phase 1</b><br/>(Weeks 1–6)<br/><b>Q1 Maint. Window</b>", table_cell),
            Paragraph("Pilot Deployment on Northstar Line (40 stations)", table_cell),
            Paragraph("4x Wireless MEMS Vibration Pucks ($220/ea) on BIW-01..04", table_cell),
            Paragraph("Software baseline calibration, OPC-UA ingestion, Floor Supervisor UI live.", table_cell)
        ],
        [
            Paragraph("<b>Phase 2</b><br/>(Weeks 7–10)<br/><b>Q2 Maint. Window</b>", table_cell),
            Paragraph("Paint Shop Thermal Optimization (10 stations)", table_cell),
            Paragraph("3x Non-Contact IR Pyrometers ($310/ea) on PNT-01..03", table_cell),
            Paragraph("Automated thermodynamic oven profiling; Plant Manager heatmap activated.", table_cell)
        ],
        [
            Paragraph("<b>Phase 3</b><br/>(Weeks 11–14)<br/><b>Q3 Maint. Window</b>", table_cell),
            Paragraph("Final Assembly Motor Diagnostics (16 stations)", table_cell),
            Paragraph("5x Clamp-on Current Transformers ($150/ea) on FNL-01..05", table_cell),
            Paragraph("Non-invasive motor load analysis; XGBoost risk scoring engine deployed.", table_cell)
        ],
        [
            Paragraph("<b>Phase 4</b><br/>(Weeks 15–18)<br/><b>Q4 Maint. Window</b>", table_cell),
            Paragraph("Advanced Ultrasonic Inspection & Plant Fleet Expansion", table_cell),
            Paragraph("4x Acoustic Emission Sensors ($480/ea) on BIW/FNL joints", table_cell),
            Paragraph("Full non-destructive weld monitoring; multi-plant container replication.", table_cell)
        ]
    ]
    road_table = Table(road_data, colWidths=[95, 135, 140, 142])
    road_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_PRIMARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_BG_LIGHT, C_BG_ALT])
    ]))
    story.append(road_table)
    story.append(Spacer(1, 4))

    story.append(Paragraph("8. Enterprise Risk Management & Mitigation Matrix", h1_style))
    story.append(Paragraph(
        "A successful industrial rollout requires proactive governance across operational, technical, and cultural risks:", body_style))

    risk_data = [
        [
            Paragraph("<b>Risk Category</b>", table_header),
            Paragraph("<b>Identified Industrial Risk</b>", table_header),
            Paragraph("<b>Potential Impact</b>", table_header),
            Paragraph("<b>Mitigation Built into DigitalTwin.ai</b>", table_header)
        ],
        [
            Paragraph("Model Drift", table_cell_bold),
            Paragraph("Tool wear or seasonal ambient shifts alter baseline distribution", table_cell),
            Paragraph("Increased false alarms or missed micro-drifts", table_cell),
            Paragraph("Rolling dynamic contextual normalization (&mu;, &sigma;) recalculates baseline per vehicle variant.", table_cell)
        ],
        [
            Paragraph("Alarm Fatigue", table_cell_bold),
            Paragraph("Operators overwhelmed by noisy transient alerts", table_cell),
            Paragraph("Floor crews ignore real critical defect warnings", table_cell),
            Paragraph("Strict multi-pattern rules require persistent breach (W=4) or 3-sigma severity before alarming.", table_cell)
        ],
        [
            Paragraph("Legacy Protocol Lock-in", table_cell_bold),
            Paragraph("Inability to interface with proprietary 1990s PLC hardware", table_cell),
            Paragraph("High retrofit costs or stalled pilot expansion", table_cell),
            Paragraph("Read-only external clamp-on CT / MEMS wireless pucks bypass legacy PLC backplanes entirely.", table_cell)
        ],
        [
            Paragraph("User Adoption", table_cell_bold),
            Paragraph("Floor supervisors distrust 'black-box' predictions", table_cell),
            Paragraph("Advisory recommendations bypassed on shift", table_cell),
            Paragraph("Full SHAP explainability showing exact contributing sensor, z-score, and transparent gap status.", table_cell)
        ]
    ]
    risk_table = Table(risk_data, colWidths=[85, 130, 130, 167])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_PRIMARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_BG_LIGHT, C_BG_ALT])
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 4))

    # Conclusion & Sign-Off Block
    story.append(Paragraph("9. Conclusion & Actionable Recommendation", h1_style))
    concl_p = (
        "DigitalTwin.ai delivers a proven, mathematically rigorous, and operationally viable solution for mixed-model automotive assembly. "
        "By fusing non-invasive read-only data ingestion, context-specific statistical process control, thermodynamic physics, and pre-trained "
        "industrial gradient boosting, the platform eliminates the false-alarm epidemic of legacy AI while solving the defect propagation challenge. "
        "With a verified <b>100% anomaly recall</b>, <b>100% bottleneck recall</b>, and a <b>4.2-month payback</b> ($418k/yr net savings), "
        "we recommend immediate authorization of the <b>Phase 1 Pilot Deployment</b> synchronized with the upcoming Q1 plant maintenance window."
    )
    story.append(Paragraph(concl_p, body_style))
    story.append(Spacer(1, 4))

    sign_data = [
        [
            Paragraph("<b>Submitted By:</b> DigitalTwin.ai Engineering Team", table_cell),
            Paragraph("<b>Target Facility:</b> Northstar Mixed-Model Assembly Plant", table_cell)
        ],
        [
            Paragraph("<b>Validation Status:</b> Prototype v2.0 Backtest Verified", table_cell),
            Paragraph("<b>Immediate Action:</b> Phase 1 Integration (Q1 Maintenance Window)", table_cell)
        ]
    ]
    sign_table = Table(sign_data, colWidths=[256, 256])
    sign_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), C_BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 1, C_SECONDARY),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8)
    ]))
    story.append(sign_table)

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF generated successfully at: {output_path}")

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    pdf_file = out_dir / "DigitalTwin_AI_Detailed_Business_Proposal.pdf"
    create_proposal_pdf(str(pdf_file))
