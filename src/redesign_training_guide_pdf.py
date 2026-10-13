"""Build an evidence-led project guide for the cooperative road-hazard system.

The PDF is intentionally regenerated from repository records and code. Values
that appear only as benchmark-script constants are labelled as reported values;
the document does not treat them as independently verified evaluations.
"""

from pathlib import Path
import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, Flowable, HRFlowable,
)


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "Model_Training_Architecture_And_Accuracy_Guide.pdf"
INK = colors.HexColor("#111827")
NAVY = colors.HexColor("#12243A")
SLATE = colors.HexColor("#506176")
MUTED = colors.HexColor("#728196")
PAPER = colors.HexColor("#F5F7FA")
LINE = colors.HexColor("#D9E1EA")
AMBER = colors.HexColor("#F2AE36")
RED = colors.HexColor("#D3544A")
GREEN = colors.HexColor("#2C8A69")
BLUE = colors.HexColor("#3978A8")
WHITE = colors.white


def load_json(path):
    with (ROOT / path).open(encoding="utf-8") as f:
        return json.load(f)


def fmt_pct(value):
    return f"{100 * value:.2f}%"


class ProcessDiagram(Flowable):
    """A compact vector process diagram with labels that remain selectable."""

    def __init__(self, boxes, width=7.18 * inch, height=1.28 * inch):
        super().__init__()
        self.boxes = boxes
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        gap = 13
        n = len(self.boxes)
        bw = (self.width - gap * (n - 1)) / n
        y, bh = 12, self.height - 24
        for i, (tag, title, sub, tint, accent) in enumerate(self.boxes):
            x = i * (bw + gap)
            c.setFillColor(tint)
            c.setStrokeColor(accent)
            c.setLineWidth(0.9)
            c.roundRect(x, y, bw, bh, 7, fill=1, stroke=1)
            c.setFillColor(accent)
            c.setFont("Helvetica-Bold", 6.2)
            c.drawString(x + 8, y + bh - 13, tag.upper())
            c.setFillColor(INK)
            c.setFont("Helvetica-Bold", 8)
            c.drawCentredString(x + bw / 2, y + bh - 29, title)
            c.setFillColor(SLATE)
            c.setFont("Helvetica", 6.6)
            lines = sub.split("\n")
            for j, line in enumerate(lines):
                c.drawCentredString(x + bw / 2, y + bh - 43 - 9 * j, line)
            if i < n - 1:
                ax = x + bw + 2
                ay = y + bh / 2
                c.setStrokeColor(MUTED)
                c.setLineWidth(1.2)
                c.line(ax, ay, ax + gap - 4, ay)
                c.line(ax + gap - 7, ay + 3, ax + gap - 4, ay)
                c.line(ax + gap - 7, ay - 3, ax + gap - 4, ay)


def make_styles():
    base = getSampleStyleSheet()
    return {
        "eyebrow": ParagraphStyle("Eyebrow", fontName="Helvetica-Bold", fontSize=8,
                                   leading=10, textColor=BLUE, spaceAfter=5),
        "title": ParagraphStyle("TitleX", fontName="Helvetica-Bold", fontSize=25,
                                 leading=29, textColor=NAVY, spaceAfter=8),
        "subtitle": ParagraphStyle("SubtitleX", fontName="Helvetica", fontSize=11,
                                    leading=16, textColor=SLATE, spaceAfter=10),
        "h1": ParagraphStyle("H1X", fontName="Helvetica-Bold", fontSize=15,
                              leading=19, textColor=NAVY, spaceBefore=3, spaceAfter=8,
                              keepWithNext=True),
        "h2": ParagraphStyle("H2X", fontName="Helvetica-Bold", fontSize=10.5,
                              leading=14, textColor=INK, spaceBefore=6, spaceAfter=4,
                              keepWithNext=True),
        "body": ParagraphStyle("BodyX", fontName="Helvetica", fontSize=8.8,
                                leading=13, textColor=INK, spaceAfter=6),
        "small": ParagraphStyle("SmallX", fontName="Helvetica", fontSize=7.5,
                                 leading=10.2, textColor=SLATE, spaceAfter=4),
        "note": ParagraphStyle("NoteX", fontName="Helvetica", fontSize=8.1,
                                leading=11.4, textColor=NAVY, backColor=colors.HexColor("#FFF5DF"),
                                borderColor=AMBER, borderWidth=0.7, borderPadding=7,
                                spaceBefore=4, spaceAfter=8),
        "good": ParagraphStyle("GoodX", fontName="Helvetica", fontSize=8.1,
                                leading=11.4, textColor=NAVY, backColor=colors.HexColor("#EAF5F0"),
                                borderColor=GREEN, borderWidth=0.7, borderPadding=7,
                                spaceBefore=4, spaceAfter=8),
        "coverline": ParagraphStyle("CoverLine", fontName="Helvetica-Bold", fontSize=9,
                                     leading=12, textColor=WHITE, alignment=TA_LEFT),
        "covercopy": ParagraphStyle("CoverCopy", fontName="Helvetica", fontSize=8.3,
                                     leading=12, textColor=colors.HexColor("#DEE8F1")),
        "center": ParagraphStyle("CenterX", fontName="Helvetica", fontSize=7.4,
                                 leading=10, textColor=SLATE, alignment=TA_CENTER),
    }


ST = make_styles()


def P(text, style="body"):
    return Paragraph(text, ST[style])


def heading(kicker, title, deck=None):
    out = [P(kicker.upper(), "eyebrow"), P(title, "h1")]
    if deck:
        out.append(P(deck, "body"))
    out.append(HRFlowable(width="100%", thickness=1, color=LINE, spaceBefore=1, spaceAfter=9))
    return out


def callout(text, kind="note"):
    return P(text, kind)


def card_table(rows, widths, header=True, font=7.7):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.45, LINE),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [WHITE, PAPER]),
        ("FONTSIZE", (0, 0), (-1, -1), font),
    ]
    if header:
        commands += [("BACKGROUND", (0, 0), (-1, 0), NAVY),
                     ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                     ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(commands))
    return t


def cover_banner():
    data = [[P("THE BUILD", "coverline"), P("THE CURRENT BOUNDARY", "coverline")],
            [P("Three focused road-hazard detectors feed an edge-to-vehicle warning concept, with SUMO used to explore how earlier warnings could change traffic behavior.", "covercopy"),
             P("Model files and simulation code are in the repo. Edge speed, full live inference in the corridor, and real vehicle/V2X hardware behavior still need validation.", "covercopy")]]
    t = Table(data, colWidths=[3.45 * inch, 3.45 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("BACKGROUND", (0, 0), (0, 0), BLUE),
        ("BACKGROUND", (1, 0), (1, 0), RED),
        ("BOX", (0, 0), (-1, -1), 0, NAVY),
        ("INNERGRID", (0, 0), (-1, -1), 0, NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
    ]))
    return t


class GuideDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(filename, pagesize=letter, leftMargin=0.72*inch,
                         rightMargin=0.72*inch, topMargin=0.68*inch,
                         bottomMargin=0.62*inch, title="Edge-AI Cooperative Road Hazard Detection — Project Guide",
                         author="PFPD-ll project")
        frame = Frame(self.leftMargin, self.bottomMargin,
                      self.width, self.height, id="main",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="guide", frames=[frame], onPage=self.decorate))

    @staticmethod
    def decorate(canvas, doc):
        canvas.saveState()
        w, h = letter
        canvas.setFillColor(NAVY)
        canvas.rect(0, h-7, w, 7, fill=1, stroke=0)
        if doc.page > 1:
            canvas.setFont("Helvetica-Bold", 7)
            canvas.setFillColor(SLATE)
            canvas.drawString(doc.leftMargin, h-28, "PFPD-ll  /  COOPERATIVE ROAD HAZARD SYSTEM")
            canvas.setStrokeColor(LINE)
            canvas.line(doc.leftMargin, h-34, w-doc.rightMargin, h-34)
        canvas.setStrokeColor(LINE)
        canvas.line(doc.leftMargin, 34, w-doc.rightMargin, 34)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc.leftMargin, 22, "PROJECT CONTEXT  /  LOCAL REPOSITORY EVIDENCE  /  13 OCT 2026")
        canvas.drawRightString(w-doc.rightMargin, 22, f"{doc.page:02d}")
        canvas.restoreState()


def build_story():
    pothole = load_json("model_training/pothole_detector/metrics/evaluation_summary.json")
    construction = load_json("model_training/road_construction_detector/metrics/evaluation_summary.json")
    profile = load_json("results/empirical_model_comparison.json")
    state = load_json("reports/canonical_model_summary.json")
    models = profile["models"]
    story = []

    # 1 — Purpose and one-page overview.
    story += [Spacer(1, 14), P("PROJECT BRIEF  /  EDGE AI × COOPERATIVE MOBILITY", "eyebrow"),
              P("See a road hazard.<br/>Warn the next vehicle.", "title"),
              P("A project guide to the models, edge-inference work, and SUMO simulations in PFPD-ll", "subtitle"),
              HRFlowable(width="100%", thickness=2, color=AMBER, spaceAfter=14),
              cover_banner(), Spacer(1, 15), P("The idea in one loop", "h2"),
              ProcessDiagram([
                  ("01 · SENSE", "Road camera", "Image / frame input", colors.HexColor("#EAF1F8"), BLUE),
                  ("02 · DETECT", "3 hazard models", "Pothole · works · crash", colors.HexColor("#FFF4DD"), AMBER),
                  ("03 · SHARE", "RSU warning", "DENM-shaped message", colors.HexColor("#FCECE9"), RED),
                  ("04 · RESPOND", "Traffic response", "Alert · slow · change lane", colors.HexColor("#EAF5F0"), GREEN),
              ]), Spacer(1, 8),
              P("This is a cooperative safety concept: a vehicle or roadside camera detects a hazard, an edge device turns it into a compact warning, and other vehicles can react before reaching the same location. The repository brings together model artifacts, inference code, message construction, and SUMO traffic scenarios to develop and study that loop.", "body"),
              P("What is in the project today", "h2")]
    summary = [
        [P("MODEL ARTIFACTS", "eyebrow"), P("EDGE CODE", "eyebrow"), P("TRAFFIC STUDY", "eyebrow")],
        [P("Three YOLO11s INT8 ONNX files in <font name='Courier'>quantization/</font>, each about 9.48 MB.", "small"),
         P("ONNX Runtime inference and DENM-shaped message code in <font name='Courier'>simulation/scripts/</font>.", "small"),
         P("A 1 km SUMO corridor with two signalized intersections and three placed hazard zones.", "small")]
    ]
    t = Table(summary, colWidths=[2.3*inch, 2.3*inch, 2.3*inch])
    t.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,-1), PAPER), ("BOX",(0,0),(-1,-1),0.7,LINE),
                           ("INNERGRID",(0,0),(-1,-1),0.5,LINE), ("VALIGN",(0,0),(-1,-1),"TOP"),
                           ("LEFTPADDING",(0,0),(-1,-1),9), ("RIGHTPADDING",(0,0),(-1,-1),9),
                           ("TOPPADDING",(0,0),(-1,-1),8), ("BOTTOMPADDING",(0,0),(-1,-1),8)]))
    story += [t, PageBreak()]

    # 2 — Architecture and implementation boundaries.
    story += heading("01 / System architecture", "From a camera frame to a traffic warning",
                     "The repository has several working pieces. The diagram shows the intended system path and labels the boundaries between implemented code and the simulation scenario.")
    story += [ProcessDiagram([
        ("INPUT", "Camera image", "RGB frame", colors.HexColor("#EAF1F8"), BLUE),
        ("EDGE", "ONNX Runtime", "3 specialized models", colors.HexColor("#FFF4DD"), AMBER),
        ("EVENT", "Hazard record", "Type · score · box", colors.HexColor("#FCECE9"), RED),
        ("MESSAGE", "DENM-shaped JSON", "Cause · place · advice", colors.HexColor("#F2EDFA"), colors.HexColor("#7C62A3")),
        ("TRAFFIC", "SUMO / TraCI", "Scripted response path", colors.HexColor("#EAF5F0"), GREEN),
    ]), Spacer(1, 8),
        P("Three specialists", "h2"),
        P("The model set divides perception into pothole / damaged surface, construction / barricades, and accident / debris. The quantized files are present under <font name='Courier'>quantization/</font>. The code in <font name='Courier'>simulation/scripts/rsu_hazard_disseminator.py</font> contains an ONNX Runtime detector and a DENM-shaped event builder.", "body"),
        P("Two separate demonstrations", "h2"),
        card_table([
            ["Component", "What its code does", "Reading it accurately"],
            ["Image-to-event script", "Loads an ONNX model when dependencies and files are available; post-processes predictions; builds a JSON warning; can apply a response through TraCI.", "This is the repository's actual inference path. If a model/runtime is unavailable, the detector has a mock fallback."],
            ["1 km cooperative SUMO run", "Places three hazards on a corridor. The first vehicle encountering each hazard is treated as detector; later vehicles receive scripted lane/speed actions.", "The run associates model names, confidence and latency values with scenarios; it does not load the ONNX models for each SUMO vehicle."],
        ], [1.25*inch, 2.75*inch, 2.9*inch], font=7.5), Spacer(1, 8),
        callout("The pieces point toward one integrated Edge AI → RSU → connected-vehicle system. Current repository evidence supports a prototype made of inference/message code plus a SUMO cooperative-response demonstration; it does not yet establish a camera-to-radio deployment running end to end.")]
    story.append(PageBreak())

    # 3 — Model suite and measured evidence.
    story += heading("02 / Model suite and evidence", "Three edge artifacts, with performance claims kept in scope",
                     "The project records two different model families. Keeping their results separate avoids attributing YOLO11n evaluation numbers to the YOLO11s quantized models.")
    model_rows = [["Quantized YOLO11s artifact", "Size", "Recorded reference P / R / mAP50", "Recorded CPU result" ]]
    for m in models:
        edge = m.get("empirical_edge", {})
        ref = m.get("gpu_training", {})
        model_rows.append([
            m["quantized_file"], f"{m['size_mb']:.2f} MB",
            f"{ref.get('precision', 0):.2f}% / {ref.get('recall', 0):.2f}% / {ref.get('map50', 0):.2f}%",
            P(f"{edge.get('mean_ms', 0):,.0f} ms mean<br/>{edge.get('fps', 0):.1f} FPS", "small"),
        ])
    story += [P("Recorded YOLO11s suite", "h2"), card_table(model_rows, [2.0*inch, 0.68*inch, 1.72*inch, 2.5*inch], font=7.2), Spacer(1, 5),
              callout("The precision / recall / mAP figures above are copied from the GPU-reference constants in <font name='Courier'>results/empirical_model_comparison.json</font> and <font name='Courier'>src/fast_empirical_benchmark.py</font>. The same result file records the CPU timings shown here. The local checkout does not include matching YOLO11s test-split reports to independently verify those reference accuracy figures. Treat them as project-reported metrics, not an audited comparison.", "note"),
              P("Separate local YOLO11n evaluation records", "h2")]
    local_rows = [["Recorded experiment", "Test images", "Precision", "Recall", "mAP50", "mAP50–95", "INT8 file"]]
    local_rows.append(["Pothole · 7,066-image report", str(pothole["test_images"]), fmt_pct(pothole["precision_standard"]), fmt_pct(pothole["recall_standard"]), fmt_pct(pothole["map50"]), fmt_pct(pothole["map50_95"]), f"{pothole['int8_size_mb']:.2f} MB"])
    local_rows.append(["Construction · YOLO11n report", str(construction["test_images"]), fmt_pct(construction["precision"]), fmt_pct(construction["recall"]), fmt_pct(construction["map50"]), fmt_pct(construction["map50_95"]), f"{construction['int8_size_mb']:.2f} MB"])
    story += [card_table(local_rows, [1.62*inch, .62*inch, .69*inch, .62*inch, .58*inch, .76*inch, .75*inch], font=6.9), Spacer(1, 6),
              P(f"These are YOLO11n records in <font name='Courier'>model_training/</font>, not the three YOLO11s quantization files. The pothole report has {pothole['test_images']} held-out images; the construction report has {construction['test_images']}. No corresponding accident-model evaluation summary was found in the local report files.", "small"),
              P("What the edge timing says", "h2"),
              P(f"The recorded CPU means for the three YOLO11s ONNX models are {models[0]['empirical_edge']['mean_ms']:,.0f}–{models[2]['empirical_edge']['mean_ms']:,.0f} ms per inference (about {models[0]['empirical_edge']['fps']:.1f} FPS in the benchmark record). That measurement does not support the README's separate ~35 FPS Raspberry Pi claim. Hardware, runtime, preprocessing, and measurement method must be pinned down before making a real-time edge-performance claim.", "body"),
              callout(f"The profile loads three models concurrently into memory; process RAM is {profile['ram_consumption_mb']:.1f} MB including about {profile['ram_overhead_mb']:.1f} MB reported runtime overhead. This is a software benchmark record, not Raspberry Pi board validation.", "good"), PageBreak()]

    # 4 — SUMO system and cooperative event.
    story += heading("03 / SUMO cooperative simulation", "Study how a warning could change what traffic does",
                     "The corridor scenario makes the cooperative idea visible: one vehicle encounters a hazard first, the RSU remembers it, and later vehicles are warned upstream.")
    story += [ProcessDiagram([
        ("CORRIDOR", "1,000 m", "Two signalized junctions", colors.HexColor("#EAF1F8"), BLUE),
        ("DETECTOR", "First vehicle", "Discovers scenario event", colors.HexColor("#FFF4DD"), AMBER),
        ("RSU", "Stores event", "DENM-shaped packet", colors.HexColor("#F2EDFA"), colors.HexColor("#7C62A3")),
        ("RECEIVERS", "Following cars", "Lane / speed response", colors.HexColor("#EAF5F0"), GREEN),
    ]), Spacer(1, 7)]
    sim_rows = [
        ["Scenario item", "Repository configuration"],
        ["Road", "1 km arterial corridor; two signalized intersections; two mainline lanes."],
        ["Roadside unit", "RSU_ZONE_01 at approximately x = 500 m."],
        ["Hazard zones", "Construction at about 550 m; pothole at about 650 m; accident / debris at about 730 m."],
        ["Traffic", "Continuous connected-car flow plus opposing and cross-street traffic in the route file."],
        ["Message fields", "Hazard type, ETSI cause-code mapping, location, severity, recommended action, and advisory speed are represented in JSON structures."],
        ["Response", "TraCI script changes lane and speed in the scenario; other code paths attempt lane restrictions or speed limits."],
    ]
    story += [card_table(sim_rows, [1.35*inch, 5.55*inch], font=7.7), Spacer(1, 10),
              P("How to explain the demo", "h2"),
              P("SUMO is a traffic simulator. It lets the team test traffic behavior and visualize hazard-warning logic on a defined road network. In the current 1 km runner, detector/receiver roles and vehicle reactions are rule-based. They illustrate the dissemination concept; they are not evidence that physical V2X radios, production DENM interoperability, or a live camera inference stream has been validated.", "body"),
              callout("There is also a separate image inference / TraCI path in <font name='Courier'>simulation/scripts/rsu_hazard_disseminator.py</font>. Treat that path and the corridor scenario as related prototype components, not as one proven integrated deployment.") ,
              P("Primary files", "h2"),
              P("<font name='Courier'>simulation/corridor_1km/run_cooperative_v2x_sim.py</font> · scenario loop<br/><font name='Courier'>simulation/corridor_1km/corridor.sumocfg</font> and <font name='Courier'>corridor.rou.xml</font> · network timing and flows<br/><font name='Courier'>simulation/scripts/rsu_hazard_disseminator.py</font> · ONNX inference and warning-message code", "small"), PageBreak()]

    # 5 — Training concepts, metric reading, current status and next steps.
    story += heading("04 / Training and project state", "A useful baseline for the next build phase",
                     "This section summarizes how the project trains and scores a detector, while tying project-specific results to the reports that are actually present.")
    story += [P("How a detector is trained and scored", "h2"),
              ProcessDiagram([
                  ("DATA", "Images + labels", "Train / val / test", colors.HexColor("#EAF1F8"), BLUE),
                  ("TRAIN", "YOLO detector", "Loss updates weights", colors.HexColor("#FFF4DD"), AMBER),
                  ("EVALUATE", "Held-out images", "Match boxes by IoU", colors.HexColor("#F2EDFA"), colors.HexColor("#7C62A3")),
                  ("EXPORT", "ONNX → INT8", "Smaller edge artifact", colors.HexColor("#EAF5F0"), GREEN),
              ]), Spacer(1, 6),
              P("Precision answers: of the boxes the model predicted, how many were correct? Recall answers: of the labeled hazards, how many did it find? AP summarizes a precision–recall curve for a class. mAP@50 averages AP using an IoU match threshold of 0.50; mAP@50–95 averages over stricter IoU thresholds too. A strong score on one frozen test set does not establish field performance or device speed.", "body"),
              P("Where the project stands", "h2")]
    status_rows = [
        [P("READY TO SHOW", "coverline"), P("NEEDS THE NEXT EVIDENCE", "coverline")],
        [P("Three quantized model artifacts; training and export scripts; per-domain YOLO11n reports for pothole and construction; multi-hazard SUMO corridor and warning logic.", "small"),
         P("Reproducible YOLO11s test-set reports for all three files; calibrated INT8-vs-FP32 accuracy; target-board latency; live camera-to-inference-to-RSU flow; radio/network interoperability; controlled safety/traffic evaluation.", "small")],
    ]
    t = Table(status_rows, colWidths=[3.45*inch, 3.45*inch])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(0,0),GREEN), ("BACKGROUND",(1,0),(1,0),RED),
                           ("BACKGROUND",(0,1),(-1,1),PAPER), ("TEXTCOLOR",(0,0),(-1,0),WHITE),
                           ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                           ("LEADING",(0,1),(-1,1),11), ("VALIGN",(0,0),(-1,-1),"TOP"),
                           ("BOX",(0,0),(-1,-1),0.5,LINE), ("INNERGRID",(0,0),(-1,-1),0.5,LINE),
                           ("LEFTPADDING",(0,0),(-1,-1),8), ("RIGHTPADDING",(0,0),(-1,-1),8),
                           ("TOPPADDING",(0,0),(-1,-1),7), ("BOTTOMPADDING",(0,0),(-1,-1),7)]))
    story += [t, Spacer(1, 8), P("Practical next sequence", "h2"),
              P("1. Freeze the three YOLO11s datasets, splits, checkpoints and evaluation commands.<br/>2. Re-run per-class FP32 and INT8 validation on the same held-out images.<br/>3. Benchmark each model and the three-model pipeline on the intended edge board, recording preprocessing and end-to-end latency.<br/>4. Connect live image detections to one hazard-event schema; then connect that message path to a controlled SUMO experiment.<br/>5. Compare warned and unwarned traffic outcomes before making safety or deployment claims.", "body"),
              P("Repository map", "h2"),
              P("<font name='Courier'>quantization/</font> · three YOLO11s INT8 files &amp; model note  &nbsp;|&nbsp;  <font name='Courier'>model_training/</font> · training, export, quantization and YOLO11n reports<br/><font name='Courier'>results/</font> · recorded three-model benchmark profile  &nbsp;|&nbsp;  <font name='Courier'>simulation/</font> · SUMO networks, corridor scenario and dissemination code", "small"),
              Spacer(1, 5), P("Prepared from the PFPD-ll working tree and the linked GitHub repository. Metric provenance and implementation boundaries are called out so the guide can introduce the project without overstating validation.", "small")]
    return story


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = GuideDoc(str(OUTPUT))
    doc.build(build_story())
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
