import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

st.set_page_config(
    page_title="RetinaAI — DR Detection",
    layout="wide",
    page_icon="👁️",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, .stApp {
    background-color: #04080F !important;
    font-family: 'Inter', sans-serif;
}
.stApp {
    background:
        radial-gradient(ellipse 60% 40% at 10% 0%, rgba(0,120,255,0.07) 0%, transparent 100%),
        radial-gradient(ellipse 50% 40% at 90% 100%, rgba(120,0,255,0.05) 0%, transparent 100%),
        #04080F;
}
section[data-testid="stSidebar"] {
    background: #060B15 !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
    width: 260px !important;
}
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

div[data-testid="stFileUploadDropzone"] {
    background: rgba(0,140,255,0.03) !important;
    border: 1.5px dashed rgba(0,140,255,0.3) !important;
    border-radius: 18px !important;
}
div[data-testid="stFileUploadDropzone"]:hover {
    border-color: rgba(0,180,255,0.7) !important;
    background: rgba(0,140,255,0.07) !important;
}
div[data-testid="stFileUploadDropzone"] p,
div[data-testid="stFileUploadDropzone"] span,
div[data-testid="stFileUploadDropzone"] small {
    color: rgba(255,255,255,0.45) !important;
    font-family: 'Inter', sans-serif !important;
}
div[data-testid="stFileUploadDropzone"] svg { stroke: rgba(0,180,255,0.5) !important; }

[data-testid="stImage"] img {
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    display: block;
    width: 100%;
}
.stMarkdown p { color: rgba(255,255,255,0.7); line-height: 1.7; }
h1,h2,h3,h4 { font-family: 'Space Grotesk', sans-serif !important; color: white !important; }
hr { border: none !important; border-top: 1px solid rgba(255,255,255,0.06) !important; margin: 1.5rem 0 !important; }
div[data-testid="column"] { padding: 0 0.5rem; }
.stSpinner > div { border-top-color: #00AAFF !important; }
[data-testid="stBarChart"] canvas { border-radius: 10px; }

            
</style>
""", unsafe_allow_html=True)

import os
import gdown

model.load_weights(MODEL_PATH)

FILE_ID = "1nj_E76r2iNSsjw17Y8LedwlYr3PxHK3w"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/file/d/1nj_E76r2iNSsjw17Y8LedwlYr3PxHK3w/view?usp=sharing"
    gdown.download(url, MODEL_PATH, quiet=False)

@st.cache_resource(show_spinner=False)
def load_model():
    base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)
    ])
    
    model = models.Sequential([
        # 1. Force a strict, static batch size of 1 right at the start
        tf.keras.layers.Input(batch_shape=(1, 224, 224, 3)),
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(5, activation="softmax")
    ])
    
    model.load_weights("models/retina_weights.weights.h5")
    return model


model = load_model()

class_names = ["Healthy", "Mild DR", "Moderate DR", "Proliferate DR", "Severe DR"]

CLASS_META = {
    "Healthy":        {"color":"#00E676","level":0,"icon":"✦","urgency":"Routine","action":"Routine annual screening recommended.","desc":"No pathological signs detected. Retinal vasculature, optic disc, and macula appear within normal limits."},
    "Mild DR":        {"color":"#FFD740","level":1,"icon":"◈","urgency":"Low",    "action":"Follow-up within 12 months.","desc":"Early microaneurysms detected — the earliest clinical sign of diabetic retinopathy."},
    "Moderate DR":    {"color":"#FF9100","level":2,"icon":"◉","urgency":"Moderate","action":"Ophthalmology referral within 3–6 months.","desc":"Retinal hemorrhages, hard exudates or venous beading observed, indicating progressive ischemia."},
    "Severe DR":      {"color":"#FF5252","level":3,"icon":"⬡","urgency":"High",   "action":"Urgent ophthalmology referral within 2–4 weeks.","desc":"Extensive hemorrhages, venous beading, or IRMA present. High risk of rapid progression."},
    "Proliferate DR": {"color":"#D500F9","level":4,"icon":"⬟","urgency":"Critical","action":"Immediate retinal specialist referral. Vision-threatening.","desc":"Neovascularization detected. Risk of vitreous hemorrhage and tractional retinal detachment."},
}
SEVERITY_LABELS = ["None","Minimal","Moderate","Severe","Critical"]


def hex_to_rgb(h):
    h = h.lstrip('#')
    return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)


with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.75rem;padding:0 0.5rem;">
        <div style="width:36px;height:36px;background:linear-gradient(135deg,#0055FF,#00AAFF);
                    border-radius:9px;display:flex;align-items:center;justify-content:center;
                    font-size:17px;flex-shrink:0;">👁️</div>
        <div>
            <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;color:white;
                        font-size:1.05rem;line-height:1.2;">RetinaAI</div>
            <div style="font-size:0.62rem;color:rgba(255,255,255,0.35);letter-spacing:0.08em;
                        text-transform:uppercase;">Diagnostic System v1.0</div>
        </div>
    </div>

    <div style="background:rgba(0,100,255,0.07);border:1px solid rgba(0,120,255,0.18);
                border-radius:12px;padding:1rem;margin-bottom:1.5rem;">
        <div style="font-size:0.6rem;color:#00AAFF;letter-spacing:0.1em;text-transform:uppercase;
                    font-weight:600;margin-bottom:0.7rem;">Model Specification</div>
    """ + "".join([
        f"""<div style="display:flex;justify-content:space-between;padding:0.28rem 0;
                        border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="font-size:0.74rem;color:rgba(255,255,255,0.4);">{k}</span>
                <span style="font-size:0.74rem;color:white;font-weight:500;
                             font-family:'JetBrains Mono',monospace;">{v}</span>
            </div>"""
        for k, v in [("Backbone","ResNet50"),("Framework","TensorFlow"),
                     ("Input","224 × 224 × 3"),("Classes","5"),("Augmentation","Active")]
    ]) + """
    </div>

    <div style="margin-bottom:1.5rem;">
        <div style="font-size:0.6rem;color:rgba(255,255,255,0.3);letter-spacing:0.1em;
                    text-transform:uppercase;margin-bottom:0.6rem;padding:0 0.1rem;">
            Severity Scale
        </div>
    """ + "".join([
        f"""<div style="display:flex;align-items:center;gap:9px;padding:0.38rem 0.1rem;
                        border-bottom:1px solid rgba(255,255,255,0.03);">
                <div style="width:7px;height:7px;border-radius:50%;background:{m['color']};
                            box-shadow:0 0 5px {m['color']};flex-shrink:0;"></div>
                <span style="font-size:0.76rem;color:rgba(255,255,255,0.65);">{cls}</span>
                <span style="margin-left:auto;font-size:0.66rem;color:rgba(255,255,255,0.25);
                             font-family:'JetBrains Mono',monospace;">G{m['level']}</span>
            </div>"""
        for cls, m in CLASS_META.items()
    ]) + """
    </div>

    <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
                border-radius:10px;padding:0.85rem;">
        <div style="font-size:0.6rem;color:rgba(255,255,255,0.3);letter-spacing:0.1em;
                    text-transform:uppercase;margin-bottom:0.55rem;">Project</div>
        <div style="font-size:0.75rem;color:rgba(255,255,255,0.55);line-height:1.9;">
            🎓 UE Potsdam<br>📐 Pattern Recognition<br>👨‍💻 Ashik Kirmani
        </div>
    </div>

    <div style="margin-top:1.75rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.04);">
        <div style="font-size:0.62rem;color:rgba(255,255,255,0.18);line-height:1.6;text-align:center;">
            Research & educational use only.<br>Not a substitute for clinical diagnosis.
        </div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div style="padding:2rem 0 1.25rem;">
    <div style="font-size:0.62rem;color:#00AAFF;letter-spacing:0.16em;text-transform:uppercase;
                font-weight:600;margin-bottom:0.7rem;font-family:'Space Grotesk',sans-serif;">
        Deep Learning · Ophthalmology · CNN Classification
    </div>
    <h1 style="font-family:'Space Grotesk',sans-serif;font-size:2.6rem;font-weight:700;
               color:white;line-height:1.1;margin-bottom:0.5rem;">
        Diabetic Retinopathy
        <span style="background:linear-gradient(90deg,#0066FF,#00AAFF,#00E5FF);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                     background-clip:text;"> Severity Detection</span>
    </h1>
    <p style="font-size:0.92rem;color:rgba(255,255,255,0.45);max-width:540px;line-height:1.75;
              margin-top:0.4rem;">
        Upload a fundus photograph for automated grading using a ResNet50 CNN trained
        on five ICDR severity classes of diabetic retinopathy.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex;gap:0.75rem;margin-bottom:1.75rem;flex-wrap:wrap;">
""" + "".join([
    f"""<div style="background:rgba(0,80,255,0.08);border:1px solid rgba(0,100,255,0.2);
                border-radius:12px;padding:0.85rem 1.1rem;flex:1;min-width:130px;">
        <div style="font-size:0.58rem;color:rgba(255,255,255,0.35);letter-spacing:0.1em;
                    text-transform:uppercase;margin-bottom:0.3rem;">{label}</div>
        <div style="font-size:1.25rem;font-weight:700;color:white;
                    font-family:'Space Grotesk',sans-serif;line-height:1;">{val}</div>
        <div style="font-size:0.68rem;color:#00AAFF;margin-top:0.2rem;">{sub}</div>
    </div>"""
    for label, val, sub in [
        ("Architecture","ResNet50","50-layer deep CNN"),
        ("Classes","5 Grades","ICDR severity scale"),
        ("Input Size","224²","Pixels per channel"),
        ("Augmentation","Active","Flip · Rotate · Zoom"),
    ]
]) + "</div>", unsafe_allow_html=True)

st.markdown("""
<div style="font-size:0.6rem;color:rgba(255,255,255,0.3);letter-spacing:0.12em;
            text-transform:uppercase;font-weight:600;margin-bottom:0.6rem;">
    Fundus Image Input
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop a retinal fundus image here, or click to browse",
    type=["jpg","jpeg","png"],
    label_visibility="visible"
)

st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

if uploaded_file is not None:
    # 1. Open and convert the image
    image = Image.open(uploaded_file).convert("RGB")
    
    # 2. Resize, convert to NumPy array, and normalize to [0, 1]
    img_arr = np.array(image.resize((224,224))).astype("float32")
    
    # 3. Add the batch dimension: (224, 224, 3) becomes (1, 224, 224, 3)
    img_arr = np.expand_dims(img_arr, axis=0)

    # 4. Now img_arr is perfectly defined and ready for inference
    with st.spinner("Running inference…"):
            # 2. Convert to a static TensorFlow constant
            img_tensor = tf.constant(img_arr, dtype=tf.float32)
            
            # 3. Call the model directly with training=False to disable the augmentation layers
            prediction = model(img_tensor, training=False).numpy()

    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction)) * 100
    meta = CLASS_META[predicted_class]
    raw_probs = prediction[0]
    r, g, b = hex_to_rgb(meta["color"])

    st.markdown(f"""
    <div style="height:1px;background:linear-gradient(90deg,{meta['color']}60,transparent);
                margin-bottom:1.5rem;"></div>
    <div style="font-size:0.6rem;color:rgba(255,255,255,0.3);letter-spacing:0.12em;
                text-transform:uppercase;font-weight:600;margin-bottom:1rem;">
        Analysis Results
    </div>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([1,1], gap="large")

    with left_col:
        st.image(image, use_container_width=True)
        w, h = image.size
        st.markdown(f"""
        <div style="display:flex;gap:0.5rem;margin-top:0.6rem;flex-wrap:wrap;">
            <span style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);
                         border-radius:7px;padding:0.3rem 0.65rem;font-size:0.7rem;
                         color:rgba(255,255,255,0.45);font-family:'JetBrains Mono',monospace;">
                {w} × {h} px
            </span>
            <span style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);
                         border-radius:7px;padding:0.3rem 0.65rem;font-size:0.7rem;
                         color:rgba(255,255,255,0.45);font-family:'JetBrains Mono',monospace;">RGB</span>
            <span style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);
                         border-radius:7px;padding:0.3rem 0.65rem;font-size:0.7rem;
                         color:rgba(255,255,255,0.45);font-family:'JetBrains Mono',monospace;">
                {uploaded_file.name.split('.')[-1].upper()}
            </span>
        </div>

        <div style="margin-top:1.5rem;">
            <div style="font-size:0.6rem;color:rgba(255,255,255,0.3);letter-spacing:0.1em;
                        text-transform:uppercase;margin-bottom:0.75rem;">Preprocessing Pipeline</div>
        """ + "".join([
            f"""<div style="display:flex;align-items:flex-start;gap:9px;margin-bottom:{'0' if i==4 else '2px'};">
                    <div style="width:16px;height:16px;border-radius:50%;background:{c};flex-shrink:0;
                                display:flex;align-items:center;justify-content:center;
                                font-size:0.55rem;font-weight:700;color:white;margin-top:1px;">{i+1}</div>
                    <div style="flex:1;">
                        <div style="font-size:0.75rem;color:white;font-weight:500;">{s}</div>
                        <div style="font-size:0.67rem;color:rgba(255,255,255,0.35);
                                    font-family:'JetBrains Mono',monospace;">{d}</div>
                    </div>
                </div>
                {'<div style="width:1px;height:10px;background:rgba(255,255,255,0.07);margin-left:7px;"></div>' if i < 4 else ''}"""
            for i,(s,d,c) in enumerate([
                ("Load & Convert","PIL → RGB","#0055FF"),
                ("Resize",f"{w}×{h} → 224×224","#0077FF"),
                ("Normalize","[0,255] → [0,1] float32","#0099FF"),
                ("Batch Expand","(224,224,3) → (1,224,224,3)","#00BBFF"),
                ("Inference","ResNet50 forward pass","#00DDFF"),
            ])
        ]) + "</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba({r},{g},{b},0.09),rgba(4,8,15,0.95));
                    border:1px solid rgba({r},{g},{b},0.35);border-radius:18px;
                    padding:1.6rem;margin-bottom:1rem;position:relative;overflow:hidden;">
            <div style="position:absolute;top:-50px;right:-50px;width:160px;height:160px;
                        border-radius:50%;background:rgba({r},{g},{b},0.12);
                        filter:blur(50px);pointer-events:none;"></div>
            <div style="font-size:0.6rem;color:{meta['color']};letter-spacing:0.12em;
                        text-transform:uppercase;font-weight:600;margin-bottom:0.5rem;">
                Classification Result
            </div>
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:1.1rem;">
                <div style="font-size:2rem;color:{meta['color']};line-height:1;">{meta['icon']}</div>
                <div>
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:1.8rem;
                                font-weight:700;color:white;line-height:1.1;">{predicted_class}</div>
                    <div style="font-size:0.72rem;color:rgba(255,255,255,0.4);">
                        ICDR Grade {meta['level']} · {SEVERITY_LABELS[meta['level']]} severity
                    </div>
                </div>
            </div>
            <div style="margin-bottom:1.1rem;">
                <div style="display:flex;justify-content:space-between;margin-bottom:0.35rem;">
                    <span style="font-size:0.72rem;color:rgba(255,255,255,0.45);">Model confidence</span>
                    <span style="font-size:0.82rem;font-weight:700;color:{meta['color']};
                                 font-family:'JetBrains Mono',monospace;">{confidence:.2f}%</span>
                </div>
                <div style="height:5px;background:rgba(255,255,255,0.06);border-radius:100px;overflow:hidden;">
                    <div style="height:100%;width:{confidence}%;border-radius:100px;
                                background:linear-gradient(90deg,{meta['color']}88,{meta['color']});"></div>
                </div>
            </div>
            <div style="display:flex;gap:0.5rem;flex-wrap:wrap;">
                <div style="background:rgba({r},{g},{b},0.15);border:1px solid rgba({r},{g},{b},0.4);
                            border-radius:6px;padding:0.28rem 0.65rem;font-size:0.68rem;
                            color:{meta['color']};font-weight:700;letter-spacing:0.04em;">
                    {meta['urgency']} Priority
                </div>
                <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);
                            border-radius:6px;padding:0.28rem 0.65rem;font-size:0.68rem;
                            color:rgba(255,255,255,0.55);">Grade {meta['level']} / 4</div>
            </div>
        </div>

        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
                    border-radius:14px;padding:1.1rem;margin-bottom:1rem;">
            <div style="font-size:0.6rem;color:rgba(255,255,255,0.3);letter-spacing:0.1em;
                        text-transform:uppercase;margin-bottom:0.55rem;">Clinical Interpretation</div>
            <p style="font-size:0.82rem;color:rgba(255,255,255,0.65);line-height:1.75;
                      margin-bottom:0.7rem;">{meta['desc']}</p>
            <div style="display:flex;align-items:flex-start;gap:7px;
                        background:rgba(0,150,255,0.07);border-left:2px solid #00AAFF;
                        border-radius:0 8px 8px 0;padding:0.55rem 0.8rem;">
                <span style="font-size:0.72rem;color:#00AAFF;flex-shrink:0;margin-top:1px;">→</span>
                <span style="font-size:0.75rem;color:rgba(255,255,255,0.6);line-height:1.65;">
                    <strong style="color:white;">Action:</strong> {meta['action']}
                </span>
            </div>
        </div>

        <div style="font-size:0.6rem;color:rgba(255,255,255,0.3);letter-spacing:0.1em;
                    text-transform:uppercase;margin-bottom:0.65rem;">
            Confidence Distribution
        </div>
        """ + "".join([
            f"""<div style="background:rgba({hex_to_rgb(CLASS_META[cls]['color'])[0]},{hex_to_rgb(CLASS_META[cls]['color'])[1]},{hex_to_rgb(CLASS_META[cls]['color'])[2]},{'0.09' if i==predicted_index else '0.03'});
                        border:1px solid {CLASS_META[cls]['color']}{'44' if i==predicted_index else '12'};
                        border-radius:9px;padding:0.55rem 0.85rem;margin-bottom:0.4rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem;">
                    <div style="display:flex;align-items:center;gap:7px;">
                        <div style="width:6px;height:6px;border-radius:50%;background:{CLASS_META[cls]['color']};
                                    {'box-shadow:0 0 5px '+CLASS_META[cls]['color']+';' if i==predicted_index else ''}"></div>
                        <span style="font-size:0.75rem;font-weight:{'700' if i==predicted_index else '400'};
                                     color:{'white' if i==predicted_index else 'rgba(255,255,255,0.5)'};">
                            {cls}{' ← predicted' if i==predicted_index else ''}
                        </span>
                    </div>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                                 font-weight:{'700' if i==predicted_index else '400'};
                                 color:{CLASS_META[cls]['color'] if i==predicted_index else 'rgba(255,255,255,0.4)'};">
                        {float(raw_probs[i])*100:.2f}%
                    </span>
                </div>
                <div style="height:3px;background:rgba(255,255,255,0.05);border-radius:100px;overflow:hidden;">
                    <div style="height:100%;width:{float(raw_probs[i])*100}%;border-radius:100px;
                                background:{CLASS_META[cls]['color']};
                                opacity:{'1' if i==predicted_index else '0.45'};"></div>
                </div>
            </div>"""
            for i, cls in enumerate(class_names)
        ]), unsafe_allow_html=True)

    st.markdown("""
    <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.07),transparent);
                margin:2rem 0;"></div>
    <div style="font-size:0.6rem;color:rgba(255,255,255,0.3);letter-spacing:0.12em;
                text-transform:uppercase;font-weight:600;margin-bottom:1.1rem;">
        Raw Model Output
    </div>
    """, unsafe_allow_html=True)

    raw_col1, raw_col2 = st.columns([1,2], gap="large")

    with raw_col1:
        rows_html = "".join([
            f"""<div style="display:flex;justify-content:space-between;padding:0.32rem 0;
                            border-bottom:1px solid rgba(255,255,255,0.04);font-size:0.74rem;">
                    <span style="color:rgba(255,255,255,0.45);">{cls}</span>
                    <span style="font-family:'JetBrains Mono',monospace;color:rgba(255,255,255,0.65);">
                        {float(raw_probs[i]):.6f}
                    </span>
                </div>"""
            for i, cls in enumerate(class_names)
        ])
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
                    border-radius:13px;padding:1.1rem;">
            <div style="font-size:0.6rem;color:rgba(255,255,255,0.3);letter-spacing:0.1em;
                        text-transform:uppercase;margin-bottom:0.65rem;">Softmax Probabilities</div>
            {rows_html}
        </div>
        """, unsafe_allow_html=True)

    with raw_col2:
        chart_df = pd.DataFrame(
            {"Confidence (%)": (raw_probs * 100).tolist()},
            index=class_names
        )
        st.bar_chart(chart_df, height=200, use_container_width=True)


else:
    st.markdown("""
    <div style="background:rgba(0,80,255,0.04);border:1px solid rgba(0,100,255,0.1);
                border-radius:20px;padding:2.5rem 2rem;text-align:center;margin:0.5rem 0 2rem;">
        <div style="font-size:2.5rem;margin-bottom:0.75rem;opacity:0.5;">👁️</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:600;
                    color:rgba(255,255,255,0.65);margin-bottom:0.4rem;">No Image Loaded</div>
        <p style="font-size:0.82rem;color:rgba(255,255,255,0.3);max-width:360px;
                  margin:0 auto;line-height:1.75;">
            Upload a retinal fundus photograph above to begin automated grading.
        </p>
        <div style="display:flex;gap:0.6rem;justify-content:center;margin-top:1.25rem;flex-wrap:wrap;">
            <span style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);
                         border-radius:7px;padding:0.4rem 0.85rem;font-size:0.73rem;
                         color:rgba(255,255,255,0.4);">JPEG · PNG · JPG</span>
            <span style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);
                         border-radius:7px;padding:0.4rem 0.85rem;font-size:0.73rem;
                         color:rgba(255,255,255,0.4);">Color fundus photography</span>
            <span style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);
                         border-radius:7px;padding:0.4rem 0.85rem;font-size:0.73rem;
                         color:rgba(255,255,255,0.4);">Any resolution</span>
        </div>
    </div>
    <div style="font-size:0.6rem;color:rgba(255,255,255,0.3);letter-spacing:0.12em;
                text-transform:uppercase;font-weight:600;margin-bottom:1rem;">
        Classification Categories
    </div>
    """, unsafe_allow_html=True)

    preview_html = '<div style="display:flex;gap:0.75rem;flex-wrap:nowrap;">'
    for cls, m in CLASS_META.items():
        cr, cg, cb = hex_to_rgb(m["color"])
        preview_html += f"""
        <div style="flex:1;min-width:0;background:rgba({cr},{cg},{cb},0.04);
                    border:1px solid rgba({cr},{cg},{cb},0.15);
                    border-radius:14px;padding:1.25rem 0.85rem;text-align:center;">
            <div style="font-size:1.75rem;color:{m['color']};margin-bottom:0.55rem;">{m['icon']}</div>
            <div style="font-size:0.72rem;font-weight:700;color:white;
                        font-family:'Space Grotesk',sans-serif;margin-bottom:0.25rem;">{cls}</div>
            <div style="font-size:0.63rem;color:rgba(255,255,255,0.3);margin-bottom:0.55rem;">Grade {m['level']}</div>
            <div style="width:100%;height:2px;background:rgba(255,255,255,0.05);
                        border-radius:2px;overflow:hidden;margin-bottom:0.3rem;">
                <div style="height:100%;width:{m['level']*25}%;background:{m['color']};"></div>
            </div>
            <div style="font-size:0.63rem;color:{m['color']};font-weight:600;">{SEVERITY_LABELS[m['level']]}</div>
        </div>"""
    preview_html += "</div>"
    st.markdown(preview_html, unsafe_allow_html=True)


st.markdown("""
<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.07),transparent);
            margin:2.5rem 0 1.5rem;"></div>
<div style="display:flex;justify-content:space-between;align-items:flex-end;
            flex-wrap:wrap;gap:1rem;padding-bottom:1.75rem;">
    <div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:0.9rem;font-weight:600;
                    color:rgba(255,255,255,0.6);margin-bottom:0.25rem;">
            RetinaAI — Diabetic Retinopathy Severity Grading
        </div>
        <div style="font-size:0.7rem;color:rgba(255,255,255,0.25);line-height:1.6;">
            Pattern Recognition · University of Europe, Potsdam · Ashik Kirmani
        </div>
    </div>
    <div style="display:flex;gap:0.45rem;flex-wrap:wrap;">
        <span style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
                     border-radius:7px;padding:0.35rem 0.7rem;font-size:0.68rem;
                     color:rgba(255,255,255,0.3);font-family:'JetBrains Mono',monospace;">TensorFlow 2.x</span>
        <span style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
                     border-radius:7px;padding:0.35rem 0.7rem;font-size:0.68rem;
                     color:rgba(255,255,255,0.3);font-family:'JetBrains Mono',monospace;">ResNet50</span>
        <span style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
                     border-radius:7px;padding:0.35rem 0.7rem;font-size:0.68rem;
                     color:rgba(255,255,255,0.3);font-family:'JetBrains Mono',monospace;">Streamlit</span>
    </div>
</div>
<div style="background:rgba(255,160,0,0.05);border:1px solid rgba(255,160,0,0.18);
            border-radius:9px;padding:0.65rem 1rem;margin-bottom:1rem;">
    <span style="font-size:0.72rem;color:rgba(255,195,70,0.75);">
        ⚠️ Research prototype only. Not validated for clinical use. Always consult a qualified ophthalmologist.
    </span>
</div>
""", unsafe_allow_html=True)
