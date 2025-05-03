# ================================================================
#                   CATTLE MUZZLE IDENTIFIER – Streamlit
# ================================================================
#  ─ UI Modes ────────────────────────────────────────────────────
#    • “Add new cattle”     : Upload 1‒∞ images + enter cattle_id
#    • “Verify photo”       : Upload 1 image – returns top-K hits
#  ─ Notes ───────────────────────────────────────────────────────
#    • Heavy objects (YOLO, ResNet, DB) are cached with
#      st.cache_resource / st.cache_data – only loaded once.
#    • All original CLI functionality is kept; visualisation is
#      done with Streamlit’s image columns.
# ================================================================

import os, json, pickle, io
import cv2, numpy as np, streamlit as st
from PIL import Image
import torch, torch.nn.functional as F
from torch import nn
import torchvision
from torchvision import transforms
from ultralytics import YOLO

# -------------------- 6. Streamlit UI ---------------------------
import streamlit as st
st.set_page_config(page_title="Cattle Muzzle Identifier", layout="wide")
# --------------------------- SETTINGS ---------------------------
BASE_DIR        = 'data'                         # images are stored here
YOLO_WEIGHTS    = 'model.pt'
FEAT_WEIGHTS    = 'feature_extractor.pth'
PKL_FILE        = 'cattle_embs.pkl'
META_FILE       = 'cattle_meta.json'
CONF_THRES      = 0.15
PADDING         = 10
SIM_THRESHOLD   = 0.75
TOP_K           = 1
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
# ----------------------------------------------------------------


# -------------------- 1. Load models (cached) -------------------
@st.cache_resource(show_spinner=True)
def load_detector():
    return YOLO(YOLO_WEIGHTS)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        net = torchvision.models.resnet50(weights=None)
        self.features = nn.Sequential(*list(net.children())[:-1])
    def forward(self, x):
        return self.features(x).flatten(1)

@st.cache_resource(show_spinner=True)
def load_feature_extractor():
    fe = FeatureExtractor().to(DEVICE)
    fe.load_state_dict(torch.load(FEAT_WEIGHTS, map_location=DEVICE))
    fe.eval()
    return fe

detector = load_detector()
fe        = load_feature_extractor()

# -------------------- 2. Pre-processing -------------------------
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
])

# -------------------- 3. Helper fns -----------------------------
def crop_first_muzzle(img_bytes, pad=PADDING, conf=CONF_THRES):
    """
    img_bytes (bytes) → (full_rgb, crop_rgb or None, bbox or None)
    """
    file_bytes = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    bgr        = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb        = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    res = detector.predict(rgb, conf=conf, verbose=False)[0]
    if len(res.boxes) == 0:
        return rgb, None, None

    i = res.boxes.conf.argmax()
    x1, y1, x2, y2 = res.boxes.xyxy[i].cpu().numpy().astype(int)

    h, w = rgb.shape[:2]
    x1, y1 = max(0,x1-pad), max(0,y1-pad)
    x2, y2 = min(w,x2+pad), min(h,y2+pad)
    crop   = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return rgb, None, None
    return rgb, crop, (x1,y1,x2,y2)

@torch.inference_mode()
def embed(crop_rgb):
    x    = tfm(Image.fromarray(crop_rgb)).unsqueeze(0).to(DEVICE)
    feat = fe(x)
    feat = F.normalize(feat, p=2, dim=1)
    return feat.squeeze(0).cpu()

# -------------------- 4. DB I/O (cached) ------------------------
def _load_database_files():
    if not (os.path.exists(PKL_FILE) and os.path.exists(META_FILE)):
        return None, None, None
    with open(PKL_FILE,'rb') as f: arr  = pickle.load(f)
    with open(META_FILE,'r')  as f: meta = json.load(f)
    embs = torch.from_numpy(arr)
    embs = F.normalize(embs, p=2, dim=1)
    return embs, meta['ids'], meta['paths']

@st.cache_data(show_spinner=False)
def load_database():
    return _load_database_files()

def flush_database_cache():
    load_database.clear()

def save_database(embs, ids, paths):
    arr = torch.stack(embs).numpy().astype(np.float32)
    with open(PKL_FILE,'wb') as f: pickle.dump(arr,f)
    with open(META_FILE,'w') as f: json.dump({'ids':ids,'paths':paths},f)
    flush_database_cache()

# -------------------- 5. Identification -------------------------
@torch.inference_mode()
def identify(img_bytes, embs, ids, paths, top_k=TOP_K):
    """
    Returns:
        best_id      : ID string of the most similar cattle
        matches      : list[(id, sim, ref_path)] length ≤ top_k,
                       one line per UNIQUE id (highest-sim only)
        full, crop   : full RGB image and cropped muzzle (or None)
    """
    full, crop, _ = crop_first_muzzle(img_bytes)
    if crop is None:
        return None, [], full, crop

    # --- query embedding ----------------------------------------------------
    q     = embed(crop).to(embs.device)          # (512,)
    sims  = torch.matmul(embs, q).cpu()          # (N,)

    # --- keep only highest score per cattle_id ------------------------------
    best_per_id = {}                             # id → (sim, path)
    for i, (cid, sim) in enumerate(zip(ids, sims)):
        sim_val = float(sim)
        if (cid not in best_per_id) or (sim_val > best_per_id[cid][0]):
            best_per_id[cid] = (sim_val, paths[i])

    # --- sort & pick TOP_K ---------------------------------------------------
    sorted_items = sorted(best_per_id.items(),
                          key=lambda kv: kv[1][0], reverse=True)

    matches = [(cid, sim, pth)           # [(id, sim, ref_path), …]
               for cid, (sim, pth) in sorted_items[:top_k]]

    best_id = matches[0][0] if matches else None
    return best_id, matches, full, crop


mode = st.sidebar.radio("Choose mode", ("Verify photo", "Add new cattle"))

# ----------- load DB (or notify missing) -----------
embs, ids, paths = load_database()
if embs is None:
    st.sidebar.warning("Database empty → please add at least one cow first.")

# ===================================================
#                   MODE : VERIFY
# ===================================================
if mode == "Verify photo":
    st.header("Verify a cattle photo")

    uploaded = st.file_uploader("Upload one image",
                                type=['jpg','jpeg','png'])
    if uploaded:
        img_bytes = uploaded.read()
        if embs is None:
            st.error("Database is empty. Add cattle first.")
            st.stop()

        with st.spinner("Identifying…"):
            best_id, matches, full, crop = identify(
                img_bytes,
                embs.to(DEVICE),
                ids,
                paths,
                top_k=1               # ask for ONE best match only
            )

        # ────────── no muzzle found ──────────
        if crop is None:
            st.error("❌  No muzzle detected.")
            st.image(full, caption="Uploaded image",
                     use_column_width=True)
            st.stop()

        # ────────── decision based on similarity ──────────
        best_sim = matches[0][1] if matches else 0.0
        if best_sim >= SIM_THRESHOLD:
            best_id, best_sim, ref_path = matches[0]
            st.success(f"✅  Given sample is {best_id} "
                       f"(sim = {best_sim:.3f})")

            # show query + reference side by side
            col_q, col_r = st.columns(2)
            col_q.image(crop, caption="Query muzzle",
                        use_column_width=True)

            with open(ref_path, 'rb') as f:
                ref_bytes = f.read()
            _, ref_crop, _ = crop_first_muzzle(ref_bytes)

            col_r.image(ref_crop,
                        caption=f"Reference ({best_id})",
                        use_column_width=True)
        else:
            st.warning("⚠️  No confident match "
                       f"(best sim = {best_sim:.3f} < 0.85).")
            st.image(crop, caption="Query muzzle",
                     use_column_width=True)
# ===================================================
#               MODE : ADD NEW CATTLE
# ===================================================
else:
    st.header("Add a new cattle to the database")

    new_id = st.text_input("Cattle ID (e.g. cattle_123)")
    uploaded_list = st.file_uploader("Upload one or more images",
                                     type=['jpg','jpeg','png'],
                                     accept_multiple_files=True)

    if st.button("Add to database") and new_id and uploaded_list:
        new_embs, new_ids, new_paths = [], [], []

        st.write("Processing …")
        progress = st.progress(0)
        for n,file in enumerate(uploaded_list,1):
            img_bytes = file.read()
            _, crop, _ = crop_first_muzzle(img_bytes)
            if crop is None:
                st.warning(f"{file.name}: no muzzle found – skipped.")
                continue

            e = embed(crop)
            # save full image to BASE_DIR/new_id/
            dst_dir  = os.path.join(BASE_DIR, new_id)
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, file.name)
            with open(dst_path,'wb') as f: f.write(img_bytes)

            new_embs.append(e)
            new_ids.append(new_id)
            new_paths.append(dst_path)
            progress.progress(n/len(uploaded_list))

        if len(new_embs)==0:
            st.error("Nothing added – all uploads failed muzzle detection.")
            st.stop()

        # merge with existing DB (if any)
        if embs is not None:
            all_embs  = list(embs.cpu()) + new_embs
            all_ids   = ids + new_ids
            all_paths = paths + new_paths
        else:
            all_embs, all_ids, all_paths = new_embs, new_ids, new_paths

        save_database(all_embs, all_ids, all_paths)
        st.success(f"Added {len(new_embs)} embeddings for '{new_id}'.")
        st.balloons()