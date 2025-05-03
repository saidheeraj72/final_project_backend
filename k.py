#!/usr/bin/env python3
# ================================================================
#                CATTLE MUZZLE IDENTIFIER  v1.1
# ================================================================
#  • Builds an embedding DB on first run, saves it.
#  • Later runs load the DB instantly; no re-build unless forced.
#  • Type an image path → prints the nearest sample once.
#  • Optional matplotlib visualisation.
# ---------------------------------------------------------------
#  pip install ultralytics torch torchvision opencv-python pillow \
#              matplotlib numpy
# ================================================================

import os, json, pickle
import cv2, numpy as np, matplotlib.pyplot as plt
from PIL import Image
import torch, torch.nn.functional as F
from torch import nn
import torchvision
from torchvision import transforms
from ultralytics import YOLO

# --------------------------- SETTINGS ---------------------------
BASE_DIR        = '/Users/saidheeraj/Desktop/Bot/filtered_cattle_dataset'
YOLO_WEIGHTS    = 'model.pt'                 # YOLOv8 muzzle detector
FEAT_WEIGHTS    = 'feature_extractor.pth'    # ResNet-50 weights
PKL_FILE        = 'cattle_embs.pkl'          # stacked (N,2048) float32
META_FILE       = 'cattle_meta.json'         # { ids:[], paths:[] }
CONF_THRES      = 0.15                       # YOLO confidence
PADDING         = 10                         # px around bbox
SIM_THRESHOLD   = 0.75                       # “confident” if ≥
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
VISUALISE       = True                       # matplotlib pop-up?
FORCE_REBUILD   = False                      # force full rebuild?
TOP_K           = 5                          # #matches to print
# ----------------------------------------------------------------

print(f'Running on {DEVICE}')

# -------------------- 1. Load models -----------------------------
print('Loading YOLO detector …')
detector = YOLO(YOLO_WEIGHTS)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        net = torchvision.models.resnet50(weights=None)
        self.features = nn.Sequential(*list(net.children())[:-1])  # no FC
    def forward(self, x):                                         # (B,3,224,224) → (B,2048)
        return self.features(x).flatten(1)

print('Loading feature extractor …')
fe = FeatureExtractor().to(DEVICE)
fe.load_state_dict(torch.load(FEAT_WEIGHTS, map_location=DEVICE))
fe.eval()

# -------------------- 2. Preprocessing ---------------------------
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
])

# -------------------- 3. Helper functions ------------------------
def crop_first_muzzle(img_path, pad=PADDING, conf=CONF_THRES):
    """
    Detect first muzzle → returns (full_rgb, crop_rgb or None, bbox or None)
    """
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    res = detector.predict(rgb, conf=conf, verbose=False)[0]
    if len(res.boxes) == 0:
        return rgb, None, None

    i = res.boxes.conf.argmax()
    x1, y1, x2, y2 = res.boxes.xyxy[i].cpu().numpy().astype(int)

    h, w = rgb.shape[:2]
    x1, y1 = max(0,x1-pad), max(0,y1-pad)
    x2, y2 = min(w,x2+pad), min(h,y2+pad)
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return rgb, None, None
    return rgb, crop, (x1,y1,x2,y2)

@torch.inference_mode()
def embed(crop_rgb):
    """
    crop_rgb → (2048,) L2-normalised CPU tensor
    """
    x = tfm(Image.fromarray(crop_rgb)).unsqueeze(0).to(DEVICE)
    feat = fe(x)
    feat = F.normalize(feat, p=2, dim=1)
    return feat.squeeze(0).cpu()

# -------------------- 4. Build / load DB -------------------------
def build_database(base_dir):
    """
    Walk through cattle_x folders, return embeddings, ids, paths.
    """
    embeddings, ids, paths = [], [], []
    dirs = [d for d in os.listdir(base_dir)
            if d.startswith('cattle_') and os.path.isdir(os.path.join(base_dir,d))]
    print(f'Found {len(dirs)} cattle folders.')
    for cid in sorted(dirs):
        folder = os.path.join(base_dir,cid)
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg','.jpeg','.png'))]
        print(f'  • {cid}: {len(files)} images')
        for fn in files:
            path = os.path.join(folder,fn)
            _, crop, _ = crop_first_muzzle(path)
            if crop is None:
                continue
            embeddings.append(embed(crop))
            ids.append(cid)
            paths.append(path)
        print(f'    kept {len([i for i in ids if i==cid])} embeddings')
    print(f'Finished: total embeddings = {len(embeddings)}')
    return embeddings, ids, paths

def save_database(embs, ids, paths):
    arr = torch.stack(embs).numpy().astype(np.float32)
    with open(PKL_FILE,'wb') as f: pickle.dump(arr,f)
    with open(META_FILE,'w') as f: json.dump({'ids':ids,'paths':paths},f)
    print(f'Saved {len(ids)} embeddings → {PKL_FILE} / {META_FILE}')

def load_database():
    if not (os.path.exists(PKL_FILE) and os.path.exists(META_FILE)):
        return None, None, None
    with open(PKL_FILE,'rb') as f: arr = pickle.load(f)
    with open(META_FILE,'r')  as f: meta = json.load(f)
    embs = torch.from_numpy(arr).to(DEVICE)
    embs = F.normalize(embs, p=2, dim=1)     # safety
    print(f'Loaded {len(meta["ids"])} embeddings from disk.')
    return embs, meta['ids'], meta['paths']

def build_or_load_database(force=False):
    if not force:
        embs, ids, paths = load_database()
        if embs is not None:                  # success
            return embs, ids, paths
    print('Building embedding database … (first run can be slow)')
    embs, ids, paths = build_database(BASE_DIR)
    save_database(embs, ids, paths)
    return load_database()

# -------------------- 5. Identification --------------------------
@torch.inference_mode()
def identify(query_path, embs, ids, paths, top_k=TOP_K):
    """
    Prints TOP_K matches exactly once. Returns (best_id, best_sim, best_path)
    """
    full, crop, box = crop_first_muzzle(query_path)
    if crop is None:
        print('❌  No muzzle detected.')
        return None, 0.0, None

    q = embed(crop).to(embs.device)
    sims = torch.matmul(embs, q)              # cosine similarity
    topk_sim, topk_idx = torch.topk(sims, k=min(top_k, len(sims)))
    best_idx = int(topk_idx[0])
    best_sim = float(topk_sim[0])
    best_id  = ids[best_idx]
    best_path = paths[best_idx]

    # -------- single consolidated print --------
    print(f'\nQuery : {os.path.basename(query_path)}')
    print('Top matches:')
    for rank,(i,s) in enumerate(zip(topk_idx,topk_sim),1):
        cid = ids[int(i)]
        mark = '✅' if s >= SIM_THRESHOLD else '❌'
        print(f'{rank:2d}. {cid:<15}  sim={float(s):.4f}  {mark}')

    # -------- optional visualisation ----------
    if VISUALISE:
        make_vis(full, crop, box,
                 [(ids[int(i)], float(s), paths[int(i)])
                  for i,s in zip(topk_idx,topk_sim)])

    return best_id, best_sim, best_path

# -------------------- 6. Visualisation ---------------------------
def make_vis(query_img, query_crop, query_box, matches):
    """
    Show query + three best reference crops.
    matches: list[(cattle_id, sim, ref_path)] sorted
    """
    fig = plt.figure(figsize=(12,9))
    # full query
    ax0 = plt.subplot2grid((3,4),(0,0),colspan=2,rowspan=2)
    ax0.imshow(query_img); ax0.axis('off')
    if query_box:
        x1,y1,x2,y2 = query_box
        ax0.add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,color='red',lw=3))
    ax0.set_title('Query image')
    # query crop
    ax1 = plt.subplot2grid((3,4),(0,2),colspan=2,rowspan=2)
    ax1.imshow(query_crop); ax1.axis('off'); ax1.set_title('Query muzzle')
    # best 3
    for i,(cid,sim,ref_path) in enumerate(matches[:3]):
        _, ref_crop, _ = crop_first_muzzle(ref_path)
        ax = plt.subplot2grid((3,4),(2,i))
        if ref_crop is not None: ax.imshow(ref_crop)
        ax.axis('off')
        ax.set_title(f'{cid}\n{sim:.4f}',
                     color=('green' if sim>=SIM_THRESHOLD else 'red'))
    plt.suptitle('Identification result')
    plt.tight_layout(); plt.show()

# --------------------------- MAIN LOOP ---------------------------
def main():
    embs, ids, paths = build_or_load_database(FORCE_REBUILD)
    if embs is None:
        print('❌  Database could not be built.')
        return
    print('\nEnter an image path (empty line = quit):')
    while True:
        query = input('> ').strip('"').strip("'").strip()
        if query == '':
            break
        if not os.path.isfile(query):
            print('  File not found.')
            continue
        identify(query, embs, ids, paths)
    print('Bye!')

if __name__ == '__main__':
    main()