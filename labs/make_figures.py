"""
Regenerates every figure used in DIP-Python-Introduction.pptx.
Run from the folder that should hold fig/, then rebuild the deck.
Palette matches the NDU slide template.
"""
import numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import data
import os

os.makedirs('fig', exist_ok=True)
NAVY='#0A2136'; BLUE='#0F2E4C'; GOLD='#BE8B29'; SLATE='#6B7B88'
PALE='#EDF2F6'; CREAM='#F8F3E9'; INK='#1C2B36'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'axes.edgecolor':SLATE,'text.color':INK,
                     'axes.labelcolor':INK,'xtick.color':SLATE,'ytick.color':SLATE})

def save(fig, name):
    fig.savefig(f'fig/{name}', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig); print(name)

cam, coff, moon = data.camera(), data.coffee(), data.moon()

# --- an image is an array -------------------------------------------------
patch = cam[210:218, 200:208]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4))
ax[0].imshow(cam, cmap='gray', vmin=0, vmax=255)
ax[0].add_patch(plt.Rectangle((200, 210), 8, 8, ec=GOLD, fc='none', lw=2))
ax[0].set_title('camera()   shape (512, 512)   dtype uint8', fontsize=9)
ax[1].imshow(patch, cmap='gray', vmin=0, vmax=255)
for i in range(8):
    for j in range(8):
        ax[1].text(j, i, str(patch[i, j]), ha='center', va='center', fontsize=7.5,
                   color='white' if patch[i, j] < 128 else 'black')
ax[1].set_title('an 8 x 8 corner of it', fontsize=9)
for a in ax: a.axis('off')
save(fig, 'fig_array.png')

# --- indexing conventions -------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 3.2))
H, W = 5, 8
for i in range(H):
    for j in range(W):
        ax.add_patch(plt.Rectangle((j, H-1-i), 1, 1, fc=PALE, ec='white', lw=1.5))
ax.add_patch(plt.Rectangle((5, H-3), 1, 1, fc=GOLD, ec='white', lw=1.5))
ax.text(5.5, H-2.5, 'p', ha='center', va='center', color='white', fontsize=12, fontweight='bold')
ax.annotate('', xy=(0, H-2.5), xytext=(-0.9, H-2.5), arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.6))
ax.text(-1.05, H-2.5, 'row 2', ha='right', va='center', color=BLUE, fontsize=10)
ax.annotate('', xy=(5.5, H), xytext=(5.5, H+0.85), arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.6))
ax.text(5.5, H+1.0, 'column 5', ha='center', va='bottom', color=BLUE, fontsize=10)
ax.text(W/2, -1.15, 'img[2, 5]   but   cv2.circle(img, (5, 2), ...)', ha='center',
        fontsize=11, color=INK, family='DejaVu Sans Mono')
ax.text(W/2, -1.9, 'array: row first        OpenCV point: x first', ha='center', fontsize=9.5, color=SLATE)
ax.set_xlim(-3.2, W+0.6); ax.set_ylim(-2.4, H+1.5); ax.axis('off'); ax.set_aspect('equal')
save(fig, 'fig_index.png')

# --- slicing --------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.4))
ax[0].imshow(cam, cmap='gray', vmin=0, vmax=255); ax[0].set_title('img', fontsize=9); ax[0].axis('off')
ax[0].add_patch(plt.Rectangle((180, 60), 160, 160, ec=GOLD, fc='none', lw=2.5))
ax[1].imshow(cam[60:220, 180:340], cmap='gray', vmin=0, vmax=255)
ax[1].set_title('img[60:220, 180:340]', fontsize=9, family='DejaVu Sans Mono')
for s in ax[1].spines.values(): s.set_color(GOLD); s.set_linewidth(2.5)
ax[1].set_xticks([]); ax[1].set_yticks([])
save(fig, 'fig_slice.png')

# --- BGR ------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
ax[0].imshow(coff[:, :, ::-1]); ax[0].set_title('plt.imshow(img)   — wrong', fontsize=9, color='#B03030')
ax[1].imshow(coff); ax[1].set_title('after cv2.cvtColor(..., BGR2RGB)', fontsize=9, color=BLUE)
for a in ax: a.axis('off')
save(fig, 'fig_bgr.png')

# --- negative -------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.3))
ax[0].imshow(cam, cmap='gray', vmin=0, vmax=255); ax[0].set_title('img', fontsize=9)
ax[1].imshow(255-cam, cmap='gray', vmin=0, vmax=255)
ax[1].set_title('255 - img', fontsize=9, family='DejaVu Sans Mono')
for a in ax: a.axis('off')
save(fig, 'fig_negative.png')

# --- wrap vs saturate -----------------------------------------------------
r = np.arange(0, 256, dtype=np.uint8)
fig, ax = plt.subplots(figsize=(6.2, 3.0))
ax.plot(r, r + np.uint8(100), color='#B03030', lw=2, label='img + 100      (wraps)')
ax.plot(r, cv2.add(r.reshape(-1, 1), np.full((256, 1), 100, np.uint8)).ravel(),
        color=BLUE, lw=2, label='cv2.add(img,100)  (saturates)')
ax.set_xlabel('input value'); ax.set_ylabel('output value')
ax.set_xlim(0, 255); ax.set_ylim(0, 260); ax.legend(frameon=False, fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
save(fig, 'fig_wrap.png')

# --- lookup table ---------------------------------------------------------
lut = np.clip((np.arange(256) - 80) * 255.0 / 80, 0, 255).astype(np.uint8)
fig = plt.figure(figsize=(7.4, 2.9))
a0 = fig.add_subplot(1, 3, 1); a0.plot(lut, color=GOLD, lw=2.2)
a0.set_xlim(0, 255); a0.set_ylim(0, 255); a0.set_title('the lookup table', fontsize=9)
a0.set_xlabel('r'); a0.set_ylabel('s')
a0.spines['top'].set_visible(False); a0.spines['right'].set_visible(False)
for k, (im, t) in enumerate([(moon, 'before'), (cv2.LUT(moon, lut), 'after cv2.LUT')]):
    a = fig.add_subplot(1, 3, k+2); a.imshow(im, cmap='gray', vmin=0, vmax=255)
    a.set_title(t, fontsize=9); a.axis('off')
save(fig, 'fig_lut.png')

# --- histogram ------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax[0].imshow(moon, cmap='gray', vmin=0, vmax=255); ax[0].axis('off'); ax[0].set_title('moon()', fontsize=9)
ax[1].bar(np.arange(256), np.bincount(moon.ravel(), minlength=256), width=1.0, color=BLUE)
ax[1].set_xlim(0, 255)
ax[1].set_title('np.bincount(img.ravel(), minlength=256)', fontsize=8.5, family='DejaVu Sans Mono')
ax[1].spines['top'].set_visible(False); ax[1].spines['right'].set_visible(False)
save(fig, 'fig_hist.png')

# --- show() helper output -------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.6))
for a, (im, t) in zip(ax, [(cam, 'original'), (cv2.GaussianBlur(cam, (9, 9), 0), 'Gaussian 9x9'),
                           (cv2.Canny(cam, 100, 200), 'Canny')]):
    a.imshow(im, cmap='gray', vmin=0, vmax=255); a.set_title(t, fontsize=9); a.axis('off')
save(fig, 'fig_show.png')

# --- vmin/vmax trap -------------------------------------------------------
dark = (cam * 0.35).astype(np.uint8)
fig, ax = plt.subplots(1, 2, figsize=(6.2, 3.2))
ax[0].imshow(dark, cmap='gray'); ax[0].set_title("cmap='gray'  — autoscaled", fontsize=9, color='#B03030')
ax[1].imshow(dark, cmap='gray', vmin=0, vmax=255); ax[1].set_title('vmin=0, vmax=255  — true', fontsize=9, color=BLUE)
for a in ax: a.axis('off')
save(fig, 'fig_vminmax.png')
