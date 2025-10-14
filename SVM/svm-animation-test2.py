import matplotlib
matplotlib.use('TkAgg')  # VS Code / Windows GUI

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from sklearn import datasets
from sklearn.svm import SVC

# --- داده ---
X, y = datasets.make_circles(n_samples=200, factor=0.4, noise=0.1, random_state=42)

# --- مدل ---
model = SVC(kernel='rbf', C=100, gamma=0.5)
model.fit(X, y)
support_vectors = model.support_vectors_

# --- شبکه ---
xx, yy = np.meshgrid(
    np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 200),
    np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 200)
)
Z_final = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# --- شکل و محورها ---
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.25)

pts = ax.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', s=80, edgecolors='k')
sv_pts = ax.scatter(support_vectors[:,0], support_vectors[:,1],
                    s=300, facecolors='none', edgecolors='gold', linewidths=2,
                    label="Support Vectors")

status_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
decision_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=12)

ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_title("🎯 SVM Interactive Animation (Final Version)")
ax.legend()
ax.grid(True)

# --- وضعیت انیمیشن ---
paused = False
frame = 0
_contourf = None
_contour_lines = []

# --- انیمیشن ---
def animate(i):
    global _contourf, _contour_lines, frame
    if paused:
        status_text.set_text(f"⏸ Paused | Frame {frame}")
        return []
    alpha = i / 50.0
    frame = i
    status_text.set_text(f"▶️ Running | Frame {frame}")

    if _contourf is not None:
        _contourf.remove()
    for c in _contour_lines:
        try:
            c.remove()
        except:
            pass
    _contour_lines = []

    Z = Z_final * alpha
    _contourf = ax.contourf(xx, yy, Z, levels=50, cmap='coolwarm', alpha=0.3)
    c0 = ax.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)
    c_minus = ax.contour(xx, yy, Z, levels=[-1], colors='blue', linestyles='--', linewidths=1.5)
    c_plus  = ax.contour(xx, yy, Z, levels=[1], colors='green', linestyles='--', linewidths=1.5)
    _contour_lines = [c0, c_minus, c_plus]

    return [_contourf, *_contour_lines, status_text, decision_text]

ani = FuncAnimation(fig, animate, frames=51, interval=100, blit=False, repeat=False)

# --- رویداد موس ---
def on_mouse_move(event):
    if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
        val = model.decision_function([[event.xdata, event.ydata]])[0]
        decision_text.set_text(f"Decision: {val:.2f}")
        fig.canvas.draw_idle()

def on_click(event):
    if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
        pred = model.predict([[event.xdata, event.ydata]])[0]
        val = model.decision_function([[event.xdata, event.ydata]])[0]
        decision_text.set_text(f"Clicked → class {pred}, val={val:.2f}")
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.canvas.mpl_connect('button_press_event', on_click)

# --- دکمه‌ها ---
ax_play = plt.axes([0.2, 0.05, 0.15, 0.075])
ax_pause = plt.axes([0.4, 0.05, 0.15, 0.075])
ax_reset = plt.axes([0.6, 0.05, 0.15, 0.075])
btn_play = Button(ax_play, '▶️ Play')
btn_pause = Button(ax_pause, '⏸ Pause')
btn_reset = Button(ax_reset, '🔄 Reset')

def play(event):
    global paused
    paused = False

def pause(event):
    global paused
    paused = True

def reset(event):
    global frame, paused
    paused = False
    frame = 0
    try:
        if ani.event_source:
            ani.event_source.stop()
            ani.frame_seq = ani.new_frame_seq()
            ani.event_source.start()
    except Exception as e:
        print("⚠️ Reset skipped:", e)

btn_play.on_clicked(play)
btn_pause.on_clicked(pause)
btn_reset.on_clicked(reset)

plt.show()
