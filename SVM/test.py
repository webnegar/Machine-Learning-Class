import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, RadioButtons
from sklearn.svm import SVC

# --- داده اولیه ---
np.random.seed(42)
X_class0 = np.random.randn(30,2) - 1.5
X_class1 = np.random.randn(30,2) + 1.5

# --- پارامترهای اولیه ---
kernel = 'rbf'
C_val = 100
gamma_val = 0.5

# --- تابع اصلی ---
def create_figure(X_init, y_init, kernel, C_val, gamma_val):
    X = X_init.copy()
    y = y_init.copy()

    fig, ax = plt.subplots(figsize=(8,8))
    plt.subplots_adjust(bottom=0.35)
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_title("🌈 Ultra-Visual SVM Trainer")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.grid(True)

    scatter_pts = None
    sv_points = None
    contours = []
    gradient_contour = None
    status_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    decision_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=12)

    paused_flag = {'paused': False}

    # --- کنترل‌ها ---
    ax_play = plt.axes([0.1, 0.05, 0.12, 0.05])
    ax_pause = plt.axes([0.25, 0.05, 0.12, 0.05])
    ax_reset = plt.axes([0.4, 0.05, 0.12, 0.05])
    btn_play = Button(ax_play, '▶️ Play')
    btn_pause = Button(ax_pause, '⏸ Pause')
    btn_reset = Button(ax_reset, '🔄 Reset')

    ax_C = plt.axes([0.6,0.05,0.3,0.03])
    slider_C = Slider(ax_C,'C',0.1,200,valinit=C_val)
    ax_gamma = plt.axes([0.6,0.01,0.3,0.03])
    slider_gamma = Slider(ax_gamma,'Gamma',0.01,5,valinit=gamma_val)

    ax_kernel = plt.axes([0.05,0.7,0.12,0.18])
    radio_kernel = RadioButtons(ax_kernel, ('linear','rbf','poly','sigmoid'), active=1 if kernel=='rbf' else 0)

    def train_and_plot():
        nonlocal scatter_pts, sv_points, contours, gradient_contour
        if scatter_pts: scatter_pts.remove()
        if sv_points: sv_points.remove()
        if gradient_contour: 
            for c in gradient_contour.collections: c.remove()
        for c in contours:
            for coll in getattr(c,'collections',[]):
                coll.remove()
        contours.clear()

        if len(X)<2: return

        X_np = np.array(X)
        y_np = np.array(y)
        model = SVC(kernel=kernel, C=C_val, gamma=gamma_val, probability=True)
        model.fit(X_np, y_np)

        xx, yy = np.meshgrid(np.linspace(-5,5,300), np.linspace(-5,5,300))
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        Z_prob = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1].reshape(xx.shape)

        # --- مرز تصمیم gradient ---
        gradient_contour = ax.contourf(xx,yy,Z_prob,levels=100,cmap='coolwarm', alpha=0.3)

        # --- خطوط decision boundary و margin ---
        c0 = ax.contour(xx,yy,Z,levels=[0],colors='k',linewidths=2)
        c1 = ax.contour(xx,yy,Z,levels=[-1],colors='blue',linestyles='--',linewidths=1.5)
        c2 = ax.contour(xx,yy,Z,levels=[1],colors='green',linestyles='--',linewidths=1.5)
        contours.extend([c0,c1,c2])

        # --- نقاط ---
        scatter_pts = ax.scatter(X_np[:,0], X_np[:,1], c=y_np, cmap='coolwarm', s=90, edgecolors='k')
        sv = model.support_vectors_
        sv_points = ax.scatter(sv[:,0], sv[:,1], s=300, facecolors='none', edgecolors='gold', linewidths=2.5)
        plt.draw()

    def animate(i):
        if paused_flag['paused']:
            status_text.set_text(f"⏸ Paused | Frame {i}")
            return []
        status_text.set_text(f"▶️ Running | Frame {i}")
        train_and_plot()
        return []

    ani = FuncAnimation(fig, animate, frames=51, interval=50, blit=False, repeat=True)

    # --- کنترل‌ها ---
    btn_play.on_clicked(lambda e: paused_flag.update({'paused': False}))
    btn_pause.on_clicked(lambda e: paused_flag.update({'paused': True}))

    def reset_callback(e):
        plt.close(fig)
        create_figure(
            np.vstack([X_class0,X_class1]).tolist(),
            [0]*len(X_class0)+[1]*len(X_class1),
            radio_kernel.value_selected,
            slider_C.val,
            slider_gamma.val
        )
    btn_reset.on_clicked(reset_callback)

    # --- موس و کلیک ---
    def on_mouse_move(event):
        if not event.inaxes or len(X)<2:
            decision_text.set_text('')
            plt.draw()
            return
        try:
            model = SVC(kernel=kernel, C=C_val, gamma=gamma_val)
            model.fit(np.array(X), np.array(y))
            val = model.decision_function([[event.xdata,event.ydata]])[0]
            decision_text.set_text(f"Decision: {val:.2f}")
        except: decision_text.set_text('')
        plt.draw()

    def onclick(event):
        if not event.inaxes: return
        if event.button==1: X.append([event.xdata,event.ydata]); y.append(0)
        elif event.button==3: X.append([event.xdata,event.ydata]); y.append(1)
        train_and_plot()

    fig.canvas.mpl_connect('motion_notify_event',on_mouse_move)
    fig.canvas.mpl_connect('button_press_event',onclick)

    train_and_plot()
    plt.show()

# --- اجرای اولیه ---
create_figure(
    np.vstack([X_class0,X_class1]).tolist(),
    [0]*len(X_class0)+[1]*len(X_class1),
    kernel,
    C_val,
    gamma_val
)
