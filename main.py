import os
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import sounddevice as sd
import soundfile as sf
from enhance import enhance_audio

# ---------------- PATHS ----------------
BASE_DIR = r"C:\Users\Dell\AI Shuttering Smoother Project"
IMAGE_DIR = os.path.join(BASE_DIR, "image")
ENHANCED_DIR = os.path.join(BASE_DIR, "Enhanced")
RECORDED_FILE = os.path.join(BASE_DIR, "recorded.wav")

WELCOME_IMAGE_PATH = os.path.join(IMAGE_DIR, "welcome_image.png")
DASHBOARD_IMAGE_PATH = os.path.join(IMAGE_DIR, "dashboard_image.png")

ICON_RECORD = os.path.join(IMAGE_DIR, "record.png")
ICON_CLEAN = os.path.join(IMAGE_DIR, "clean.png")
ICON_LISTEN = os.path.join(IMAGE_DIR, "listen.png")
ICON_SELECT = os.path.join(IMAGE_DIR, "select.png")

os.makedirs(ENHANCED_DIR, exist_ok=True)
SR = 16000

# 👉 OPTIONAL (UI matching ke liye)
DASH_BG = "#5d8ed4"   # change ya hata bhi sakte ho

# ---------------- WELCOME SCREEN ----------------
def open_dashboard():
    welcome.destroy()
    dashboard_ui()

welcome = tk.Tk()
welcome.title("Shuttering Voice Smoother")
welcome.geometry("1200x700")

welcome_bg = Image.open(WELCOME_IMAGE_PATH)
welcome_label = tk.Label(welcome)
welcome_label.pack(fill="both", expand=True)

def resize_welcome(e):
    img = welcome_bg.resize((e.width, e.height))
    photo = ImageTk.PhotoImage(img)
    welcome_label.config(image=photo)
    welcome_label.image = photo

welcome.bind("<Configure>", resize_welcome)

tk.Button(
    welcome,
    text="Get Started",
    font=("Arial", 20, "bold"),
    bg="#FF7F50",
    fg="white",
    width=16,
    height=2,
    relief="flat",
    command=open_dashboard
).place(relx=0.5, rely=0.9, anchor="center")

# ---------------- DASHBOARD ----------------
def dashboard_ui():
    root = tk.Tk()
    root.title("Dashboard")
    root.geometry("1200x700")

    dash_bg = Image.open(DASHBOARD_IMAGE_PATH)
    dash_label = tk.Label(root)
    dash_label.pack(fill="both", expand=True)

    def resize_dash(e):
        img = dash_bg.resize((e.width, e.height))
        photo = ImageTk.PhotoImage(img)
        dash_label.config(image=photo)
        dash_label.image = photo

    root.bind("<Configure>", resize_dash)

    panel = tk.Frame(root, bg=DASH_BG)
    panel.place(relx=0.5, rely=0.65, anchor="center")

    # -------- ICON LOADER (NO BLACK BG) --------
    def load_icon(path, size=(85, 85)):
        img = Image.open(path).convert("RGBA")
        img = img.resize(size)
        return ImageTk.PhotoImage(img)

    icons = {
        "record": load_icon(ICON_RECORD),
        "clean": load_icon(ICON_CLEAN),
        "listen": load_icon(ICON_LISTEN),
        "select": load_icon(ICON_SELECT)
    }

    # ---------------- AUDIO FUNCTIONS ----------------
    def record():
        messagebox.showinfo("Recording", "Recording for 5 seconds...")
        audio = sd.rec(int(5 * SR), samplerate=SR, channels=1)
        sd.wait()
        sf.write(RECORDED_FILE, audio, SR)
        messagebox.showinfo("Saved", "Recording complete!")

    def enhance_recorded():
        if not os.path.exists(RECORDED_FILE):
            messagebox.showwarning("Error", "No recording found!")
            return
        out = os.path.join(ENHANCED_DIR, "Enhanced_Recorded.wav")
        enhance_audio(RECORDED_FILE, out)
        messagebox.showinfo("Done", "Recorded audio enhanced!")

    def listen_audio():
        file = filedialog.askopenfilename(
            initialdir=ENHANCED_DIR,
            filetypes=[("WAV Files", "*.wav")]
        )
        if file:
            audio, _ = sf.read(file)
            sd.play(audio, SR)

    def select_audio():
        files = filedialog.askopenfilenames(
            filetypes=[("WAV Files", "*.wav")]
        )
        for f in files:
            out = os.path.join(
                ENHANCED_DIR,
                "Enhanced_" + os.path.basename(f)
            )
            enhance_audio(f, out)
        if files:
            messagebox.showinfo("Done", "Selected files enhanced!")

    # ---------------- BUTTON CREATOR ----------------
    def create_button(col, icon, text, cmd, color):
        frame = tk.Frame(
            panel,
            bg=DASH_BG,
            bd=0,
            highlightthickness=0
        )
        frame.grid(row=0, column=col, padx=28)

        lbl = tk.Label(frame, image=icon, bg=DASH_BG)
        lbl.image = icon
        lbl.pack(pady=(0, 8))

        tk.Button(
            frame,
            text=text,
            font=("Arial", 13, "bold"),
            bg=color,
            fg="white",
            width=20,
            height=2,
            relief="flat",
            highlightthickness=0,
            cursor="hand2",
            command=cmd
        ).pack()

    # ---------------- DASHBOARD BUTTONS ----------------
    create_button(0, icons["record"], "Record Voice", record, "#4CAF50")
    create_button(1, icons["clean"], "Enhance Recorded Voice", enhance_recorded, "#2196F3")
    create_button(2, icons["listen"], "Listen Enhanced Audio", listen_audio, "#FF5722")
    create_button(3, icons["select"], "Select & Enhance Audio", select_audio, "#9C27B0")

    root.mainloop()

welcome.mainloop()
