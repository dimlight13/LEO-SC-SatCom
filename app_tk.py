import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import argparse, os, multiprocessing, threading

# Set multiprocessing start method to prevent GUI duplication
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

sim_proc = None                 
proc_lock = threading.Lock()      

# Ensure output directory exists
os.makedirs("plot_results", exist_ok=True)

# Mapping datasets to model directories (internal use only)
dataset_model_paths = {
    "cifar10": {"rl": "./rl_model/cifar10", "post": "./post_model/cifar10"},
    "eurosat": {"rl": "./rl_model/eurosat", "post": "./post_model/eurosat"}
}

def _worker(args):
    import sys
    import os
    
    # Prevent GUI creation in subprocess
    os.environ["MPLBACKEND"] = "Agg"
    
    # Import and run the evaluation function directly
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Import only the specific function we need
    from eval_satcom_model import main_with_args
    
    # Call the function directly without triggering main
    main_with_args(args)

# --- Handlers and Helpers ---
def update_model_dirs(event=None):
    ds = combo_dataset.get()
    paths = dataset_model_paths.get(ds, {})
    rl_model_var.set(paths.get("rl", ""))
    post_model_var.set(paths.get("post", ""))

def stop_simulation():
    global sim_proc
    with proc_lock:
        if sim_proc and sim_proc.is_alive():
            sim_proc.terminate()
            sim_proc.join()
            sim_proc = None
    btn_run.config(state="normal")
    btn_stop.config(state="disabled")
    status_label.config(text="Simulation stopped.", foreground="red")

def _wait_and_finish(proc):
    proc.join()        
    exitcode = proc.exitcode    
    root.after(0, lambda: _finish_ui(exitcode))

def _finish_ui(exitcode):
    btn_run.config(state="normal")
    btn_stop.config(state="disabled")

    if exitcode == 0:            
        status_label.config(text="Evaluation complete!", foreground="green")
        _show_result_image()    
    elif exitcode == -15:            
        status_label.config(text="Simulation stopped.", foreground="red")
    else:                    
        status_label.config(text=f"Simulation failed (code {exitcode})", foreground="red")

def _show_result_image():
    dataset_name_val = combo_dataset.get()
    image_path = f"plot_results/{dataset_name_val}/psnr_vs_snr_{combo_modulation.get()}.png"
    if os.path.exists(image_path):
        img = Image.open(image_path).resize((600, 400), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        label_image.config(image=tk_img)
        label_image.image = tk_img
    else:
        status_label.config(text="Result image not found.", foreground="red")

def run_simulation():
    global sim_proc
    with proc_lock:
        if sim_proc and sim_proc.is_alive():
            status_label.config(text="Simulation already running", foreground="red")
            return

    rl_model_dir_val   = rl_model_var.get()
    post_model_dir_val = post_model_var.get()

    try:
        batch_size_val = int(entry_batch_size.get())
        snr_min_val    = int(entry_snr_min.get())
        snr_max_val    = int(entry_snr_max.get())
    except ValueError:
        status_label.config(text="Invalid numeric input.", foreground="red")
        return

    dataset_name_val = combo_dataset.get()
    if dataset_name_val.lower() == "eurosat":
        img_size_val = 64
    else:
        img_size_val = 32 

    args = argparse.Namespace(
        dataset_name        = dataset_name_val,
        save_rl_model_dir   = rl_model_dir_val,
        post_model_dir      = post_model_dir_val,
        channel_coding_mode = combo_channel_coding.get(),
        batch_size          = batch_size_val,
        input_bits_len      = 2304,
        modulation          = combo_modulation.get(),
        img_size            = img_size_val,
        M_number            = 64,
        N_number            = 16,
        snr_min             = snr_min_val,
        snr_max             = snr_max_val,
        compensation_type   = combo_compensation_type.get(),
        profile             = combo_doppler_profile.get(),
        channel_type        = 'awgn',
        doppler_type        = 'multi'
    )

    status_label.config(text="Running evaluation...", foreground="orange")
    btn_run.config(state="disabled")
    btn_stop.config(state="normal")

    proc = multiprocessing.Process(target=_worker, args=(args,), daemon=True)
    proc.start()
    with proc_lock:
        globals()["sim_proc"] = proc

    threading.Thread(target=_wait_and_finish, args=(proc,), daemon=True).start()

# --- GUI Layout ---
root = tk.Tk()
root.title("Satellite Comm. PSNR Evaluation")
root.geometry("1200x1000")
root.resizable(False, False)
root.option_add("*Font", ("Helvetica", 12))

style = ttk.Style(root)
style.theme_use('clam')
style.configure("TLabel", font=("Helvetica", 14))
style.configure("TButton", font=("Helvetica", 14))
style.configure("TEntry", font=("Helvetica", 14))
style.configure("TCombobox", font=("Helvetica", 14))
style.configure("TLabelframe.Label", font=("Helvetica", 16, "bold"))

pad_x, pad_y = 10, 10

# Dataset selector
frame_dataset = ttk.LabelFrame(root, text="Dataset")
frame_dataset.pack(padx=20, pady=10, fill="x")

ttk.Label(frame_dataset, text="Dataset:").grid(row=0, column=0, sticky="e", padx=pad_x, pady=pad_y)
combo_dataset = ttk.Combobox(
    frame_dataset,
    values=list(dataset_model_paths.keys()),
    state="readonly",
    width=18
)
combo_dataset.set("cifar10")
combo_dataset.grid(row=0, column=1, sticky="w", padx=pad_x, pady=pad_y)

rl_model_var   = tk.StringVar(value=dataset_model_paths["cifar10"]["rl"])
post_model_var = tk.StringVar(value=dataset_model_paths["cifar10"]["post"])

combo_dataset.bind("<<ComboboxSelected>>", update_model_dirs)

# Simulation parameters
frame_params = ttk.LabelFrame(root, text="Parameters")
frame_params.pack(padx=20, pady=10, fill="x")

ttk.Label(frame_params, text="Batch Size:").grid(row=0, column=0, sticky="e", padx=pad_x, pady=pad_y)
entry_batch_size = ttk.Entry(frame_params, width=20)
entry_batch_size.insert(0, "1")
entry_batch_size.grid(row=0, column=1, sticky="w", padx=pad_x, pady=pad_y)

ttk.Label(frame_params, text="Channel Coding:").grid(row=0, column=2, sticky="e", padx=pad_x, pady=pad_y)
combo_channel_coding = ttk.Combobox(
    frame_params, values=["true", "false", "both"], state="readonly", width=18
)
combo_channel_coding.set("both")
combo_channel_coding.grid(row=0, column=3, sticky="w", padx=pad_x, pady=pad_y)

ttk.Label(frame_params, text="Compensation:").grid(row=1, column=0, sticky="e", padx=pad_x, pady=pad_y)
combo_compensation_type = ttk.Combobox(
    frame_params,
    values=["lmmse", "mrc", "none"],
    state="readonly", width=18
)
combo_compensation_type.set("lmmse")
combo_compensation_type.grid(row=1, column=1, sticky="w", padx=pad_x, pady=pad_y)

ttk.Label(frame_params, text="Modulation:").grid(row=1, column=2, sticky="e", padx=pad_x, pady=pad_y)
combo_modulation = ttk.Combobox(
    frame_params,
    values=["SC_auto", "SC_none", "TN_auto", "BPSK", "QPSK", "16QAM", "64QAM", "256QAM"],
    state="readonly", width=18
)
combo_modulation.set("SC_auto")
combo_modulation.grid(row=1, column=3, sticky="w", padx=pad_x, pady=pad_y)

ttk.Label(frame_params, text="Profile:").grid(row=2, column=0, sticky="e", padx=pad_x, pady=pad_y)
combo_doppler_profile = ttk.Combobox(
    frame_params,
    values=["NTN-TDL-A", "NTN-TDL-B", "NTN-TDL-C", "NTN-TDL-D"],
    state="readonly", width=18
)
combo_doppler_profile.set("NTN-TDL-A")
combo_doppler_profile.grid(row=2, column=1, sticky="w", padx=pad_x, pady=pad_y)

ttk.Label(frame_params, text="SNR Min (dB):").grid(row=2, column=2, sticky="e", padx=pad_x, pady=pad_y)
entry_snr_min = ttk.Entry(frame_params, width=20)
entry_snr_min.insert(0, "0")
entry_snr_min.grid(row=2, column=3, sticky="w", padx=pad_x, pady=pad_y)

ttk.Label(frame_params, text="SNR Max (dB):").grid(row=2, column=4, sticky="e", padx=pad_x, pady=pad_y)
entry_snr_max = ttk.Entry(frame_params, width=20)
entry_snr_max.insert(0, "60")
entry_snr_max.grid(row=2, column=5, sticky="w", padx=pad_x, pady=pad_y)

btn_run = ttk.Button(root, text="Run PSNR Eval", command=run_simulation)
btn_stop = ttk.Button(root, text="Stop",          command=stop_simulation)
btn_stop.config(state="disabled")  

btn_run.pack(pady=20, ipadx=10, ipady=5)
btn_stop.pack(pady=0,  ipadx=10, ipady=5) 
status_label = ttk.Label(root, text="Ready", foreground="blue")
status_label.pack(pady=5)

frame_image = ttk.Frame(root)
frame_image.pack(padx=20, pady=10, fill="both", expand=True)
canvas = tk.Canvas(frame_image, width=600, height=400)
canvas.pack()
label_image = ttk.Label(canvas)
canvas.create_window(300, 200, window=label_image)

if __name__ == '__main__':
    # Set multiprocessing start method to prevent GUI duplication
    multiprocessing.set_start_method('spawn', force=True)
    root.mainloop()
