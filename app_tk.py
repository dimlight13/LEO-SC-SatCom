import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import argparse
import os
from eval_ber_psnr import main_with_args

def run_simulation():
    try:
        batch_size_val = int(entry_batch_size.get())
        doppler_mode_val = combo_doppler_mode.get()
        channel_type_val = combo_channel_type.get()
        compensation_type_val = combo_compensation_type.get()
        modulation_val = combo_modulation.get()
        experiment_type = experiment_option.get()  # "BER" or "PSNR"

        channel_coding_val = combo_channel_coding.get()  # "true", "false", "both"

        snr_min_val = int(entry_snr_min.get())
        snr_max_val = int(entry_snr_max.get())

        if experiment_type == "PSNR":
            eval_type_val = "psnr"
        else:
            eval_type_val = "ber"

        args = argparse.Namespace(
            eval_type=eval_type_val,
            channel_coding_mode=channel_coding_val,
            size=32,
            epoch=400,
            lr_max=1e-3,
            lr_min=5e-5,
            cycle_length=10,
            batch_size=batch_size_val,
            embedding_dim=64,
            num_embeddings=512,
            commitment_cost=0.25,
            decay=0.99,
            n_res_block=2,
            n_res_channel=32,
            num_samples=5,
            input_bits_len=2304,
            random_seed=128,
            save_interval=10,
            modulation=modulation_val,
            channel_type=channel_type_val,
            doppler_type=doppler_mode_val,
            snr_min=snr_min_val,
            snr_max=snr_max_val,
            compensation_type=compensation_type_val,
            M_number=64,
            N_number=16,
        )

        status_label.config(text="Running simulation...", foreground="orange")
        root.update_idletasks()

        main_with_args(args)

        # 결과 이미지 파일명 결정
        if eval_type_val == "psnr":
            image_filename = f"plot_results/{channel_type_val}_psnr_result_with_and_without_channel_coding.png"
        else:
            image_filename = f"plot_results/{channel_type_val}_ber_result_with_and_without_channel_coding.png"

        status_label.config(text="Simulation complete!", foreground="green")
        root.update_idletasks()

        if os.path.exists(image_filename):
            img = Image.open(image_filename)
            max_width, max_height = 600, 400
            img = img.resize((max_width, max_height), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(img)
            label_image.config(image=tk_img)
            label_image.image = tk_img
        else:
            status_label.config(text="Plot image not found. Check main function output.", foreground="red")

    except ValueError:
        status_label.config(text="Check number.", foreground="red")
    except Exception as e:
        status_label.config(text=f"Error occur: {e}", foreground="red")

# ============== GUI 생성 ==============
root = tk.Tk()
root.title("Doppler simulation")
root.geometry("1200x1000")
root.resizable(False, False)

default_font = ("Helvetica", 12)
root.option_add("*Font", default_font)

style = ttk.Style(root)
style.theme_use('clam')
style.configure("TLabel", font=("Helvetica", 14))
style.configure("TButton", font=("Helvetica", 14))
style.configure("TEntry", font=("Helvetica", 14))
style.configure("TCombobox", font=("Helvetica", 14))
style.configure("TCheckbutton", font=("Helvetica", 14))
style.configure("TLabelframe.Label", font=("Helvetica", 16, "bold"))

pad_x = 10
pad_y = 10

frame_experiment = ttk.LabelFrame(root, text="Select simulation environments")
frame_experiment.pack(padx=20, pady=10, fill="x")

experiment_option = tk.StringVar()
experiment_option.set("BER")

radio_ber = ttk.Radiobutton(frame_experiment, text="evaluate BER", variable=experiment_option, value="BER")
radio_psnr = ttk.Radiobutton(frame_experiment, text="evaluate PSNR", variable=experiment_option, value="PSNR")
radio_ber.pack(side="left", padx=pad_x, pady=pad_y)
radio_psnr.pack(side="left", padx=pad_x, pady=pad_y)

frame_params = ttk.LabelFrame(root, text="parameters")
frame_params.pack(padx=20, pady=20, fill="x")

frame_params.columnconfigure(0, weight=1, pad=pad_x)
frame_params.columnconfigure(1, weight=2, pad=pad_x)

label_channel_coding = ttk.Label(frame_params, text="Channel Coding:")
label_channel_coding.grid(row=0, column=2, sticky="e", padx=pad_x, pady=pad_y)
combo_channel_coding = ttk.Combobox(
    frame_params,
    values=["true", "false", "both"],
    state="readonly",
    width=18
)
combo_channel_coding.set("both")  # 기본값
combo_channel_coding.grid(row=0, column=3, sticky="w", padx=pad_x, pady=pad_y)

label_batch_size = ttk.Label(frame_params, text="batch size:")
label_batch_size.grid(row=0, column=0, sticky="e", padx=pad_x, pady=pad_y)
entry_batch_size = ttk.Entry(frame_params, width=20)
entry_batch_size.insert(0, "1")
entry_batch_size.grid(row=0, column=1, sticky="w", padx=pad_x, pady=pad_y)

label_compensation_type = ttk.Label(frame_params, text="Compensation type:")
label_compensation_type.grid(row=1, column=0, sticky="e", padx=pad_x, pady=pad_y)
combo_compensation_type = ttk.Combobox(
    frame_params,
    values=["lmmse", "single_tap", "mrc", "mrc_low_complexity", "none"],
    state="readonly",
    width=18
)
combo_compensation_type.set("lmmse")
combo_compensation_type.grid(row=1, column=1, sticky="w", padx=pad_x, pady=pad_y)

label_doppler_mode = ttk.Label(frame_params, text="Doppler type:")
label_doppler_mode.grid(row=2, column=0, sticky="e", padx=pad_x, pady=pad_y)
combo_doppler_mode = ttk.Combobox(
    frame_params,
    values=["multi", "none"],
    state="readonly",
    width=18
)
combo_doppler_mode.set("none")
combo_doppler_mode.grid(row=2, column=1, sticky="w", padx=pad_x, pady=pad_y)

label_channel_type = ttk.Label(frame_params, text="Channel Type:")
label_channel_type.grid(row=3, column=0, sticky="e", padx=pad_x, pady=pad_y)
combo_channel_type = ttk.Combobox(
    frame_params,
    values=["awgn", "rayleigh", "rician"],
    state="readonly",
    width=18
)
combo_channel_type.set("awgn")
combo_channel_type.grid(row=3, column=1, sticky="w", padx=pad_x, pady=pad_y)

label_modulation = ttk.Label(frame_params, text="Modulation Scheme:")
label_modulation.grid(row=4, column=0, sticky="e", padx=pad_x, pady=pad_y)
combo_modulation = ttk.Combobox(
    frame_params,
    values=["auto", "BPSK", "QPSK", "16QAM", "64QAM", "256QAM"],
    state="readonly",
    width=18
)
combo_modulation.set("auto")
combo_modulation.grid(row=4, column=1, sticky="w", padx=pad_x, pady=pad_y)

label_snr_min = ttk.Label(frame_params, text="SNR min (dB):")
label_snr_min.grid(row=1, column=2, sticky="e", padx=pad_x, pady=pad_y)
entry_snr_min = ttk.Entry(frame_params, width=20)
entry_snr_min.insert(0, "-10")
entry_snr_min.grid(row=1, column=3, sticky="w", padx=pad_x, pady=pad_y)

label_snr_max = ttk.Label(frame_params, text="SNR max (dB):")
label_snr_max.grid(row=1, column=4, sticky="e", padx=pad_x, pady=pad_y)
entry_snr_max = ttk.Entry(frame_params, width=20)
entry_snr_max.insert(0, "30")
entry_snr_max.grid(row=1, column=5, sticky="w", padx=pad_x, pady=pad_y)

btn_run = ttk.Button(root, text="Run simulation", command=run_simulation)
btn_run.pack(pady=10, ipadx=10, ipady=5)

status_label = ttk.Label(root, text="prepare", foreground="blue", font=("Helvetica", 14))
status_label.pack(pady=10)

frame_image = ttk.Frame(root)
frame_image.pack(padx=20, pady=20, fill="both", expand=True)

canvas = tk.Canvas(frame_image, width=600, height=400)
canvas.pack()

label_image = ttk.Label(canvas)
canvas.create_window(300, 200, window=label_image)

root.mainloop()
