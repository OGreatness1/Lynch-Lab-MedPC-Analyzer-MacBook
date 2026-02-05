import os
import re
import sys
import glob
import logging
import traceback
import json
import threading
import queue
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import customtkinter as ctk
from tkinter import filedialog, messagebox
from datetime import datetime

# ============================================================================
# 1. SETUP & CONFIGURATION
# ============================================================================

# Theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# Log Path (User Home)
user_home = os.path.expanduser("~")
log_file_path = os.path.join(user_home, 'medpc_processing_log.txt')

# Configure Logging
logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Thread-safe queue for GUI updates
log_queue = queue.Queue()

# ============================================================================
# 2. UI CLASS (MODERN DASHBOARD)
# ============================================================================

class MedPCApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Lynch Lab MedPC Analyzer")
        self.geometry("900x650")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # 1. Header
        self.header = ctk.CTkFrame(self, corner_radius=0, fg_color="#1F538D")
        self.header.grid(row=0, column=0, sticky="ew")
        self.title_lbl = ctk.CTkLabel(self.header, text="Lynch Lab MedPC Analyzer", 
                                      font=ctk.CTkFont(size=20, weight="bold"), text_color="white")
        self.title_lbl.pack(padx=20, pady=15)

        # 2. Input Area
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.grid(row=1, column=0, padx=20, pady=20, sticky="ew")
        self.input_frame.grid_columnconfigure(1, weight=1)

        # Folder
        ctk.CTkLabel(self.input_frame, text="Data Folder:").grid(row=0, column=0, padx=15, pady=15, sticky="w")
        self.folder_entry = ctk.CTkEntry(self.input_frame, placeholder_text="Path to raw files...")
        self.folder_entry.grid(row=0, column=1, padx=10, pady=15, sticky="ew")
        ctk.CTkButton(self.input_frame, text="Browse", width=100, command=self.select_folder).grid(row=0, column=2, padx=15)

        # ID List
        ctk.CTkLabel(self.input_frame, text="ID List (.txt):").grid(row=1, column=0, padx=15, pady=15, sticky="w")
        self.id_entry = ctk.CTkEntry(self.input_frame, placeholder_text="Path to allowed IDs...")
        self.id_entry.grid(row=1, column=1, padx=10, pady=15, sticky="ew")
        ctk.CTkButton(self.input_frame, text="Browse", width=100, command=self.select_id).grid(row=1, column=2, padx=15)

        # 3. Action Bar
        self.action_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.action_frame.grid(row=2, column=0, padx=20, sticky="ew")
        
        self.run_btn = ctk.CTkButton(self.action_frame, text="START PROCESSING", height=50, 
                                     fg_color="#2CC985", hover_color="#229A65",
                                     font=ctk.CTkFont(size=16, weight="bold"), 
                                     command=self.start_thread)
        self.run_btn.pack(fill="x")

        # 4. Log Console
        ctk.CTkLabel(self, text="Live Process Log:", anchor="w").grid(row=3, column=0, padx=20, pady=(20, 0), sticky="w")
        self.log_box = ctk.CTkTextbox(self, font=("Consolas", 12), state="disabled")
        self.log_box.grid(row=4, column=0, padx=20, pady=(5, 20), sticky="nsew")

        # Start Log Poller
        self.after(100, self.poll_log_queue)

    def select_folder(self):
        d = filedialog.askdirectory()
        if d: 
            self.folder_entry.delete(0, "end")
            self.folder_entry.insert(0, d)

    def select_id(self):
        f = filedialog.askopenfilename(filetypes=[("Text", "*.txt")])
        if f: 
            self.id_entry.delete(0, "end")
            self.id_entry.insert(0, f)

    def poll_log_queue(self):
        while not log_queue.empty():
            msg = log_queue.get()
            self.log_box.configure(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.configure(state="disabled")
            self.log_box.see("end")
        self.after(100, self.poll_log_queue)

    def start_thread(self):
        folder = self.folder_entry.get()
        id_file = self.id_entry.get()
        
        if not folder or not id_file:
            messagebox.showwarning("Input Error", "Please select both Folder and ID List.")
            return

        self.run_btn.configure(state="disabled", text="RUNNING...")
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

        threading.Thread(target=self.run_analysis, args=(folder, id_file), daemon=True).start()

    def run_analysis(self, folder, id_file):
        try:
            analyze_dataset(folder, id_file)
            log_queue.put("\n✅ DONE! All files processed.")
            messagebox.showinfo("Success", "Processing Complete!")
        except Exception as e:
            log_queue.put(f"\n❌ CRITICAL ERROR: {e}")
            log_queue.put(traceback.format_exc())
        finally:
            self.run_btn.configure(state="normal", text="START PROCESSING")

# ============================================================================
# 3. DEFAULTS & DEFINITIONS
# ============================================================================

METADATA_KEYS = ["start date", "end date", "subject", "msn", "experiment", 
                 "group", "box", "start time", "end time", "time unit"]

def log_msg(msg):
    log_queue.put(msg)
    logging.info(msg)

DEFAULT_MSN_PATTERNS = {
    "RAT - FR20": ["fr20"], "RAT - FR40": ["fr40"], "RAT - FR FOOD": ["frfood", "2025newfrfoodtrain", "newfrfoodtrain"],
    "RAT - FENTANYL FR40": ["fentanyl1secfr40ldesd"], "RAT - FENTANYL FR40 (FOOD RESTRICT)": ["fentanyl1secfr40ldfoodrestrictesd"],
    "RAT - INTERMITTENT ACCESS": ["newintermittentaccessldesd"], "RAT - WITHDRAWAL": ["withdrawalldesd"], 
    "RAT - INT ACCESS (FOOD RESTRICT)": ["newintermittentaccessldfoodrestrictesd"],
    "RAT - EXTINCTION FR20": ["g136afr20"], "RAT - EXTINCTION PROCAINE": ["g136aprocaine"], 
    "RAT - EXTINCTION BOXES": ["g136aboxes"], "RAT - REINSTATEMENT": ["g136areinstate"],
    "RAT - CUE RELAPSE A": ["g138acuerelapse7hrpreathold", "g138a"], "RAT - CUE RELAPSE B": ["g138bcuerelapse7hrpretxhold", "g138b", "cuerelapse"],
    "RAT - PR COCAINE": ["prcocaine"], "RAT - PR FENTANYL": ["prfent", "prfentesd"],
    "MOUSE - EXTENDED ACCESS": ["mouseextendedaccess", "mouseextendedaccessv2", "mouseintera", "mouseintermittentaccess"], 
    "MOUSE - PR": ["mousepr", "mouse pr"], "MOUSE - FR1": ["mousefr1", "mouse fr1"]
}

# Variable Definitions
map_rat_fr = {"infusions": ["I"], "active_presses": ["R"], "inactive_presses": [], "infusion_timestamps": ["J"], "active_timestamps": ["K"], "inactive_timestamps": [], "breakpoint": "K", "duration": "Z", "extra_vars": ["W"]}
map_rat_int = {"infusions": ["I"], "active_presses": ["U"], "inactive_presses": ["R"], "infusion_timestamps": ["F", "G"], "active_timestamps": ["L", "P"], "inactive_timestamps": ["M", "D"], "breakpoint": "Q", "duration": "Z", "extra_vars": ["W"]}
map_rat_fent = {"infusions": ["I"], "active_presses": ["R"], "inactive_presses": ["A"], "infusion_timestamps": ["J"], "active_timestamps": ["J"], "inactive_timestamps": ["J"], "breakpoint": None, "duration": "Z", "special_processing": "J_ARRAY_HOURLY", "extra_vars": ["W"]}
map_rat_cue = {"infusions": ["N"], "active_presses": ["R"], "inactive_presses": ["M"], "infusion_timestamps": ["N"], "active_timestamps": ["A", "D", "F", "G"], "inactive_timestamps": ["H", "I", "J", "K"], "breakpoint": "Q", "duration": "Z", "extra_vars": ["W"]}
map_rat_pr = {"infusions": ["I"], "active_presses": ["R"], "inactive_presses": ["A"], "infusion_timestamps": ["J"], "active_timestamps": ["J"], "inactive_timestamps": ["J"], "breakpoint": "V", "duration": "Z", "extra_vars": ["W"]}
map_rat_ext = {"infusions": ["N"], "active_presses": ["U", "M"], "inactive_presses": ["P"], "infusion_timestamps": [], "active_timestamps": [], "inactive_timestamps": [], "breakpoint": None, "duration": "Z", "special_extraction": "EXTINCTION_DETAIL"}
map_mouse = {"infusions": ["R"], "active_presses": ["A"], "inactive_presses": ["I"], "infusion_timestamps": ["G"], "active_timestamps": [], "inactive_timestamps": [], "breakpoint": None, "duration": "Z", "extra_vars": ["L", "G"]}
map_mouse_pr = {"infusions": ["R"], "active_presses": ["A"], "inactive_presses": ["I"], "infusion_timestamps": [], "active_timestamps": [], "inactive_timestamps": [], "breakpoint": "V", "duration": "Z", "extra_vars": ["L", "G"]}

DEFAULT_VARIABLE_MAPPINGS = {
    "RAT - FR20": map_rat_fr, "RAT - FR40": map_rat_fr, "RAT - FR FOOD": map_rat_fr,
    "RAT - FENTANYL FR40": map_rat_fent, "RAT - FENTANYL FR40 (FOOD RESTRICT)": map_rat_fent,
    "RAT - INTERMITTENT ACCESS": map_rat_int, "RAT - WITHDRAWAL": map_rat_int, "RAT - INT ACCESS (FOOD RESTRICT)": map_rat_int,
    "RAT - EXTINCTION FR20": map_rat_ext, "RAT - EXTINCTION PROCAINE": map_rat_ext, "RAT - EXTINCTION BOXES": map_rat_ext, "RAT - REINSTATEMENT": map_rat_ext,
    "RAT - CUE RELAPSE A": map_rat_cue, "RAT - CUE RELAPSE B": map_rat_cue,
    "RAT - PR COCAINE": map_rat_pr, "RAT - PR FENTANYL": map_rat_pr,
    "MOUSE - EXTENDED ACCESS": map_mouse, "MOUSE - PR": map_mouse_pr, "MOUSE - FR1": map_mouse
}

# ============================================================================
# 4. HELPER FUNCTIONS
# ============================================================================

def canonicalize_id(subject_id):
    if pd.isna(subject_id) or str(subject_id).strip() == "": return ""
    return re.sub(r"^O", "0", str(subject_id).strip().upper())

def extract_gender(subject_id):
    if not subject_id: return "Unknown"
    last = str(subject_id).strip().lower()[-1]
    return "Female" if last == "f" else "Male" if last == "m" else "Unknown"

def normalize_msn(msn):
    if pd.isna(msn): return ""
    return re.sub(r"[^\w]", "", str(msn)).lower()

def extract_array_data(lines, start_idx):
    """ Parses arrays like '1: 10.5 20.2' """
    data = []
    i = start_idx + 1
    while i < len(lines):
        line = lines[i].strip()
        if re.match(r"^[a-zA-Z]:", line) or (":" in line and line.split(":")[0].lower() in METADATA_KEYS): break
        if re.match(r"^\d+:", line):
            try:
                values = [float(x) for x in line.split(":", 1)[1].split()]
                data.extend(values)
            except ValueError: pass
        i += 1
    return data

def calculate_duration(arrays, scalars, key="Z", time_unit="seconds"):
    # 1. Try Scalar first (Z: 3600)
    if key in scalars: return scalars[key]
    # 2. Try Array (Z: 0: 3600)
    vals = arrays.get(key, [])
    if len(vals) >= 3: return vals[0]*3600 + vals[1]*60 + vals[2]
    if len(vals) >= 1: return vals[0] * 60 if str(time_unit).lower() == "minutes" else vals[0]
    return 0

def get_data_quality_flags(row):
    flags = []
    if row['infusions'] == 0 and row['duration_sec'] > 3600: flags.append("No infusions despite long session")
    if row['active_presses'] == 0 and row['infusions'] > 0: flags.append("Infusions without active presses")
    if row['duration_sec'] == 0: flags.append("Missing or zero duration")
    return "; ".join(flags) if flags else "OK"

# ============================================================================
# 5. VISUALIZATION ENGINE
# ============================================================================

def generate_visualizations(sess, hr, daily, avgs, output_dir, prog_safe):
    sns.set_style("whitegrid")
    sns.set_palette("Set1")
    
    def save_plot(filename):
        try:
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()
        except Exception as e:
            log_msg(f"⚠️ Plot Error ({filename}): {e}")

    # A. HOURLY
    if not hr.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=hr, x="hour", y="infusion_events", hue="canonical_subject", errorbar=None)
        plt.title("Infusions by Hour")
        save_plot(f"{prog_safe}_01_hourly_infusions.png")

        plt.figure(figsize=(12, 6))
        sns.barplot(data=hr, x="hour", y="active_events", hue="canonical_subject", errorbar=None)
        plt.title("Active Presses by Hour")
        save_plot(f"{prog_safe}_02_hourly_active.png")

        if hr["inactive_events"].sum() > 0:
            plt.figure(figsize=(12, 6))
            sns.barplot(data=hr, x="hour", y="inactive_events", hue="canonical_subject", errorbar=None)
            plt.title("Inactive Presses by Hour")
            save_plot(f"{prog_safe}_03_hourly_inactive.png")
            
        for gender in ["Male", "Female"]:
            gen_data = hr[hr["gender"] == gender]
            if not gen_data.empty:
                plt.figure(figsize=(10, 6))
                sns.barplot(data=gen_data, x="hour", y="active_events", hue="canonical_subject", errorbar=None)
                plt.title(f"{gender} | Hourly Active Presses")
                save_plot(f"{prog_safe}_10_{gender.lower()}_hourly.png")

    # B. DAILY
    if not daily.empty:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=daily, x="start_date", y="total_infusions", hue="canonical_subject", marker="o")
        plt.xticks(rotation=45)
        plt.title("Daily Trend: Infusions")
        save_plot(f"{prog_safe}_04_daily_infusions.png")
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=daily, x="start_date", y="total_active_presses", hue="canonical_subject", marker="o")
        plt.xticks(rotation=45)
        plt.title("Daily Trend: Active Presses")
        save_plot(f"{prog_safe}_05_daily_active.png")

        if daily["total_inactive_presses"].sum() > 0:
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=daily, x="start_date", y="total_inactive_presses", hue="canonical_subject", marker="o")
            plt.xticks(rotation=45)
            plt.title("Daily Trend: Inactive Presses")
            save_plot(f"{prog_safe}_06_daily_inactive.png")

        plt.figure(figsize=(14, 8))
        sns.lineplot(data=daily, x="start_date", y="total_active_presses", hue="canonical_subject", style="gender", markers=True)
        plt.xticks(rotation=45)
        plt.title("Trajectory: Active Presses by Subject")
        save_plot(f"{prog_safe}_12_trajectory_active.png")

    # C. AVERAGES
    if not avgs.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=avgs, x="canonical_subject", y="infusions_mean", hue="canonical_subject", dodge=False)
        plt.title("Mean Infusions per Session")
        save_plot(f"{prog_safe}_07_avg_infusions.png")
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=avgs, x="canonical_subject", y="active_presses_mean", hue="canonical_subject", dodge=False)
        plt.title("Mean Active Presses per Session")
        save_plot(f"{prog_safe}_08_avg_active.png")

    # D. SCATTER & BOX
    if not sess.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=sess, x="active_presses", y="infusions", hue="gender", size="duration_sec", alpha=0.7)
        plt.title("Efficiency: Active vs Infusions")
        save_plot(f"{prog_safe}_06b_efficiency.png")
    
    if not daily.empty:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=daily, x="gender", y="total_infusions", hue="gender")
        sns.stripplot(data=daily, x="gender", y="total_infusions", color='black', alpha=0.3)
        plt.title("Distribution: Daily Infusions by Gender")
        save_plot(f"{prog_safe}_11_boxplot_infusions.png")

# ============================================================================
# 6. DATA PROCESSING (FIXED SCALAR LOGIC)
# ============================================================================

def analyze_dataset(folder_path, id_list_path):
    log_msg(f"📂 Scanning: {folder_path}")

    # 1. Load Settings
    msn_patterns = DEFAULT_MSN_PATTERNS.copy()
    variable_mappings = DEFAULT_VARIABLE_MAPPINGS.copy()
    
    if os.path.exists(os.path.join(folder_path, "Settings.json")):
        try:
            with open(os.path.join(folder_path, "Settings.json"), 'r') as f:
                s = json.load(f)
                if "msn_patterns" in s: msn_patterns = s["msn_patterns"]
                if "variable_mappings" in s: variable_mappings = s["variable_mappings"]
            log_msg("⚙️ Loaded Settings.json")
        except: log_msg("⚠️ Settings.json error. Using defaults.")

    # 2. Load IDs
    with open(id_list_path, 'r') as f:
        allowed_ids = {line.strip() for line in f if line.strip()}
    allowed_ids_canon = {canonicalize_id(x) for x in allowed_ids}
    log_msg(f"ℹ️ Allowed IDs: {len(allowed_ids)}")

    # 3. Process Files
    files = glob.glob(os.path.join(folder_path, "!*")) + glob.glob(os.path.join(folder_path, "Data", "!*"))
    all_sess, all_hr = [], []
    found_ids_canon = set() # Track found IDs
    
    log_msg(f"🔎 Found {len(files)} files. Starting parse...")

    for idx, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
        
        # Split Sessions
        raw_sessions = re.split(r"(Start Date:)", content)
        sessions = [raw_sessions[i] + raw_sessions[i+1] for i in range(1, len(raw_sessions), 2) if i+1 < len(raw_sessions)]
        
        for sess_block in sessions:
            lines = sess_block.splitlines()
            if len(lines) < 5: continue
            
            # PARSE SCALARS FIRST (THE FIX)
            meta, scalars = {}, {}
            for line in lines:
                if ":" in line:
                    k, v = line.split(":", 1)
                    k_clean, v_clean = k.strip().lower(), v.strip()
                    if k_clean in METADATA_KEYS: meta[k_clean] = v_clean
                    if k_clean == "msn": meta["msn"] = v_clean
                    # Capture scalars like "A: 10" or "R: 500"
                    if len(k_clean) == 1 and k_clean.isalpha():
                        try: scalars[k_clean.upper()] = float(v_clean)
                        except: pass

            if "subject" not in meta: continue
            canon = canonicalize_id(meta["subject"])
            
            if canon not in allowed_ids_canon: continue
            
            # Mark ID as Found
            found_ids_canon.add(canon)

            # Determine Program
            msn_norm = normalize_msn(meta.get("msn", ""))
            prog = "Unknown"
            for p_name, patterns in msn_patterns.items():
                if any(pat in msn_norm for pat in patterns):
                    prog = p_name
                    break
            if prog == "Unknown": continue
            
            # Log Progress
            log_msg(f"   Processing: {filename} -> {prog}")

            mapping = variable_mappings.get(prog, {})
            
            # Parse Arrays
            arrays = {}
            for i, line in enumerate(lines):
                if re.match(r"^[A-Z]:", line.strip()):
                    arrays[line.strip()[0]] = extract_array_data(lines, i)

            # --- HYBRID SUMMATION LOGIC (MATCHES R) ---
            # Priority: 1. Scalar Value (if exists) -> 2. Sum of Array (if scalar 0 or missing)
            def get_val(keys):
                total = 0
                for k in keys:
                    # Check Scalar First
                    if k in scalars and scalars[k] > 0:
                        total += scalars[k]
                    # Fallback to Array Sum
                    elif k in arrays:
                        total += sum(arrays[k])
                return total

            inf = get_val(mapping.get("infusions", []))
            act = get_val(mapping.get("active_presses", []))
            inact = get_val(mapping.get("inactive_presses", []))
            dur = calculate_duration(arrays, scalars, mapping.get("duration", "Z"), meta.get("time unit", "seconds"))
            
            bp = 0
            bp_key = mapping.get("breakpoint")
            if bp_key and bp_key in arrays and inf > 0:
                bp_vals = arrays[bp_key]
                if len(bp_vals) >= int(inf): bp = bp_vals[int(inf)-1]

            row = {
                "source_file": filename, "subject": meta["subject"], "canonical_subject": canon, 
                "gender": extract_gender(meta["subject"]),
                "start_date": pd.to_datetime(meta["start date"], errors='coerce'), "program_name": prog,
                "infusions": inf, "active_presses": act, "inactive_presses": inact, 
                "duration_sec": dur, "efficiency_ratio": inf/act if act > 0 else 0
            }
            row["data_quality_flag"] = get_data_quality_flags(row)
            
            # Extract Extras
            for extra in mapping.get("extra_vars", []): row[f"Value_{extra}"] = get_val([extra])
            if mapping.get("special_extraction") == "EXTINCTION_DETAIL":
                for l in list("UMABCDEFGHIJKL"): row[f"Response_{l}"] = get_val([l])
            
            all_sess.append(row)

            # Hourly Logic
            if mapping.get("special_processing") == "J_ARRAY_HOURLY" and "J" in arrays:
                j = arrays["J"]
                for i in range(0, len(j), 7):
                    if i+6 < len(j):
                        all_hr.append({"canonical_subject": canon, "gender": row["gender"], "program_name": prog, 
                                       "hour": int(j[i]), "infusion_events": j[i+2], "active_events": j[i+1], "inactive_events": j[i+4]})
            else:
                ts_inf = [t for k in mapping.get("infusion_timestamps", []) for t in arrays.get(k, []) if t>=0]
                ts_act = [t for k in mapping.get("active_timestamps", []) for t in arrays.get(k, []) if t>=0]
                ts_inact = [t for k in mapping.get("inactive_timestamps", []) for t in arrays.get(k, []) if t>=0]
                
                if meta.get("time unit", "").lower() == "minutes":
                    ts_inf, ts_act, ts_inact = [[t*60 for t in l] for l in [ts_inf, ts_act, ts_inact]]

                max_h = int(dur // 3600)
                all_ts = ts_inf + ts_act + ts_inact
                if all_ts: max_h = max(max_h, int(max(all_ts)//3600))
                
                for h in range(max_h + 1):
                    all_hr.append({
                        "canonical_subject": canon, "gender": row["gender"], "program_name": prog, "hour": h,
                        "infusion_events": sum(1 for t in ts_inf if int(t//3600)==h),
                        "active_events": sum(1 for t in ts_act if int(t//3600)==h),
                        "inactive_events": sum(1 for t in ts_inact if int(t//3600)==h)
                    })

    # --- ID SUMMARY LOGGING ---
    missing_ids = allowed_ids_canon - found_ids_canon
    log_msg("\n--- ID MATCH SUMMARY ---")
    log_msg(f"✅ IDs Found: {len(found_ids_canon)}/{len(allowed_ids_canon)}")
    if missing_ids:
        log_msg(f"❌ IDs Missing ({len(missing_ids)}): {', '.join(sorted(missing_ids))}")
    else:
        log_msg("✨ Perfect! All IDs found.")

    if not all_sess:
        log_msg("❌ No data matched your ID list.")
        return

    # Generate Reports
    df_sess = pd.DataFrame(all_sess)
    df_hr = pd.DataFrame(all_hr) if all_hr else pd.DataFrame()

    for prog in df_sess["program_name"].unique():
        log_msg(f"📊 Generating Excel & Plots for: {prog}")
        safe_prog = re.sub(r"[^A-Za-z0-9]", "_", prog)
        sess_sub = df_sess[df_sess["program_name"] == prog].copy()
        hr_sub = df_hr[df_hr["program_name"] == prog].copy() if not df_hr.empty else pd.DataFrame()
        
        # Aggregations
        daily = sess_sub.groupby(["canonical_subject", "gender", "start_date"]).agg({
            "infusions": "sum", "active_presses": "sum", "inactive_presses": "sum", "duration_sec": "sum"
        }).reset_index()
        daily.rename(columns={"infusions": "total_infusions", "active_presses": "total_active_presses", "inactive_presses": "total_inactive_presses"}, inplace=True)
        
        avgs = sess_sub.groupby(["canonical_subject", "gender"]).agg({
            "infusions": ["mean", "std"], "active_presses": ["mean"], "duration_sec": "mean"
        }).reset_index()
        avgs.columns = ['_'.join(col).strip() if col[1] else col[0] for col in avgs.columns.values]
        
        # Excel
        output_file = os.path.join(folder_path, f"{safe_prog}_Analysis.xlsx")
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                sess_sub.to_excel(writer, "01_Session_Data", index=False)
                if not hr_sub.empty: hr_sub.to_excel(writer, "02_Hourly_Data", index=False)
                daily.to_excel(writer, "03_Daily_Summaries", index=False)
                avgs.to_excel(writer, "04_Subject_Averages", index=False)
            
            generate_visualizations(sess_sub, hr_sub, daily, avgs, folder_path, safe_prog)
        except Exception as e:
            log_msg(f"⚠️ Error saving {prog}: {e}")

# ============================================================================
# 7. MAIN
# ============================================================================
if __name__ == "__main__":
    app = MedPCApp()
    app.mainloop()