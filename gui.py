import os
import threading
from tkinter import filedialog, messagebox
import customtkinter as ctk
from device_manager import check_hardware_acceleration
from upscaler_engine import AIImageUpscaler, ProcessingCancelledException

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class ModernUpscalerApp:

  def __init__(self, root):
    self.root = root
    self.language = "tr"
    self.root.title(self.t("UpixAi - Yapay Zeka Fotoğraf Büyütücü Pro", "UpixAi - AI Image Upscaler Pro"))
    # Yükseklik 740'tan 570'e düşürülerek alttaki boşluk kaldırıldı
    self.root.geometry("640x570")
    self.root.resizable(False, False)

    # Değişkenler
    self.selected_files = []
    self.output_dir = ctk.StringVar()
    self.scale_var = ctk.StringVar(value="4")
    self.model_var = ctk.StringVar(value="Real-ESRGAN x4plus")
    self.sharpen_var = ctk.StringVar(value="25")
    self.format_var = ctk.StringVar(value="JPEG")
    self.quality_var = ctk.IntVar(value=95)
    self.auto_brightness_var = ctk.BooleanVar(value=False)

    # Otomatik Gren Giderme varsayılanı açık ve %20
    self.auto_denoise_var = ctk.BooleanVar(value=False)
    self.denoise_var = ctk.StringVar(value="20")

    # Thread ve Durdurma kontrol bayrakları
    self.stop_event = threading.Event()
    self.is_processing = False

    self.setup_ui()
    self.check_initial_hardware()

  def setup_ui(self):
    # Başlık
    header = ctk.CTkFrame(self.root, fg_color="transparent")
    header.pack(fill="x", padx=20, pady=(15, 10))
    ctk.CTkLabel(
      header,
      text=self.t("UpixAi - AI Görüntü Büyütme ve İyileştirme Stüdyosu", "UpixAi - AI Image Upscaling & Enhancement Studio"),
        font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
    ).pack(side="left")
    self.btn_language = ctk.CTkButton(
      header,
      text="English" if self.language == "tr" else "Türkçe",
      width=90,
      height=30,
      command=self.toggle_language,
    )
    self.btn_language.pack(side="right")

    # KART 1: Dosya / Klasör Seçim Alanı
    file_card = ctk.CTkFrame(
        self.root, corner_radius=12, fg_color=("#2b2b2b", "#1e1e1e")
    )
    file_card.pack(fill="x", padx=20, pady=6, ipady=5)

    ctk.CTkLabel(
        file_card, text=self.t("Girdi Fotoğrafları:", "Input Photos:"), font=("Segoe UI", 12, "bold")
    ).pack(anchor="w", padx=15, pady=(5, 0))

    in_btn_frame = ctk.CTkFrame(file_card, fg_color="transparent")
    in_btn_frame.pack(fill="x", padx=15, pady=4)

    self.lbl_selected_count = ctk.CTkLabel(
        in_btn_frame,
        text=self.t(
          f"{len(self.selected_files)} dosya seçildi." if self.selected_files else "Seçilen dosya yok.",
          f"{len(self.selected_files)} file(s) selected." if self.selected_files else "No files selected.",
        ),
        font=("Segoe UI", 11),
        text_color="gray",
    )
    self.lbl_selected_count.pack(side="left", padx=5)

    ctk.CTkButton(
        in_btn_frame,
        text=self.t("Klasör Seç", "Select Folder"),
        width=100,
        height=32,
        command=self.select_folder,
    ).pack(side="right", padx=(5, 0))
    ctk.CTkButton(
        in_btn_frame,
        text=self.t("Dosya(lar) Seç", "Select File(s)"),
        width=110,
        height=32,
        command=self.select_files,
    ).pack(side="right")

    # Çıkış Klasörü
    ctk.CTkLabel(
        file_card, text=self.t("Kayıt Klasörü:", "Output Folder:"), font=("Segoe UI", 12, "bold")
    ).pack(anchor="w", padx=15, pady=(6, 0))

    out_btn_frame = ctk.CTkFrame(file_card, fg_color="transparent")
    out_btn_frame.pack(fill="x", padx=15, pady=(4, 8))

    self.ent_output = ctk.CTkEntry(
        out_btn_frame, textvariable=self.output_dir, height=32
    )
    self.ent_output.pack(side="left", fill="x", expand=True, padx=(0, 10))
    ctk.CTkButton(
        out_btn_frame,
        text=self.t("Hedef Klasör", "Browse"),
        width=100,
        height=32,
        command=self.select_output_dir,
    ).pack(side="right")

    # KART 2: AI ve Filtre Ayarları
    settings_card = ctk.CTkFrame(
        self.root, corner_radius=12, fg_color=("#2b2b2b", "#1e1e1e")
    )
    settings_card.pack(fill="x", padx=20, pady=6, ipady=5)

    grid_frame = ctk.CTkFrame(settings_card, fg_color="transparent")
    grid_frame.pack(fill="x", padx=15, pady=6)

    ctk.CTkLabel(grid_frame, text=self.t("AI Model:", "AI Model:")).grid(
        row=0, column=0, sticky="w", pady=4
    )
    self.combo_model = ctk.CTkComboBox(
        grid_frame,
        values=["Real-ESRGAN x4plus", "LapSRN_x4.pb", "FSRCNN_x2.pb"],
        variable=self.model_var,
        width=140,
    )
    self.combo_model.grid(row=0, column=1, padx=(10, 20), sticky="w")

    ctk.CTkLabel(grid_frame, text=self.t("Keskinlik (%):", "Sharpness (%):")).grid(
        row=0, column=2, sticky="w", pady=4
    )
    self.combo_sharpen = ctk.CTkComboBox(
        grid_frame,
        values=["0", "25", "50", "100"],
        variable=self.sharpen_var,
        width=85,
    )
    self.combo_sharpen.grid(row=0, column=3, padx=10, sticky="w")

    # Filtre Checkbox'ları
    self.chk_brightness = ctk.CTkCheckBox(
        settings_card,
        text=self.t("Otomatik Parlaklık Dengeleme", "Auto Brightness Balance"),
        variable=self.auto_brightness_var,
    )
    self.chk_brightness.pack(anchor="w", padx=15, pady=(4, 4))

    # Otomatik Gren Giderme Satırı (%10, %20, %30)
    denoise_row = ctk.CTkFrame(settings_card, fg_color="transparent")
    denoise_row.pack(fill="x", padx=15, pady=(2, 6))

    self.chk_denoise = ctk.CTkCheckBox(
        denoise_row,
        text=self.t("Otomatik Gren Giderme", "Auto Noise Reduction"),
        variable=self.auto_denoise_var,
        command=self.toggle_denoise,
    )
    self.chk_denoise.pack(side="left")

    ctk.CTkLabel(denoise_row, text=self.t("Oran (%):", "Amount (%):"), font=("Segoe UI", 11)).pack(
        side="left", padx=(15, 5)
    )

    self.combo_denoise = ctk.CTkComboBox(
        denoise_row,
        values=["10", "20", "30"],
        variable=self.denoise_var,
        width=80,
    )
    self.combo_denoise.pack(side="left")

    # KART 3: Çıktı Formatı ve Kalite
    export_card = ctk.CTkFrame(
        self.root, corner_radius=12, fg_color=("#2b2b2b", "#1e1e1e")
    )
    export_card.pack(fill="x", padx=20, pady=6, ipady=5)

    exp_frame = ctk.CTkFrame(export_card, fg_color="transparent")
    exp_frame.pack(fill="x", padx=15, pady=6)

    ctk.CTkLabel(
        exp_frame, text="Format:", font=("Segoe UI", 12, "bold")
    ).pack(side="left")
    self.combo_format = ctk.CTkComboBox(
        exp_frame,
        values=["PNG", "JPEG", "WEBP"],
        variable=self.format_var,
        width=95,
        command=self.on_format_changed,
    )
    self.combo_format.pack(side="left", padx=(10, 25))

    self.lbl_quality_title = ctk.CTkLabel(
        exp_frame, text=self.t("Kalite (%):", "Quality (%):"), font=("Segoe UI", 12, "bold")
    )
    self.lbl_quality_title.pack(side="left")

    self.slider_quality = ctk.CTkSlider(
        exp_frame,
        from_=10,
        to=100,
        variable=self.quality_var,
        width=150,
        command=self.update_quality_label,
    )
    self.slider_quality.pack(side="left", padx=10)

    self.lbl_quality_val = ctk.CTkLabel(
        exp_frame, text=f"%{self.quality_var.get()}", width=40
    )
    self.lbl_quality_val.pack(side="left")

    # İlerleme Çubuğu
    self.progress_bar = ctk.CTkProgressBar(self.root, height=12)
    self.progress_bar.set(0)
    self.progress_bar.pack(fill="x", padx=20, pady=(15, 8))

    # Butonlar: Başlat ve Durdur
    btn_box = ctk.CTkFrame(self.root, fg_color="transparent")
    btn_box.pack(fill="x", padx=20, pady=(4, 10))

    self.btn_run = ctk.CTkButton(
        btn_box,
        text=self.t("İşlemi Başlat", "Start Processing"),
        height=45,
        fg_color="#1f6feb",
        hover_color="#388bfd",
        font=ctk.CTkFont(size=14, weight="bold"),
        command=self.start_processing_thread,
    )
    self.btn_run.pack(side="left", fill="x", expand=True, padx=(0, 10))

    self.btn_stop = ctk.CTkButton(
        btn_box,
        text=self.t("Durdur", "Stop"),
        width=120,
        height=45,
        fg_color="#da3633",
        hover_color="#f85149",
        state="disabled",
        font=ctk.CTkFont(size=14, weight="bold"),
        command=self.stop_processing,
    )
    self.btn_stop.pack(side="right")

    # Butonların Altındaki Status Bar
    status_card = ctk.CTkFrame(
        self.root,
        corner_radius=10,
        fg_color=("#e5e5e5", "#161616"),
        border_width=1,
        border_color=("#cccccc", "#2b2b2b"),
    )
    status_card.pack(fill="x", padx=20, pady=(4, 15), ipady=3)

    self.status_bar = ctk.CTkLabel(
        status_card,
        text=self.t("Durum: Hazır", "Status: Ready"),
        font=("Segoe UI", 11, "bold"),
        text_color="gray",
    )
    self.status_bar.pack(expand=True, fill="both", padx=10, pady=5)

    self.on_format_changed(self.format_var.get())

  def t(self, turkish, english):
    return turkish if self.language == "tr" else english

  def toggle_language(self):
    self.language = "en" if self.language == "tr" else "tr"
    for widget in self.root.winfo_children():
      widget.destroy()
    self.setup_ui()
    self.check_initial_hardware()

  def toggle_denoise(self):
    if self.auto_denoise_var.get():
      self.combo_denoise.configure(state="normal")
    else:
      self.combo_denoise.configure(state="disabled")

  def update_quality_label(self, val):
    self.lbl_quality_val.configure(text=f"%{int(val)}")

  def on_format_changed(self, choice):
    if choice == "PNG":
      self.slider_quality.configure(state="disabled")
      self.lbl_quality_val.configure(text_color="gray")
    else:
      self.slider_quality.configure(state="normal")
      self.lbl_quality_val.configure(text_color="white")

  def select_files(self):
    files = filedialog.askopenfilenames(
        title=self.t("Büyütülecek Fotoğrafları Seçin", "Select Photos to Upscale"),
        filetypes=[
            (self.t("Fotoğraf Dosyaları", "Image Files"), "*.jpg *.jpeg *.png *.bmp *.webp"),
            (self.t("Tüm Dosyalar", "All Files"), "*.*"),
        ],
    )
    if files:
      self.selected_files = list(files)
      self.lbl_selected_count.configure(
          text=self.t(f"{len(self.selected_files)} dosya seçildi.", f"{len(self.selected_files)} file(s) selected."), text_color="#58a6ff"
      )
      if not self.output_dir.get():
        self.output_dir.set(os.path.dirname(self.selected_files[0]))

  def select_folder(self):
    folder = filedialog.askdirectory(title=self.t("Fotoğrafların Olduğu Klasörü Seçin", "Select Folder Containing Photos"))
    if folder:
      valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
      found = [
          os.path.join(folder, f)
          for f in os.listdir(folder)
          if f.lower().endswith(valid_exts)
      ]
      if found:
        self.selected_files = found
        self.lbl_selected_count.configure(
            text=self.t(f"Klasörden {len(found)} dosya seçildi.", f"{len(found)} file(s) selected from folder."), text_color="#58a6ff"
        )
        if not self.output_dir.get():
          self.output_dir.set(os.path.join(folder, "buyutulmus_ciktilar"))
      else:
        messagebox.showwarning(
            "UpixAi", self.t("Seçilen klasörde uygun formatta fotoğraf bulunamadı!", "No supported photos found in the selected folder!")
        )

  def select_output_dir(self):
    folder = filedialog.askdirectory(title=self.t("Kayıt Edilecek Klasörü Seçin", "Select Output Folder"))
    if folder:
      self.output_dir.set(folder)

  def check_initial_hardware(self):
    is_gpu, _, _ = check_hardware_acceleration()
    if is_gpu:
      self.status_bar.configure(
          text=self.t("Donanım: GPU (CUDA) Hızlandırması Aktif", "Hardware: GPU (CUDA) Acceleration Active"), text_color="#2ea043"
      )
    else:
      self.status_bar.configure(
          text=self.t("Donanım: CPU Modu Aktif", "Hardware: CPU Mode Active"), text_color="#d29922"
      )

  def stop_processing(self):
    if self.is_processing:
      self.stop_event.set()
      self.status_bar.configure(
          text=self.t("Durduruluyor, mevcut fotoğraf tamamlanıyor...", "Stopping; finishing current photo..."),
          text_color="#f85149",
      )
      self.btn_stop.configure(state="disabled")

  def start_processing_thread(self):
    if not self.selected_files:
      messagebox.showerror(self.t("Hata", "Error"), self.t("Lütfen en az bir fotoğraf seçin!", "Please select at least one photo!"))
      return

    out_dir = self.output_dir.get().strip()
    if not out_dir:
      messagebox.showerror(self.t("Hata", "Error"), self.t("Lütfen bir çıktı klasörü belirleyin!", "Please choose an output folder!"))
      return

    os.makedirs(out_dir, exist_ok=True)

    self.stop_event.clear()
    self.is_processing = True

    self.btn_run.configure(state="disabled", text=self.t("İşleniyor...", "Processing..."))
    self.btn_language.configure(state="disabled")
    self.btn_stop.configure(state="normal")
    self.progress_bar.set(0)

    threading.Thread(target=self._worker, daemon=True).start()

  def _worker(self):
    total = len(self.selected_files)
    completed = 0
    failures = []
    scale = int(self.scale_var.get())
    sharpen_pct = int(self.sharpen_var.get())
    chosen_format = self.format_var.get()
    quality = self.quality_var.get()
    denoise_pct = (
        int(self.denoise_var.get()) if self.auto_denoise_var.get() else 0
    )

    try:
      upscaler = AIImageUpscaler(
          model_path=self.model_var.get(),
          scale=scale,
          sharpen_pct=sharpen_pct,
          auto_brightness=self.auto_brightness_var.get(),
          auto_denoise=self.auto_denoise_var.get(),
          denoise_pct=denoise_pct,
      )

      for index, file_path in enumerate(self.selected_files):
        if self.stop_event.is_set():
          break

        filename = os.path.basename(file_path)
        self.status_bar.configure(
            text=self.t(f"İşleniyor ({index + 1}/{total}): {filename}", f"Processing ({index + 1}/{total}): {filename}"),
            text_color="#58a6ff",
        )

        try:
          output_img = upscaler.upscale(
              file_path, cancel_event=self.stop_event
          )
          base_name, _ = os.path.splitext(filename)
          target_path = os.path.join(
              self.output_dir.get(), f"{base_name}_buyutulmus"
          )
          upscaler.save_image(
              output_img,
              target_path,
              img_format=chosen_format,
              quality=quality,
          )
          completed += 1
        except ProcessingCancelledException:
          break
        except Exception as e:
          failures.append((filename, str(e)))
          print(f"[UpixAi - HATA] {filename} işlenemedi: {e}")

        progress_val = (index + 1) / total
        self.progress_bar.set(progress_val)

      if self.stop_event.is_set():
        self.status_bar.configure(
            text=self.t(f"İşlem durduruldu! ({completed}/{total} tamamlandı)", f"Processing stopped! ({completed}/{total} completed)"),
            text_color="#f85149",
        )
        messagebox.showwarning(
            "UpixAi", self.t(f"İşlem durduruldu!\nTamamlanan: {completed}/{total}", f"Processing stopped!\nCompleted: {completed}/{total}")
        )
      else:
        if failures:
          failure_details = "\n".join(
            f"{filename}: {error}" for filename, error in failures[:10]
          )
          if len(failures) > 10:
            failure_details += self.t(
                f"\n... ve {len(failures) - 10} hata daha.",
                f"\n... and {len(failures) - 10} more error(s).",
            )
          self.status_bar.configure(
            text=self.t(
              f"İşlem tamamlandı: {completed}/{total} başarılı, {len(failures)} hata.",
              f"Processing finished: {completed}/{total} succeeded, {len(failures)} error(s).",
            ),
            text_color="#d29922",
          )
          messagebox.showwarning(
            "UpixAi",
            self.t(
              f"Başarılı: {completed}/{total}\nKayıt Yeri: {self.output_dir.get()}\n\nHatalar:\n{failure_details}",
              f"Succeeded: {completed}/{total}\nOutput folder: {self.output_dir.get()}\n\nErrors:\n{failure_details}",
            ),
          )
        else:
          self.status_bar.configure(
            text=self.t(f"Tüm işlemler başarıyla tamamlandı! ({completed}/{total})", f"All tasks completed successfully! ({completed}/{total})"),
            text_color="#2ea043",
          )
          messagebox.showinfo(
            "UpixAi",
            self.t(f"Tüm fotoğraflar başarıyla büyütüldü!\nKayıt Yeri: {self.output_dir.get()}", f"All photos were successfully upscaled!\nOutput folder: {self.output_dir.get()}"),
          )

    except Exception as e:
      self.status_bar.configure(
          text=self.t("İşlem sırasında hata oluştu!", "An error occurred during processing!"), text_color="#f85149"
      )
      messagebox.showerror(self.t("UpixAi Hatası", "UpixAi Error"), self.t(f"Beklenmeyen bir hata oluştu:\n{e}", f"An unexpected error occurred:\n{e}"))
    finally:
      self.is_processing = False
      self.btn_run.configure(state="normal", text=self.t("İşlemi Başlat", "Start Processing"))
      self.btn_stop.configure(state="disabled")
      self.btn_language.configure(state="normal")


if __name__ == "__main__":
  root = ctk.CTk()
  app = ModernUpscalerApp(root)
  root.mainloop()