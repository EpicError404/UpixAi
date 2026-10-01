# UpixAi

Windows photo upscaler with a CustomTkinter interface. Real-ESRGAN x4plus is
the default model and uses the bundled NCNN/Vulkan runtime. FSRCNN x2 and
LapSRN x4 are also available.

## Run from source

```powershell
py -m pip install -r requirements.txt
py gui.py
```

The Real-ESRGAN runtime, model files, and its license are in
`third_party/realesrgan/`. A Vulkan-capable graphics driver is recommended.
The OpenCV models are `FSRCNN_x2.pb` and `LapSRN_x4.pb` in the project root.

## Build a single-file executable

```powershell
py -m pip install pyinstaller
py -m PyInstaller --noconfirm --onefile --windowed --name UpixAi `
  --add-data "LapSRN_x4.pb;." `
  --add-data "FSRCNN_x2.pb;." `
  --add-data "third_party/realesrgan;third_party/realesrgan" gui.py
```

The current standalone build is `dist/UpixAi_Standalone.exe`.