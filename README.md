# UpixAi

CustomTkinter arayüzüne sahip Windows fotoğraf büyütme uygulaması. Varsayılan
model, birlikte gelen NCNN/Vulkan altyapısını kullanan Real-ESRGAN x4plus'tır.
FSRCNN x2 ve LapSRN x4 modelleri de kullanılabilir.

## Kaynaktan çalıştırma

```powershell
py -m pip install -r requirements.txt
py gui.py
```

Real-ESRGAN çalışma zamanı, model dosyaları ve lisansı `third_party/realesrgan/`
klasöründedir. Vulkan destekli bir ekran kartı sürücüsü önerilir. OpenCV
modelleri `FSRCNN_x2.pb` ve `LapSRN_x4.pb` proje kök dizininde bulunur.

## Tek dosyalık exe oluşturma

```powershell
py -m pip install pyinstaller
py -m PyInstaller --noconfirm --onefile --windowed --name UpixAi `
  --add-data "LapSRN_x4.pb;." `
  --add-data "FSRCNN_x2.pb;." `
  --add-data "third_party/realesrgan;third_party/realesrgan" gui.py
```

Güncel tek dosyalık uygulamayı [GitHub Releases](https://github.com/EpicError404/UpixAi/releases/latest)
sayfasından indirebilirsiniz.