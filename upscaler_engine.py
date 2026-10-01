import os
import sys
import subprocess
import tempfile
import time
import cv2
import numpy as np
from device_manager import check_hardware_acceleration


class ProcessingCancelledException(Exception):
  """İşlemin kullanıcı tarafından durdurulduğunu belirten özel istisna."""

  pass


class AIImageUpscaler:

  def __init__(
      self,
      model_path="LapSRN_x4.pb",  # 1. Varsayılan model LapSRN_x4 yapıldı
      scale=4,
      sharpen_pct=25,
      auto_brightness=False,
      auto_denoise=True,  # 3. Varsayılan olarak açık
      denoise_pct=15,  # 3. Varsayılan seviye %25
  ):
    self.model_path = model_path
    self.scale = scale
    self.sharpen_pct = sharpen_pct
    self.auto_brightness = auto_brightness
    self.auto_denoise = auto_denoise
    self.denoise_pct = denoise_pct
    self.use_realesrgan = self.model_path == "Real-ESRGAN x4plus"
    self.is_gpu, self.backend, self.target = check_hardware_acceleration()
    self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
    if self.use_realesrgan:
      self._load_realesrgan_runtime()
    else:
      self._load_model()

  def _load_realesrgan_runtime(self):
    resource_root = getattr(
        sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))
    )
    self.realesrgan_dir = os.path.join(
        resource_root, "third_party", "realesrgan"
    )
    self.realesrgan_exe = os.path.join(
        self.realesrgan_dir, "realesrgan-ncnn-vulkan.exe"
    )
    self.realesrgan_model_dir = os.path.join(self.realesrgan_dir, "models")
    required_files = [
        self.realesrgan_exe,
      os.path.join(self.realesrgan_model_dir, "realesrgan-x4plus.param"),
      os.path.join(self.realesrgan_model_dir, "realesrgan-x4plus.bin"),
        os.path.join(self.realesrgan_dir, "vcomp140.dll"),
    ]
    missing_files = [path for path in required_files if not os.path.isfile(path)]
    if missing_files:
      raise FileNotFoundError(
          "Real-ESRGAN runtime files are missing: " + ", ".join(missing_files)
      )
    self.model_loaded = True
    print("[UpixAi] Real-ESRGAN x4plus runtime loaded")

  def _upscale_with_realesrgan(self, img, cancel_event=None):
    encoded, image_data = cv2.imencode(".png", img)
    if not encoded:
      raise ValueError("Input image could not be prepared for Real-ESRGAN")

    with tempfile.TemporaryDirectory(prefix="upixai_") as temp_dir:
      input_path = os.path.join(temp_dir, "input.png")
      output_path = os.path.join(temp_dir, "output.png")
      image_data.tofile(input_path)
      command = [
          self.realesrgan_exe,
          "-i", input_path,
          "-o", output_path,
          "-m", self.realesrgan_model_dir,
          "-n", "realesrgan-x4plus",
          "-s", "4",
          "-t", "128",
          "-f", "png",
      ]
      process = subprocess.Popen(
          command,
          cwd=self.realesrgan_dir,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          text=True,
          encoding="utf-8",
          errors="replace",
          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
      )
      while process.poll() is None:
        if cancel_event and cancel_event.is_set():
          process.terminate()
          try:
            process.wait(timeout=5)
          except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
          raise ProcessingCancelledException()
        time.sleep(0.1)

      output, _ = process.communicate()
      if process.returncode != 0:
        raise RuntimeError(
            f"Real-ESRGAN failed (exit code {process.returncode}): {output.strip()}"
        )
      if not os.path.isfile(output_path):
        raise RuntimeError(
            f"Real-ESRGAN did not create an output image: {output.strip()}"
        )
      result_data = np.fromfile(output_path, dtype=np.uint8)
      result = cv2.imdecode(result_data, cv2.IMREAD_COLOR)
      if result is None:
        raise ValueError("Real-ESRGAN output image could not be read")
      return result

  def _load_model(self):
    if getattr(sys, "frozen", False) and not os.path.isabs(self.model_path):
      bundled_model = os.path.join(sys._MEIPASS, self.model_path)
      if os.path.exists(bundled_model):
        self.model_path = bundled_model

    if os.path.exists(self.model_path):
      try:
        self.sr.readModel(self.model_path)
        model_name = "fsrcnn"
        lower_path = self.model_path.lower()
        if "espcn" in lower_path:
          model_name = "espcn"
        elif "edsr" in lower_path:
          model_name = "edsr"
        elif "lapsrn" in lower_path:
          model_name = "lapsrn"

        self.sr.setModel(model_name, self.scale)
        self.sr.setPreferableBackend(self.backend)
        self.sr.setPreferableTarget(self.target)
        self.model_loaded = True
        print(f"[UpixAi] Model yüklendi: {self.model_path}")
      except Exception as e:
        print(f"[UpixAi - HATA] Model yüklenirken hata: {e}")
        self.model_loaded = False
    else:
      print(
          f"[UpixAi - UYARI] '{self.model_path}' bulunamadı. Klasik yöntem"
          " kullanılacak."
      )
      self.model_loaded = False

  def upscale(self, image_path, cancel_event=None):
    if cancel_event and cancel_event.is_set():
      raise ProcessingCancelledException()

    try:
      image_data = np.fromfile(image_path, dtype=np.uint8)
      img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    except OSError as e:
      raise ValueError(f"Fotoğraf okunamadı veya bozuk: {image_path}") from e
    if img is None:
      raise ValueError(f"Fotoğraf okunamadı veya bozuk: {image_path}")

    # 1. Büyütme Adımı
    if self.use_realesrgan:
      result = self._upscale_with_realesrgan(img, cancel_event)
    elif self.model_loaded:
      try:
        result = self.sr.upsample(img)
      except Exception:
        result = self._fallback_resize(img)
    else:
      result = self._fallback_resize(img)

    if cancel_event and cancel_event.is_set():
      raise ProcessingCancelledException()

    # 2. Otomatik Parlaklık Dengeleme
    if self.auto_brightness:
      result = self._balance_brightness(result)

    if cancel_event and cancel_event.is_set():
      raise ProcessingCancelledException()

    # 3. Otomatik Gren Giderme (Denoise)
    if self.auto_denoise and self.denoise_pct > 0:
      result = self._remove_noise(result, self.denoise_pct)

    if cancel_event and cancel_event.is_set():
      raise ProcessingCancelledException()

    # 4. Keskinleştirme
    if self.sharpen_pct > 0:
      result = self._apply_sharpening(result, self.sharpen_pct)

    return result

  def _fallback_resize(self, img):
    h, w = img.shape[:2]
    return cv2.resize(
        img,
        (w * self.scale, h * self.scale),
        interpolation=cv2.INTER_LANCZOS4,
    )

  def _balance_brightness(self, img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_balanced = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_balanced, a, b)), cv2.COLOR_LAB2BGR)

  def _remove_noise(self, img, strength=25):
    # Seçilen % oranına göre filtre gücünü (h) dengeler
    # 25 -> 4, 50 -> 7, 75 -> 11, 100 -> 15
    h_val = max(1, int(strength * 0.15))
    return cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=h_val,
        hColor=h_val,
        templateWindowSize=7,
        searchWindowSize=21,
    )

  def _apply_sharpening(self, img, strength):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    alpha = min(max(strength / 100.0, 0.0), 1.0)
    return cv2.addWeighted(sharpened, alpha, img, 1 - alpha, 0)

  @staticmethod
  def save_image(img, target_filepath, img_format="JPEG", quality=95):
    fmt = img_format.lower()
    if fmt == "jpeg":
      fmt = "jpg"

    base_name, _ = os.path.splitext(target_filepath)
    final_output_path = f"{base_name}.{fmt}"

    encode_params = []
    if fmt in ["jpg", "jpeg"]:
      encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    elif fmt == "webp":
      encode_params = [cv2.IMWRITE_WEBP_QUALITY, int(quality)]
    elif fmt == "png":
      encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    encoded, image_data = cv2.imencode(f".{fmt}", img, encode_params)
    if not encoded:
      raise OSError(f"Görüntü dosyası kaydedilemedi: {final_output_path}")
    try:
      image_data.tofile(final_output_path)
    except OSError as e:
      raise OSError(f"Görüntü dosyası kaydedilemedi: {final_output_path}") from e
    return final_output_path