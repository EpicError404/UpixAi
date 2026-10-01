import argparse
import os
import cv2
from upscaler_engine import AIImageUpscaler


def main():
  parser = argparse.ArgumentParser(
      description="UpixAi - Windows AI Destekli Fotoğraf Büyütme ve İyileştirme"
  )
  parser.add_argument(
      "-i",
      "--input",
      required=True,
      help="Büyütülecek fotoğrafın dosya yolu (Örn: resim.jpg)",
  )
  parser.add_argument(
      "-o",
      "--output",
      default="output_buyutulmus.png",
      help="Çıktı fotoğrafının kaydedileceği yol",
  )
  parser.add_argument(
      "-s",
      "--scale",
      type=int,
      default=4,
      help="Büyütme katsayısı (Örn: 2, 4)",
  )
  parser.add_argument(
      "-m",
      "--model",
      default="LapSRN_x4.pb",
      help="Yapay zeka model dosyasının yolu (.pb)",
  )

  args = parser.parse_args()

  if not os.path.exists(args.input):
    print(f"[UpixAi - HATA] Giriş dosyası bulunamadı: {args.input}")
    return

  print("--- UpixAi Başlatılıyor ---")

  upscaler = AIImageUpscaler(
      model_path=args.model,
      scale=args.scale,
      auto_denoise=True,
      denoise_pct=15,
  )

  try:
    output_img = upscaler.upscale(args.input)
    cv2.imwrite(args.output, output_img)
    print(f"[UpixAi - BAŞARILI] İşlem tamamlandı! Kaydedilen: {args.output}")
  except Exception as e:
    print(f"[UpixAi - HATA] İşlem başarısız oldu: {e}")


if __name__ == "__main__":
  main()