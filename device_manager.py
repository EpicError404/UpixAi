import cv2


def check_hardware_acceleration():
  """Sistemde uyumlu bir GPU (CUDA) olup olmadığını kontrol eder.

  Yoksa CPU'ya yönlendirir.
  """
  try:
    cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
    if cuda_count > 0:
      print(
          "[BİLGİ] Uyumlu GPU (CUDA) tespit edildi. Yapay zeka"
          " hızlandırması GPU üzerinden yapılacak."
      )
      return True, cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA
    else:
      print(
          "[BİLGİ] Uyumlu GPU bulunamadı veya CUDA aktif değil. İşlem"
          " CPU üzerinden gerçekleştirilecek."
      )
      return False, cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU
  except Exception as e:
    print(
        f"[UYARI] Donanım kontrolü sırasında hata oluştu: {e}. CPU'ya geçiliyor."
    )
    return False, cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU