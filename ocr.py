"""Number-plate OCR.

Reads plate text from a cropped plate image. Engines are auto-detected in this
order: PaddleOCR -> EasyOCR -> pytesseract. If none is available the pipeline
still runs, it just leaves plate_text empty.
"""
import cv2
import numpy as np


class PlateOCR:
    def __init__(self, lang="en", engine=None):
        self.engine_name = None
        self._paddle = None
        self._easy = None
        self._tess = None

        for name in ([engine] if engine else ["paddleocr", "easyocr", "pytesseract"]):
            if name == "paddleocr" and self._init_paddle(lang):
                break
            if name == "easyocr" and self._init_easy(lang):
                break
            if name == "pytesseract" and self._init_tesseract():
                break

        if self.engine_name is None:
            print("[ocr] No OCR engine available -> plate_text will be empty.")

    # ------------------------------------------------------------- init
    def _init_paddle(self, lang):
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:
            print(f"[ocr] paddleocr not installed: {exc}")
            return False
        # PaddleOCR 3.x crashes on CPU with the oneDNN (MKLDNN) backend for the
        # text-detection op ("ConvertPirAttribute2RuntimeAttribute not support"),
        # so we disable MKLDNN. Doc-level pre-processing is also turned off
        # (it is meant for scanned documents, not small plate crops). The 2.x
        # signature (use_angle_cls) is kept as a fallback.
        candidates = (
            {"lang": lang, "enable_mkldnn": False,
             "use_doc_orientation_classify": False,
             "use_doc_unwarping": False,
             "use_textline_orientation": False},
            {"lang": lang, "enable_mkldnn": False},
            {"use_angle_cls": True, "lang": lang},
        )
        for kwargs in candidates:
            try:
                self._paddle = PaddleOCR(**kwargs)
                self.engine_name = "paddleocr"
                return True
            except Exception:
                self._paddle = None
        print("[ocr] paddleocr failed to initialise")
        return False

    def _init_easy(self, lang):
        try:
            import easyocr

            self._easy = easyocr.Reader([lang], gpu=False, verbose=False)
            self.engine_name = "easyocr"
            return True
        except Exception as exc:
            print(f"[ocr] easyocr unavailable: {exc}")
            return False

    def _init_tesseract(self):
        try:
            import pytesseract

            self._tess = pytesseract
            self.engine_name = "pytesseract"
            return True
        except Exception as exc:
            print(f"[ocr] pytesseract unavailable: {exc}")
            return False

    @property
    def available(self):
        return self.engine_name is not None

    # ---------------------------------------------------------- helpers
    @staticmethod
    def _clean(text):
        keep = "".join(ch for ch in text.upper() if ch.isalnum() or ch in " -")
        return " ".join(keep.split())

    @staticmethod
    def _upscale(image, min_width=160):
        h, w = image.shape[:2]
        if w == 0 or h == 0:
            return image
        scale = max(1.0, min_width / w)
        if scale > 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_CUBIC)
        return image

    def _prepare_gray(self, image):
        img = image
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = self._upscale(img)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img

    # -------------------------------------------------------------- read
    def read(self, image):
        """Return (text, confidence). Empty results when OCR is unavailable."""
        if not self.available or image is None or image.size == 0:
            return "", 0.0
        if self.engine_name == "paddleocr":
            return self._read_paddle(self._upscale(image))
        prepared = self._prepare_gray(image)
        if prepared.size == 0:
            return "", 0.0
        if self.engine_name == "easyocr":
            return self._read_easy(prepared)
        return self._read_tesseract(prepared)

    def _read_paddle(self, image):
        texts, scores = [], []
        # PaddleOCR 3.x -> predict()
        try:
            results = self._paddle.predict(image)
        except Exception:
            results = None
        if results:
            for res in results:
                try:
                    rec_texts = res["rec_texts"]
                    rec_scores = res["rec_scores"]
                except Exception:
                    continue
                for text, score in zip(rec_texts, rec_scores):
                    if text:
                        texts.append(text)
                        scores.append(float(score))
            if texts:
                return self._clean(" ".join(texts)), float(np.mean(scores))
        # PaddleOCR 2.x -> ocr()
        try:
            results = self._paddle.ocr(image, cls=True)
        except Exception:
            return "", 0.0
        lines = results[0] if results and isinstance(results[0], list) else []
        for item in lines or []:
            try:
                text, score = item[1]
            except Exception:
                continue
            if text:
                texts.append(text)
                scores.append(float(score))
        if not texts:
            return "", 0.0
        return self._clean(" ".join(texts)), float(np.mean(scores))

    def _read_easy(self, image):
        results = self._easy.readtext(image, detail=1, paragraph=False)
        if not results:
            return "", 0.0
        text = "".join(r[1] for r in results)
        return self._clean(text), float(np.mean([r[2] for r in results]))

    def _read_tesseract(self, image):
        data = self._tess.image_to_data(
            image, output_type=self._tess.Output.DICT, config="--psm 7"
        )
        words, confs = [], []
        for token, conf in zip(data["text"], data["conf"]):
            try:
                c = float(conf)
            except (TypeError, ValueError):
                c = -1.0
            if token.strip() and c >= 0:
                words.append(token)
                confs.append(c)
        if not words:
            return "", 0.0
        return self._clean(" ".join(words)), float(np.mean(confs)) / 100.0
