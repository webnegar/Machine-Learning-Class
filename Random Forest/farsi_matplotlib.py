# -------------------------------
# 📘 farsi_matplotlib.py
# تنظیمات کامل برای نمایش صحیح فارسی در matplotlib
# نویسنده: ChatGPT (Ehsan’s version)
# -------------------------------

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import arabic_reshaper
from bidi.algorithm import get_display
import os


def fa(text: str) -> str:
    """
    تبدیل متن فارسی برای نمایش صحیح در matplotlib.
    شامل چسباندن حروف و تنظیم جهت راست‌به‌چپ.
    """
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)


def set_farsi_font(font_path: str = None):
    """
    تنظیم فونت پیش‌فرض فارسی برای matplotlib.
    اگر مسیر فونت داده نشود، سعی می‌کند یکی از فونت‌های متداول سیستم را پیدا کند.
    """
    common_fonts = [
        "VAZIRMATN-REGULAR.TTF",  # ویندوز
        "/usr/share/fonts/truetype/vazirmatn/Vazirmatn-Regular.ttf",  # لینوکس
        "/Library/Fonts/Vazirmatn-Regular.ttf",  # مک
        "C:/Windows/Fonts/B Nazanin.ttf",
        "C:/Windows/Fonts/B Yekan.ttf",
        "C:/Windows/Fonts/Tahoma.ttf",
    ]

    # اگر مسیر داده نشده، به دنبال اولین فونت موجود می‌گردد
    if font_path is None:
        for path in common_fonts:
            if os.path.exists(path):
                font_path = path
                break

    if font_path is None or not os.path.exists(font_path):
        print("⚠️ فونت فارسی پیدا نشد! لطفاً مسیر فونت را به صورت دستی تعیین کنید.")
        print("مثال:")
        print("  set_farsi_font('C:/Windows/Fonts/Vazirmatn-Regular.ttf')")
        return

    font_prop = font_manager.FontProperties(fname=font_path)
    rcParams['font.family'] = font_prop.get_name()
    rcParams['axes.unicode_minus'] = False  # برای درست نمایش دادن علامت منفی

    print(f"✅ فونت فارسی فعال شد: {font_prop.get_name()}")

# اگر بخواهی مستقیماً با import اجرا شود، می‌توانی این خط را فعال کنی:
# set_farsi_font()
