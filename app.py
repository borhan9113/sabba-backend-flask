from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS

# تحميل النموذج الأول والـ LabelEncoder
model = pickle.load(open('crop_model_v2.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder_v2.pkl', 'rb'))

# تحميل نموذج التنبؤ المستقبلي
forecast_model = pickle.load(open('forecast_model.pkl', 'rb'))

# تحميل بيانات السلاسل الزمنية
with open('forecast_data_dict.pkl', 'rb') as f:
    forecast_data_dict = pickle.load(f)
# تحميل بيانات المحاصيل من ملف pkl
with open('crops_data.pkl', 'rb') as f:
    df = pickle.load(f)

# قاموس ترجمة المحاصيل (للتوصية)
translations = {
    "rice": "رز مطري، قمح صلب، شعير، بطاطا، خرشف (أرضي شوكي)",
    "maize": "ذرة صفراء، ذرة بيضاء، جزر، بطاطا قرنون",
    "jute": "الحلفاء، كتان، خس، بسباس",
    "cotton": "الكتان، السمسم، بصل، بسباس",
    "coconut": "زيتون، خرشف",
    "papaya": "تين شوكي، تين، طماطم، خرشف",
    "orange": "برتقال، كلمنتين، ليمون، فلفل حلو، خس",
    "apple": "تفاح، إجاص، خيار، بسباس",
    "muskmelon": "شمام، بطيخ أصفر، كوسة، جزر",
    "watermelon": "بطيخ أحمر، شمام، كوسة",
    "grapes": "عنب، توت بري، طماطم كرزية، خرشف",
    "mango": "مشمش، خوخ، جزر، طماطم",
    "banana": "بطاطا حلوة، جزر، كوسة",
    "pomegranate": "رمان، تين، فلفل حار",
    "lentil": "عدس، حمص، بازلاء، فول",
    "blackgram": "فول، بازلاء، فاصوليا خضراء، عدس",
    "mungbean": "بامية، فاصوليا خضراء، لوبيا، فول أخضر",
    "mothbeans": "تمر، لوبيا صغيرة، عدس صغير، بقوليات متنوعة",
    "pigeonpeas": "حمص، عدس أحمر، فول",
    "kidneybeans": "فاصوليا حمراء، لوبيا، فول، بازلاء",
    "chickpea": "بامية، حمص، عدس، فول",
    "coffee": "زعتر، نعناع، خروب، شيح"
}

# قاموس ترجمة لمحاصيل التوقع (forecast)
forecast_translations = {
    "Apples": "تفاح",
    "Apricots": "مشمش",
    "Barley": "شعير",
    "Beans, dry": "فول جاف",
    "Carrots and turnips": "جزر ولفت",
    "Cauliflowers and broccoli": "قرنبيط وبروكلي",
    "Chillies and peppers, dry": "فلفل حار مجفف",
    "Cucumbers and gherkins": "خيار",
    "Dates": "تمر",
    "Figs": "تين",
    "Garlic": "ثوم",
    "Grapes": "عنب",
    "Lemons and limes": "ليمون",
    "Lettuce and chicory": "خس وشيكوريا",
    "Maize": "ذرة",
    "Onions, shallots, green": "بصل أخضر",
    "Onions, dry": "بصل جاف",
    "Oranges": "برتقال",
    "Peas, green": "بازلاء خضراء",
    "Potatoes": "بطاطا",
    "Pumpkins, squash and gourds": "قرع وكوسة",
    "Tomatoes": "طماطم",
    "Watermelons": "بطيخ",
    "Wheat": "قمح"
}

# قاموس عكسي: من العربية إلى الإنجليزية
reverse_forecast_translations = {v: k for k, v in forecast_translations.items()}

# دالة حساب أفضل محصول
crops_data = [
        # ولاية أدرار
        {'ولاية': 'أدرار', 'crop': 'تمور', 'productivity_qx_per_hectare': 934562, 'price_dz_per_qx': 2000, 'cost_dz_per_hectare': 500000},
        {'ولاية': 'أدرار', 'crop': 'حبوب', 'productivity_qx_per_hectare': 732515, 'price_dz_per_qx': 1500, 'cost_dz_per_hectare': 250000},
        {'ولاية': 'أدرار', 'crop': 'خضروات', 'productivity_qx_per_hectare': 695285, 'price_dz_per_qx': 1000, 'cost_dz_per_hectare': 150000},
        {'ولاية': 'أدرار', 'crop': 'طماطم', 'productivity_qx_per_hectare': 170113, 'price_dz_per_qx': 1200, 'cost_dz_per_hectare': 200000},
        {'ولاية': 'أدرار', 'crop': 'بصل', 'productivity_qx_per_hectare': 85560, 'price_dz_per_qx': 800, 'cost_dz_per_hectare': 100000},

        # ولاية الشلف
        {'ولاية': 'الشلف', 'crop': 'حبوب', 'productivity_qx_per_hectare': 2157295, 'price_dz_per_qx': 1200, 'cost_dz_per_hectare': 200000},
        {'ولاية': 'الشلف', 'crop': 'خضروات', 'productivity_qx_per_hectare': 3450680, 'price_dz_per_qx': 900, 'cost_dz_per_hectare': 180000},
        {'ولاية': 'الشلف', 'crop': 'بقوليات', 'productivity_qx_per_hectare': 113291, 'price_dz_per_qx': 1500, 'cost_dz_per_hectare': 220000},
        {'ولاية': 'الشلف', 'crop': 'أشجار مثمرة', 'productivity_qx_per_hectare': 2077030, 'price_dz_per_qx': 2000, 'cost_dz_per_hectare': 350000},
        {'ولاية': 'الشلف', 'crop': 'علف', 'productivity_qx_per_hectare': 1218015, 'price_dz_per_qx': 500, 'cost_dz_per_hectare': 100000},
        {'ولاية': 'الشلف', 'crop': 'طماطم', 'productivity_qx_per_hectare': 340000, 'price_dz_per_qx': 1200, 'cost_dz_per_hectare': 200000},
        {'ولاية': 'الشلف', 'crop': 'بطاطا', 'productivity_qx_per_hectare': 350000, 'price_dz_per_qx': 1100, 'cost_dz_per_hectare': 180000},

        # ولاية الاغواط
        {'ولاية': 'الاغواط', 'crop': 'حبوب', 'productivity_qx_per_hectare': 365852, 'price_dz_per_qx': 1400, 'cost_dz_per_hectare': 200000},
        {'ولاية': 'الاغواط', 'crop': 'أشجار مثمرة', 'productivity_qx_per_hectare': 211807, 'price_dz_per_qx': 2500, 'cost_dz_per_hectare': 300000},
        {'ولاية': 'الاغواط', 'crop': 'خضروات', 'productivity_qx_per_hectare': 2468131, 'price_dz_per_qx': 1000, 'cost_dz_per_hectare': 150000},
        {'ولاية': 'الاغواط', 'crop': 'علف', 'productivity_qx_per_hectare': 776700, 'price_dz_per_qx': 500, 'cost_dz_per_hectare': 100000},

        # ولاية ام البواقي
        {'ولاية': 'ام البواقي', 'crop': 'حبوب', 'productivity_qx_per_hectare': 38500000, 'price_dz_per_qx': 1100, 'cost_dz_per_hectare': 150000},
        {'ولاية': 'ام البواقي', 'crop': 'خضروات', 'productivity_qx_per_hectare': 854000, 'price_dz_per_qx': 800, 'cost_dz_per_hectare': 100000},
        {'ولاية': 'ام البواقي', 'crop': 'علف', 'productivity_qx_per_hectare': 796240, 'price_dz_per_qx': 400, 'cost_dz_per_hectare': 120000},

        # ولاية باتنة
        {'ولاية': 'باتنة', 'crop': 'حبوب', 'productivity_qx_per_hectare': 1503030, 'price_dz_per_qx': 1300, 'cost_dz_per_hectare': 220000},
        {'ولاية': 'باتنة', 'crop': 'خضروات', 'productivity_qx_per_hectare': 1479589, 'price_dz_per_qx': 750, 'cost_dz_per_hectare': 150000},

        # ولاية بجاية
        {'ولاية': 'بجاية', 'crop': 'حبوب', 'productivity_qx_per_hectare': 191961, 'price_dz_per_qx': 1300, 'cost_dz_per_hectare': 180000},
        {'ولاية': 'بجاية', 'crop': 'علف', 'productivity_qx_per_hectare': 963140, 'price_dz_per_qx': 400, 'cost_dz_per_hectare': 120000},
        {'ولاية': 'بجاية', 'crop': 'خضروات', 'productivity_qx_per_hectare': 1517894, 'price_dz_per_qx': 500, 'cost_dz_per_hectare': 100000},

        # ولاية بسكرة
        {'ولاية': 'بسكرة', 'crop': 'تمور', 'productivity_qx_per_hectare': 4380041, 'price_dz_per_qx': 2500, 'cost_dz_per_hectare': 300000},
        {'ولاية': 'بسكرة', 'crop': 'حبوب', 'productivity_qx_per_hectare': 187000, 'price_dz_per_qx': 1400, 'cost_dz_per_hectare': 220000},
        {'ولاية': 'بسكرة', 'crop': 'خضروات', 'productivity_qx_per_hectare': 110000, 'price_dz_per_qx': 700, 'cost_dz_per_hectare': 180000},

        # ولاية وهران
        {'ولاية': 'وهران', 'crop': 'حبوب', 'productivity_qx_per_hectare': 1136330, 'price_dz_per_qx': 1200, 'cost_dz_per_hectare': 250000},
        {'ولاية': 'وهران', 'crop': 'طماطم', 'productivity_qx_per_hectare': 670000, 'price_dz_per_qx': 1500, 'cost_dz_per_hectare': 200000},
        {'ولاية': 'وهران', 'crop': 'خضروات', 'productivity_qx_per_hectare': 820000, 'price_dz_per_qx': 1000, 'cost_dz_per_hectare': 150000},

        # ولاية البيض
        {'ولاية': 'البيض', 'crop': 'حبوب', 'productivity_qx_per_hectare': 225550, 'price_dz_per_qx': 1300, 'cost_dz_per_hectare': 180000},
        {'ولاية': 'البيض', 'crop': 'علف', 'productivity_qx_per_hectare': 142200, 'price_dz_per_qx': 600, 'cost_dz_per_hectare': 120000},

        # ولاية إليزي
        {'ولاية': 'إليزي', 'crop': 'حبوب', 'productivity_qx_per_hectare': 235200, 'price_dz_per_qx': 1500, 'cost_dz_per_hectare': 220000},
        {'ولاية': 'إليزي', 'crop': 'أعلاف', 'productivity_qx_per_hectare': 18061, 'price_dz_per_qx': 500, 'cost_dz_per_hectare': 120000},

        # ولاية غرداية
        {'ولاية': 'غرداية', 'crop': 'تمور', 'productivity_qx_per_hectare': 612000, 'price_dz_per_qx': 2500, 'cost_dz_per_hectare': 300000},
        {'ولاية': 'غرداية', 'crop': 'محاصيل بستانية', 'productivity_qx_per_hectare': 866100, 'price_dz_per_qx': 2000, 'cost_dz_per_hectare': 350000},
        {'ولاية': 'غرداية', 'crop': 'علف', 'productivity_qx_per_hectare': 1227100, 'price_dz_per_qx': 500, 'cost_dz_per_hectare': 100000},

        # ولاية غليزان
        {'ولاية': 'غليزان', 'crop': 'حبوب', 'productivity_qx_per_hectare': 1136330, 'price_dz_per_qx': 1400, 'cost_dz_per_hectare': 250000},
        {'ولاية': 'غليزان', 'crop': 'خضروات', 'productivity_qx_per_hectare': 820000, 'price_dz_per_qx': 1000, 'cost_dz_per_hectare': 150000},

        # ولاية معسكر
        {'ولاية': 'معسكر', 'crop': 'حبوب', 'productivity_qx_per_hectare': 1136330, 'price_dz_per_qx': 1200, 'cost_dz_per_hectare': 250000},

        # ولاية ورقلة
        {'ولاية': 'ورقلة', 'crop': 'حبوب', 'productivity_qx_per_hectare': 96915, 'price_dz_per_qx': 1500, 'cost_dz_per_hectare': 200000},
        {'ولاية': 'ورقلة', 'crop': 'علف', 'productivity_qx_per_hectare': 1971975, 'price_dz_per_qx': 500, 'cost_dz_per_hectare': 100000},

        # ولاية البيض
        {'ولاية': 'البيض', 'crop': 'حبوب', 'productivity_qx_per_hectare': 225550, 'price_dz_per_qx': 1300, 'cost_dz_per_hectare': 180000},
        {'ولاية': 'البيض', 'crop': 'علف', 'productivity_qx_per_hectare': 142200, 'price_dz_per_qx': 600, 'cost_dz_per_hectare': 120000},
        # ولاية سكيكدة
        {'ولاية': 'سكيكدة', 'crop': 'حبوب', 'productivity_qx_per_hectare': 777000, 'price_dz_per_qx': 1300, 'cost_dz_per_hectare': 220000},
        {'ولاية': 'سكيكدة', 'crop': 'علف', 'productivity_qx_per_hectare': 476544, 'price_dz_per_qx': 600, 'cost_dz_per_hectare': 100000},
        {'ولاية': 'سكيكدة', 'crop': 'خضروات', 'productivity_qx_per_hectare': 7200, 'price_dz_per_qx': 900, 'cost_dz_per_hectare': 150000},
        {'ولاية': 'سكيكدة', 'crop': 'محاصيل صناعية', 'productivity_qx_per_hectare': 350000, 'price_dz_per_qx': 2000, 'cost_dz_per_hectare': 350000},

        # ولاية سيدي بلعباس
        {'ولاية': 'سيدي بلعباس', 'crop': 'حبوب', 'productivity_qx_per_hectare': 1491678, 'price_dz_per_qx': 1300, 'cost_dz_per_hectare': 220000},
        {'ولاية': 'سيدي بلعباس', 'crop': 'علف', 'productivity_qx_per_hectare': 381620, 'price_dz_per_qx': 600, 'cost_dz_per_hectare': 120000},
        {'ولاية': 'سيدي بلعباس', 'crop': 'خضروات', 'productivity_qx_per_hectare': 2385380, 'price_dz_per_qx': 500, 'cost_dz_per_hectare': 180000},

        # ولاية عنابة
        {'ولاية': 'عنابة', 'crop': 'حبوب', 'productivity_qx_per_hectare': 459600, 'price_dz_per_qx': 1200, 'cost_dz_per_hectare': 200000},
        {'ولاية': 'عنابة', 'crop': 'كروم', 'productivity_qx_per_hectare': 4155, 'price_dz_per_qx': 1500, 'cost_dz_per_hectare': 220000},
        {'ولاية': 'عنابة', 'crop': 'خضروات', 'productivity_qx_per_hectare': 3300500, 'price_dz_per_qx': 800, 'cost_dz_per_hectare': 120000},

        # ولاية قالمة
        {'ولاية': 'قالمة', 'crop': 'حبوب', 'productivity_qx_per_hectare': 2140900, 'price_dz_per_qx': 1300, 'cost_dz_per_hectare': 180000},
        {'ولاية': 'قالمة', 'crop': 'زيتون', 'productivity_qx_per_hectare': 72275, 'price_dz_per_qx': 2500, 'cost_dz_per_hectare': 350000},
        {'ولاية': 'قالمة', 'crop': 'خضروات', 'productivity_qx_per_hectare': 103616, 'price_dz_per_qx': 700, 'cost_dz_per_hectare': 150000},

        # ولاية قسنطينة
        {'ولاية': 'قسنطينة', 'crop': 'حبوب', 'productivity_qx_per_hectare': 2657140, 'price_dz_per_qx': 1300, 'cost_dz_per_hectare': 220000},
        {'ولاية': 'قسنطينة', 'crop': 'خضروات', 'productivity_qx_per_hectare': 500000, 'price_dz_per_qx': 900, 'cost_dz_per_hectare': 180000},
        {'ولاية': 'قسنطينة', 'crop': 'أشجار مثمرة', 'productivity_qx_per_hectare': 850000, 'price_dz_per_qx': 1500, 'cost_dz_per_hectare': 350000},

        # ولاية المدية
        {'ولاية': 'المدية', 'crop': 'حبوب', 'productivity_qx_per_hectare': 113400, 'price_dz_per_qx': 1400, 'cost_dz_per_hectare': 200000},
        {'ولاية': 'المدية', 'crop': 'أعلاف', 'productivity_qx_per_hectare': 285000, 'price_dz_per_qx': 500, 'cost_dz_per_hectare': 150000},
        {'ولاية': 'المدية', 'crop': 'خضراوات', 'productivity_qx_per_hectare': 244000, 'price_dz_per_qx': 1000, 'cost_dz_per_hectare': 120000},
        {'ولاية': 'المدية', 'crop': 'بطاطا', 'productivity_qx_per_hectare': 350000, 'price_dz_per_qx': 1200, 'cost_dz_per_hectare': 180000},
        {'ولاية': 'المدية', 'crop': 'كروم', 'productivity_qx_per_hectare': 200000, 'price_dz_per_qx': 1000, 'cost_dz_per_hectare': 120000},
        # ولاية مستغانم
        {'ولاية': 'مستغانم', 'crop': 'حبوب', 'productivity_qx_per_hectare': 118212, 'price_dz_per_qx': 1400, 'cost_dz_per_hectare': 250000},
        {'ولاية': 'مستغانم', 'crop': 'بطاطا', 'productivity_qx_per_hectare': 355730, 'price_dz_per_qx': 1200, 'cost_dz_per_hectare': 180000},
        {'ولاية': 'مستغانم', 'crop': 'خضروات', 'productivity_qx_per_hectare': 295000, 'price_dz_per_qx': 1000, 'cost_dz_per_hectare': 150000},

        # ولاية المسيلة
        {'ولاية': 'المسيلة', 'crop': 'حبوب', 'productivity_qx_per_hectare': 175729, 'price_dz_per_qx': 1300, 'cost_dz_per_hectare': 220000},
        {'ولاية': 'المسيلة', 'crop': 'علف', 'productivity_qx_per_hectare': 175000, 'price_dz_per_qx': 600, 'cost_dz_per_hectare': 150000},

        # ولاية معسكر
        {'ولاية': 'معسكر', 'crop': 'حبوب', 'productivity_qx_per_hectare': 1136330, 'price_dz_per_qx': 1200, 'cost_dz_per_hectare': 250000},

         {'ولاية': 'بشار', 'crop': 'تمور', 'productivity_qx_per_hectare': 398130, 'price_dz_per_qx': 2500, 'cost_dz_per_hectare': 300000},
         {'ولاية': 'بشار', 'crop': 'حبوب', 'productivity_qx_per_hectare': 53123, 'price_dz_per_qx': 1400, 'cost_dz_per_hectare': 220000},
         {'ولاية': 'بشار', 'crop': 'بقوليات', 'productivity_qx_per_hectare': 557421, 'price_dz_per_qx': 1500, 'cost_dz_per_hectare': 250000},
         {'ولاية': 'بشار', 'crop': 'حمضيات', 'productivity_qx_per_hectare': 6673, 'price_dz_per_qx': 2000, 'cost_dz_per_hectare': 280000},
         {'ولاية': 'بشار', 'crop': 'زيتون', 'productivity_qx_per_hectare': 9962, 'price_dz_per_qx': 2500, 'cost_dz_per_hectare': 350000},
        # ولاية ورقلة
        {'ولاية': 'ورقلة', 'crop': 'حبوب', 'productivity_qx_per_hectare': 96915, 'price_dz_per_qx': 1500, 'cost_dz_per_hectare': 200000},
        {'ولاية': 'ورقلة', 'crop': 'علف', 'productivity_qx_per_hectare': 1971975, 'price_dz_per_qx': 500, 'cost_dz_per_hectare': 100000},
        
        # ولاية سطيف
        {'ولاية': 'سطيف', 'crop': 'حبوب', 'productivity_qx_per_hectare': 196015, 'price_dz_per_qx': 1300, 'cost_dz_per_hectare': 220000},
        {'ولاية': 'سطيف', 'crop': 'علف', 'productivity_qx_per_hectare': 27429, 'price_dz_per_qx': 500, 'cost_dz_per_hectare': 100000},
        {'ولاية': 'سطيف', 'crop': 'خضروات', 'productivity_qx_per_hectare': 2563420, 'price_dz_per_qx': 700, 'cost_dz_per_hectare': 150000},
        
        # ولاية تلمسان
        {'ولاية': 'تلمسان', 'crop': 'حبوب', 'productivity_qx_per_hectare': 1886900, 'price_dz_per_qx': 1300, 'cost_dz_per_hectare': 200000},
        {'ولاية': 'تلمسان', 'crop': 'علف', 'productivity_qx_per_hectare': 1467200, 'price_dz_per_qx': 600, 'cost_dz_per_hectare': 120000},
        {'ولاية': 'تلمسان', 'crop': 'خضروات', 'productivity_qx_per_hectare': 1500000, 'price_dz_per_qx': 800, 'cost_dz_per_hectare': 180000},

         # ولاية تيبازة
        {'ولاية': 'تيبازة', 'crop': 'حبوب', 'productivity_qx_per_hectare': 2015000, 'price_dz_per_qx': 1300, 'cost_dz_per_hectare': 220000},
        {'ولاية': 'تيبازة', 'crop': 'خضروات', 'productivity_qx_per_hectare': 1200000, 'price_dz_per_qx': 700, 'cost_dz_per_hectare': 150000},
        {'ولاية': 'تيبازة', 'crop': 'زيتون', 'productivity_qx_per_hectare': 30000, 'price_dz_per_qx': 2500, 'cost_dz_per_hectare': 350000},
         # ولاية تيسمسيلت
        {'ولاية': 'تيسمسيلت', 'crop': 'حبوب', 'productivity_qx_per_hectare': 250000, 'price_dz_per_qx': 1200, 'cost_dz_per_hectare': 220000},
        {'ولاية': 'تيسمسيلت', 'crop': 'أعلاف', 'productivity_qx_per_hectare': 600000, 'price_dz_per_qx': 500, 'cost_dz_per_hectare': 120000},
         # ولاية الجلفة
        {'ولاية': 'الجلفة', 'crop': 'حبوب', 'productivity_qx_per_hectare': 500000, 'price_dz_per_qx': 1200, 'cost_dz_per_hectare': 250000},
        {'ولاية': 'الجلفة', 'crop': 'علف', 'productivity_qx_per_hectare': 700000, 'price_dz_per_qx': 400, 'cost_dz_per_hectare': 120000},
        # ولاية الوادي
        {'ولاية': 'الوادي', 'crop': 'حبوب', 'productivity_qx_per_hectare': 800000, 'price_dz_per_qx': 1300, 'cost_dz_per_hectare': 250000},
        {'ولاية': 'الوادي', 'crop': 'زيتون', 'productivity_qx_per_hectare': 120000, 'price_dz_per_qx': 2500, 'cost_dz_per_hectare': 300000},
        {'ولاية': 'الوادي', 'crop': 'تمور', 'productivity_qx_per_hectare': 500000, 'price_dz_per_qx': 2000, 'cost_dz_per_hectare': 280000},
        {'ولاية': 'الوادي', 'crop': 'خضروات', 'productivity_qx_per_hectare': 1200000, 'price_dz_per_qx': 700, 'cost_dz_per_hectare': 150000},

    ]


df = pd.DataFrame(crops_data)

df.columns = df.columns.astype(str).str.strip()


def get_best_crop(state_name):
    # إزالة المسافات وتحويل الاسم إلى صيغة موحدة (حروف صغيرة)
    state_name = state_name.strip().lower()

    # تصفية البيانات حسب اسم الولاية
    df['ولاية'] = df['ولاية'].str.strip().str.lower()  # إزالة المسافات وتحويل الحروف إلى حروف صغيرة
    state_data = df[df['ولاية'] == state_name]

    if not state_data.empty:
        # حساب القيمة السوقية والربح الصافي
        state_data['market_value'] = state_data['productivity_qx_per_hectare'] * state_data['price_dz_per_qx']
        state_data['net_profit'] = state_data['market_value'] - state_data['cost_dz_per_hectare']

        # تحويل القيم إلى int عادي (تجنب int64)
        state_data['net_profit'] = state_data['net_profit'].astype(int)

        # ترتيب المحاصيل حسب الربح
        state_data_sorted = state_data.sort_values(by='net_profit', ascending=False)

        # المحصول الأنسب
        best_crop = state_data_sorted.iloc[0]
        best_crop_result = {
            'crop': best_crop['crop'],
            'net_profit': int(best_crop['net_profit'])  # تحويل إلى int هنا أيضًا
        }

        # عرض المحاصيل مرتبة حسب الربح في الـ console
        print(f"🔝 المحاصيل في ولاية {state_name} مرتبة حسب الربح:")
        print(state_data_sorted[['crop', 'net_profit']])

        # عرض المحصول الأنسب مع الربح الصافي
        print(f"\n✅ المحصول الأنسب اقتصادياً في ولاية {state_name}: {best_crop_result['crop']} بربح صافٍ يقدر بـ {int(best_crop_result['net_profit']):,} دج")

        # إرجاع المحاصيل مرتبة والمحصول الأنسب
        return {
            'ranked_crops': state_data_sorted[['crop', 'net_profit']].to_dict(orient='records'),
            'best_crop': best_crop_result
        }
    else:
        return {'error': f"لا توجد بيانات للولاية {state_name}"}

# إنشاء تطبيق Flask
app = Flask(__name__)
CORS(app)


# ✅ endpoint 1: توصية بالمحاصيل
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array([
        data['N'],
        data['P'],
        data['K'],
        data['temperature'],
        data['humidity'],
        data['ph'],
        data['rainfall']
    ]).reshape(1, -1)

    probabilities = model.predict_proba(input_data)[0]
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_crops = label_encoder.inverse_transform(top_indices)

    results = [
        {
            'name': translations.get(crop, crop),
            'percentage': round(probabilities[i] * 100, 2)
        }
        for i, crop in zip(top_indices, top_crops)
    ]
    return jsonify({'recommended_crops': results})

# ✅ endpoint 2: توقع إنتاج محصول لسنوات قادمة
@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.get_json()
    user_input = data.get('crop')
    crop = reverse_forecast_translations.get(user_input, user_input)
    years = data.get('years', 5)

    try:
        if crop not in forecast_data_dict:
            return jsonify({'error': 'المحصول غير موجود في البيانات'}), 404

        series = forecast_data_dict[crop]
        last_year = series.index.max()
        future_years = [last_year + i for i in range(1, years + 1)]
        full_range = np.arange(len(series) + years).reshape(-1, 1)

        predictions = forecast_model.predict(full_range)[-years:]

        results = {
            str(int(year)): round(value, 2)
            for year, value in zip(future_years, predictions)
        }

        translated_crop = forecast_translations.get(crop, crop)

        return jsonify({'crop': translated_crop, 'forecast': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ✅ endpoint 3: الحصول على أفضل محصول في ولاية معينة
@app.route('/best_crop', methods=['POST'])
def best_crop():
    data = request.get_json()
    state_name = data.get('state_name')

    if not state_name:
        return jsonify({'error': 'الرجاء إدخال اسم الولاية'}), 400

    # استدعاء دالة get_best_crop لحساب أفضل محصول للولاية
    result = get_best_crop(state_name)

    # إرجاع النتيجة للمستخدم
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
