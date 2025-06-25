from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS


# تحميل النموذج الأول والـ LabelEncoder
model = pickle.load(open('crop_model_v2.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder_v2.pkl', 'rb'))

# تحميل نموذج التنبؤ المستقبلي
forecast_model = pickle.load(open('forecast_model_new.pkl', 'rb'))
# تحميل بيانات السلاسل الزمنية
forecast_data_dict = {
    "Almonds": [339953, 340171, 340167, 340167, 350167, 340167],
    "Apples": [41233352, 48754374, 45160942, 46618428, 48812048, 47752730],
    "Apricots": [2427646, 2972538, 4105570, 3282676, 3823991, 3712624],
    "Artichokes": [1718608, 927436, 1300325, 1124578, 1207410, 1168370],
    "Barley": [13176955, 10415608, 10171014, 9523834, 9862435, 9674468],
    "Beans, dry": [53485, 51774, 51782, 58045, 58595, 59952],
    "Broad beans and horse beans, dry": [793801, 637527, 651317, 526018, 560635, 544721],
    "Carrots and turnips": [8639199, 4595139, 6529000, 5604232, 6046454, 5834984],
    "Chick peas, dry": [1270330, 709330, 963113, 848308, 900243, 876749],
    "Chillies and peppers, dry": [606827, 318878, 458530, 390800, 423649, 407718],
    "Dates": [220429806, 215706035, 266647911, 241867723, 253921808, 248058214],
    "Figs": [86072, 86011, 86006, 86005, 86005, 86005],
    "Grapes": [28129991, 24220518, 29694355, 26390986, 27851046, 27205712],
    "Hen eggs in shell, fresh": [29164081, 15553760, 22037355, 18948743, 20420076, 19719172],
    "Horse meat, fresh or chilled (indigenous)": [-4, 0, -4, -1, -3, -1],
    "Lemons and limes": [5037232, 2663560, 3806711, 3256174, 3521310, 3393622],
    "Lentils, dry": [296147, 193446, 230104, 217019, 221690, 220023],
    "Locust beans (carobs)": [932, 1004, 933, 1003, 934, 1002],
    "Maize (corn)": [128776, 81041, 99196, 92291, 94918, 93919],
    "Meat of camels, fresh or chilled": [9205503, 95675, 9205465, 95713, 9205428, 95750],
    "Meat of camels, fresh or chilled (indigenous)": [4300881, 4256973, 3250390, 2767552, 3002229, 2888167],
    "Meat of cattle with the bone, fresh or chilled": [107333380, 1131211, 107333361, 1131229, 107333343, 1131248],
    "Meat of cattle with the bone, fresh or chilled (indigenous)": [49754570, 26096612, 37586379, 32006238, 34716299, 33400126],
    "Meat of chickens, fresh or chilled": [108266237, 1133991, 108265953, 1134276, 108265668, 1134560],
    "Meat of chickens, fresh or chilled (indigenous)": [49968847, 26334733, 37769143, 32237066, 34913538, 33618635],
    "Meat of goat, fresh or chilled": [32452071, 338128, 32452069, 338129, 32452067, 338131],
    "Meat of goat, fresh or chilled (indigenous)": [15230982, 7970951, 11510101, 9784822, 10625868, 10215871],
    "Meat of pig with the bone, fresh or chilled (indigenous)": [212, 212, 213, 213, 213, 213],
    "Meat of rabbits and hares, fresh or chilled (indigenous)": [19239, 19320, 19399, 19477, 19553, 19628],
    "Meat of sheep, fresh or chilled": [405283865, 4249548, 405283394, 4250019, 405282923, 4250490],
    "Meat of sheep, fresh or chilled (indigenous)": [187399398, 98314440, 141645676, 120569211, 130820877, 125834431],
    "Meat of turkeys, fresh or chilled (indigenous)": [79020, 78936, 78931, 78930, 78930, 78930],
    "Oats": [581504, 359023, 446053, 412008, 425326, 420116],
    "Olives": [17113190, 9486791, 12965462, 11378717, 12102488, 11772350],
    "Onions and shallots, dry": [34206550, 17949303, 25847514, 22010349, 23874548, 22968870],
    "Oranges": [64066725, 33819720, 48444102, 41373235, 44791989, 43139026],
    "Other beans, green": [6260726, 3313837, 4733811, 4049589, 4379285, 4220419],
    "Other vegetables, fresh n.e.c.": [25010348, 13261133, 18912153, 16194182, 17501445, 16872690],
    "Peaches and nectarines": [2733527, 1449314, 2065979, 1769864, 1912055, 1843776],
    "Pears": [11558986, 6234268, 8734757, 7560527, 8111946, 7852999],
    "Peas, dry": [234972, 131897, 178246, 157405, 166776, 162562],
    "Plums and sloes": [1569975, 842858, 1186495, 1024091, 1100844, 1064570],
    "Pomelos and grapefruits": [703, 706, 707, 707, 707, 707],
    "Potatoes": [86703164, 46178655, 65531154, 56289359, 60702782, 58595150],
    "Raw milk of camel": [9738, 9829, 9919, 10008, 10096, 10183],
    "Raw milk of cattle": [51360109, 27061092, 38794547, 33128723, 35864623, 34543518],
    "Raw milk of goats": [148860, 148858, 148858, 148858, 148858, 148858],
    "Raw milk of sheep": [343163, 347727, 352092, 356267, 360262, 364083],
    "Rice": [3228, 3905, 4450, 4317, 4496, 4422],
    "Seed cotton, unginned": [3085, 2400, 3110, 2811, 2937, 2884],
    "Shorn wool, greasy": [138030, 139679, 141329, 142979, 144628, 146278],
    "Sorghum": [8741, 6483, 7077, 6921, 6962, 6951],
    "Sugar beet": [6009, 5276, 6009, 5276, 6009, 5276],
    "Sunflower seed": [1955, 1204, 1499, 1383, 1429, 1411],
    "Tangerines, mandarins, clementines": [16092636, 9525481, 12169442, 10414694, 11259692, 10852783],
    "Tomatoes": [37837387, 36288256, 38617661, 34664247, 36540670, 35650056],
    "Unmanufactured tobacco": [127988, 71796, 97010, 85696, 90773, 88495],
    "Vetches": [980, 582, 747, 678, 707, 695],
    "Watermelons": [31662684, 31983663, 31503039, 36898210, 39125714, 38048199],
    "Wheat": [41793022, 45181711, 46865127, 41735646, 43987691, 42998954]
}

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
     "Almonds": "لوز",
    "Apples": "تفاح",
    "Apricots": "مشمش",
    "Artichokes": "خرشف",
    "Barley": "شعير",
    "Beans, dry": "فول",
    "Broad beans and horse beans, dry": "فاصوليا",
    "Carrots and turnips": "جزر",
    "Chick peas, dry": "حمص",
    "Chillies and peppers, dry": "فلفل حار",
    "Dates": "تمر",
    "Figs": "تين",
    "Grapes": "عنب",
    "Hen eggs in shell, fresh": "بيض دجاج",
    "Horse meat, fresh or chilled (indigenous)": "لحم غنم",
    "Lemons and limes": "ليمون",
    "Lentils, dry": "عدس جاف",
    "Locust beans (carobs)": "الخرنوب",
    "Maize (corn)": "ذرة",
    "Meat of camels, fresh or chilled": "لحم الجمال طازج أو مبرد",
    "Meat of camels, fresh or chilled (indigenous)": "لحم الجمال طازج أو مبرد (محلي)",
    "Meat of cattle with the bone, fresh or chilled": "لحم الأبقار بالعظم طازج أو مبرد",
    "Meat of cattle with the bone, fresh or chilled (indigenous)": "لحم الأبقار بالعظم طازج أو مبرد (محلي)",
    "Meat of chickens, fresh or chilled": "لحم الدجاج",
    "Meat of chickens, fresh or chilled (indigenous)": "لحم الدجاج طازج أو مبرد (محلي)",
    "Meat of goat, fresh or chilled": "لحم الماعز طازج أو مبرد",
    "Meat of goat, fresh or chilled (indigenous)": "لحم الماعز طازج أو مبرد (محلي)",
    "Meat of pig with the bone, fresh or chilled (indigenous)": "لحم الخنزير بالعظم طازج أو مبرد (محلي)",
    "Meat of rabbits and hares, fresh or chilled (indigenous)": "لحم الأرانب والطقس طازج أو مبرد (محلي)",
    "Meat of sheep, fresh or chilled": "لحم الضأن طازج أو مبرد",
    "Meat of sheep, fresh or chilled (indigenous)": "لحم الضأن طازج أو مبرد (محلي)",
    "Meat of turkeys, fresh or chilled (indigenous)": "لحم الديك الرومي طازج أو مبرد (محلي)",
    "Oats": "شوفان",
    "Olives": "زيتون",
    "Onions and shallots, dry": "بصل وجبنة جافة",
    "Oranges": "برتقال",
    "Other beans, green": "فاصوليا خضراء",
    "Other vegetables, fresh n.e.c.": "لفت",
    "Peaches and nectarines": "خوخ و نكتارين",
    "Pears": "كمثرى",
    "Peas, dry": "بازلاء جافة",
    "Plums and sloes": "برقوق وخوخ بري",
    "Pomelos and grapefruits": "جريب فروت",
    "Potatoes": "بطاطا",
    "Raw milk of camel": "حليب الجمل الخام",
    "Raw milk of cattle": "حليب الأبقار الخام",
    "Raw milk of goats": "حليب الماعز الخام",
    "Raw milk of sheep": "حليب الأغنام الخام",
    "Rice": "أرز",
    "Seed cotton, unginned": "قطن غير مفروم",
    "Shorn wool, greasy": "صوف جزئ رطب",
    "Sorghum": "سرغوم",
    "Sugar beet": "شمندر السكر",
    "Sunflower seed": "بذور عباد الشمس",
    "Tangerines, mandarins, clementines": "ماندارين، كليمينتين",
    "Tomatoes": "طماطم",
    "Unmanufactured tobacco": "تبغ غير مصنع",
    "Vetches": "بقول",
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

@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.get_json()
    user_input = data.get('crop')  # المحصول المدخل بالعربية
    years = data.get('years', 5)

    # استخدام القاموس العكسي لتحويل المحصول من العربية إلى الإنجليزية
    crop = reverse_forecast_translations.get(user_input, user_input)

    if crop not in forecast_data_dict:
        return jsonify({'error': f"المحصول '{user_input}' غير موجود في البيانات."}), 404

    crop_data = forecast_data_dict[crop]

    if len(crop_data) == 0:
        return jsonify({'error': f"لا توجد بيانات للمحصول '{user_input}'."}), 400

    # تحديد السنوات المستقبلية
    future_years = [2024 + i for i in range(1, years + 1)]
    forecast_result = {str(year): round(crop_data[i], 2) for i, year in enumerate(future_years)}

    return jsonify({'crop': crop, 'forecast': forecast_result})

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
    app.run(host='0.0.0.0', port=5000, debug=True)
