from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

# تحميل النموذج والـ LabelEncoder
model = pickle.load(open('crop_model_v2.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder_v2.pkl', 'rb'))
# قاموس الترجمة: من الإنجليزية إلى العربية
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
    "chickpea":"بامية، حمص، عدس، فول",
    "coffee": "زعتر، نعناع، خروب، شيح"
}


app = Flask(__name__)
CORS(app)

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

    # توقع النسب
    probabilities = model.predict_proba(input_data)[0]
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_crops = label_encoder.inverse_transform(top_indices)

    # ترجمة النتائج
    results = [
        {
            'name': translations.get(crop, crop),
            'percentage': round(probabilities[i] * 100, 2)
        }
        for i, crop in zip(top_indices, top_crops)
    ]

    return jsonify({'recommended_crops': results})

if __name__ == '__main__':
    app.run(debug=True)