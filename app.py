from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib

# Fix for Python 3.14 / scikit-learn unpickling compatibility
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

from growth_data import GROWTH_INSIGHTS

app = Flask(__name__)

# Load the trained Kolkata estate valuation pipeline
model = joblib.load('house_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Build DataFrame with column names matching the model's feature expectations
        input_df = pd.DataFrame([{
            'Locality': str(data['Locality']),
            'BHK': int(data['BHK']),
            'Area_SqFt': float(data['Area_SqFt']),
            'Bathrooms': int(data['Bathrooms']),
            'Furnishing': str(data['Furnishing']),
            'Property_Age_Years': int(data['Property_Age_Years']),
            'Gated_Security': int(data['Gated_Security']),         # 1 or 0
            'Parking_Available': int(data['Parking_Available']),   # 1 or 0
            'Gymnasium': int(data['Gymnasium']),                   # 1 or 0
            'Swimming_Pool': int(data['Swimming_Pool']),           # 1 or 0
            'Lift_Available': int(data['Lift_Available'])          # 1 or 0
        }])
        
        # Run inference via the pipeline
        prediction = model.predict(input_df)[0]
        
        # Guardrail floor value (5 Lakhs minimum)
        final_price = max(5.0, float(prediction))
        
        return jsonify({'predicted_price': round(final_price, 2)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/future-price-forecast', methods=['GET'])
def future_price_forecast():
    locality = request.args.get('locality')

    if not locality:
        return jsonify({"status": "missing", "message": "Locality required."})

    insight = GROWTH_INSIGHTS.get(locality)

    if insight:
        reasons_text = " ".join(insight['detailed_reasons']).lower()
        drivers = []
        score = 0

        if 'metro' in reasons_text:
            score += 28
            drivers.append('Metro connectivity')
        if 'highway' in reasons_text or 'expressway' in reasons_text or 'bypass' in reasons_text:
            score += 25
            drivers.append('Highway and arterial access')
        if 'mall' in reasons_text or 'retail' in reasons_text or 'commercial' in reasons_text:
            score += 18
            drivers.append('Retail and commercial density')
        if 'airport' in reasons_text:
            score += 12
            drivers.append('Airport connectivity')
        if 'tech' in reasons_text or 'medical' in reasons_text or 'education' in reasons_text:
            score += 10
            drivers.append('Employment and lifestyle demand')
        if 'smart city' in reasons_text or 'redevelopment' in reasons_text or 'waterfront' in reasons_text:
            score += 10
            drivers.append('Urban redevelopment drivers')

        score = min(score, 95)

        if score >= 75:
            forecast_status = 'Likely Increase'
            probability = 84
        elif score >= 58:
            forecast_status = 'Probable Increase'
            probability = 70
        elif score >= 40:
            forecast_status = 'Moderate Upside'
            probability = 55
        else:
            forecast_status = 'Uncertain'
            probability = 42

        roi_text = insight['projected_roi']
        if 'to' in roi_text:
            nums = [int(n) for n in roi_text.replace('%', '').replace('+', '').split(' to ')]
            avg_growth = sum(nums) / len(nums)
        else:
            avg_growth = int(roi_text.replace('%', '').replace('+', ''))

        return jsonify({
            'status': 'found',
            'locality': locality,
            'forecast_status': forecast_status,
            'probability': probability,
            'expected_growth_pct': round(avg_growth, 1),
            'drivers': drivers,
            'headline': insight['catalyst_headline'],
            'current_avg_lakhs': insight['current_avg_lakhs'],
            'entry_tier': insight['entry_tier'],
            'summary': f"{locality} is supported by {', '.join(drivers[:3]) if drivers else 'strong demand fundamentals'} and is likely to trend upward over the next cycle."
        })

    # Generic Kolkata fallback for localities not explicitly listed in growth_data
    fallback_score = 50
    generic_drivers = ['Transit access', 'Road connectivity', 'Urban amenity profile']
    return jsonify({
        'status': 'generic',
        'locality': locality,
        'forecast_status': 'Moderate Upside',
        'probability': 58,
        'expected_growth_pct': 12,
        'drivers': generic_drivers,
        'headline': 'Balanced city growth profile',
        'current_avg_lakhs': 0,
        'entry_tier': 'Emerging market',
        'summary': f"{locality} has a moderate upside profile, driven by Kolkata's broader transit expansion, central business density, and housing demand cycle."
    })

@app.route('/growth-radar', methods=['GET'])
def growth_radar():
    locality = request.args.get('locality')
    
    if locality:
        insight = GROWTH_INSIGHTS.get(locality)
        if insight:
            return jsonify({
                "status": "found",
                "locality": locality,
                "data": insight
            })
        return jsonify({
            "status": "not_found",
            "message": "No specific speculative catalyst logged for this area."
        })
    
    # Return all 15 corridors sorted by current average price
    sorted_corridors = sorted(
        GROWTH_INSIGHTS.items(),
        key=lambda item: item[1]['current_avg_lakhs']
    )
    return jsonify({
        "status": "all",
        "corridors": dict(sorted_corridors)
    })

if __name__ == '__main__':
    app.run(debug=True)