from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

# Load your models
rice_model = joblib.load('gbr_price_predictor.pkl')
onion_model = joblib.load('price_predictor.pkl')

# Market name mapping (User-facing name -> Feature name)
# These dictionaries define the mapping from the user-friendly market names
# (used in HTML dropdowns) to the exact feature names expected by your models.
# It's CRUCIAL that the 'market_name_...' values here precisely match
# the columns your models were trained on.
ONION_MARKET_MAP = {
    'Angul': 'market_name_Angul',
    'Angul (Jarapada)': 'market_name_Angul(Jarapada)',
    'Attabira': 'market_name_Attabira',
    'Balugaon': 'market_name_Balugaon',
    'Bargarh': 'market_name_Bargarh',
    'Bargarh (Barapalli)': 'market_name_Bargarh(Barapalli)',
    'Bhadrak': 'market_name_Bhadrak',
    'Bhanjanagar': 'market_name_Bhanjanagar',
    'Bhawanipatna': 'market_name_Bhawanipatna',
    'Birmaharajpur': 'market_name_Birmaharajpur',
    'Bolangir': 'market_name_Bolangir',
    'Bonai': 'market_name_Bonai',
    'Boudh': 'market_name_Boudh',
    'Chandabali': 'market_name_Chandabali',
    'Chatta Krushak Bazar': 'market_name_Chatta Krushak Bazar',
    'Chuliaposi': 'market_name_Chuliaposi',
    'Damana Hat': 'market_name_Damana Hat',
    'Dhenkanal': 'market_name_Dhenkanal',
    'Dungurapalli': 'market_name_Dungurapalli',
    'Godabhaga': 'market_name_Godabhaga',
    'Gopa': 'market_name_Gopa',
    'Hindol': 'market_name_Hindol',
    'Hinjilicut': 'market_name_Hinjilicut',
    'Jaleswar': 'market_name_Jaleswar',
    'Jatni': 'market_name_Jatni',
    'Jharsuguda': 'market_name_Jharsuguda',
    'Kalahandi (Dharamagarh)': 'market_name_Kalahandi(Dharamagarh)',
    'Kamakhyanagar': 'market_name_Kamakhyanagar',
    'Kantabaji': 'market_name_Kantabaji',
    'Kendrapara': 'market_name_Kendrapara',
    'Kendrapara (Marshaghai)': 'market_name_Kendrapara(Marshaghai)',
    'Keonjhar': 'market_name_Keonjhar',
    'Keonjhar (Dhekikote)': 'market_name_Keonjhar(Dhekikote)',
    'Kesinga': 'market_name_Kesinga',
    'Khariar': 'market_name_Khariar',
    'Khariar Road': 'market_name_Khariar Road',
    'Khunthabandha': 'market_name_Khunthabandha',
    'Kuchinda': 'market_name_Kuchinda',
    'Mottagaon': 'market_name_Mottagaon',
    'Nilagiri': 'market_name_Nilagiri',
    'Pandkital': 'market_name_Pandkital',
    'Pattamundai': 'market_name_Pattamundai',
    'Rayagada': 'market_name_Rayagada',
    'Saharpada': 'market_name_Saharpada',
    'Sargipali': 'market_name_Sargipali',
    'Talcher': 'market_name_Talcher',
    'Tusura': 'market_name_Tusura',
    'Udala': 'market_name_Udala',
}

RICE_MARKET_MAP = {
    'Bonai': 'market_name_Bonai',
    'Jharsuguda': 'market_name_Jharsuguda',
    'Karanjia': 'market_name_Karanjia',
    'Keonjhar': 'market_name_Keonjhar',
    'Keonjhar (Dhekikote)': 'market_name_Keonjhar(Dhekikote)',
    'Kesinga': 'market_name_Kesinga',
    'Khunthabandha': 'market_name_Khunthabandha',
    'Nawarangpur': 'market_name_Nawarangpur',
    'Rahama': 'market_name_Rahama',
    'Saharpada': 'market_name_Saharpada',
    'Sohela': 'market_name_Sohela',
    'Tusura': 'market_name_Tusura',
    'Udala': 'market_name_Udala',
}

# Define the full list of feature columns for each model.
# These lists are constructed from 'day', 'month', 'year', and the values from the market maps.
# Their order and content must EXACTLY match what your respective models expect.
RICE_FEATURES = ['day', 'month', 'year'] + list(RICE_MARKET_MAP.values())
ONION_FEATURES = ['day', 'month', 'year'] + list(ONION_MARKET_MAP.values())

# User-facing lists of market names for the dropdowns
RICE_MARKETS = list(RICE_MARKET_MAP.keys())
ONION_MARKETS = list(ONION_MARKET_MAP.keys())
COMMODITIES = ['rice', 'onion']

# Helper function to generate a date range
def generate_date_range(start_date, end_date):
    return [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

# Helper function to get monthly average prices for recommendations
def get_monthly_avg_prices(commodity, market):
    # Select the correct features, market map, and model based on commodity
    features = RICE_FEATURES if commodity == 'rice' else ONION_FEATURES
    market_map = RICE_MARKET_MAP if commodity == 'rice' else ONION_MARKET_MAP
    model = rice_model if commodity == 'rice' else onion_model

    # Get the feature name for the selected market
    market_col = market_map.get(market)

    # Return an error if the market isn't found in the map for the given commodity
    if not market_col:
        return {'error': f'Market "{market}" is not recognized for {commodity}.'}

    monthly_avg = {}
    # Use the current year for forecasting monthly averages
    current_year = datetime.now().year

    # Iterate through each month of the year
    for month in range(1, 13):
        prices_for_month = []
        # Predict for a few days (1st, 10th, 20th) within the month to get a representative average.
        # This makes the monthly average more robust than predicting for just one day.
        for day in [1, 10, 20]:
            # Simple check to avoid invalid days for months (e.g., Feb 30) - model should handle if trained on sparse data
            if not (1 <= day <= 31):
                continue

            # Initialize input data dictionary with all features set to 0
            input_data = dict.fromkeys(features, 0)
            input_data['day'] = day
            input_data['month'] = month
            input_data['year'] = current_year
            input_data[market_col] = 1 # Set the specific market's dummy variable to 1

            # Create a Pandas DataFrame from the input data.
            # IMPORTANT: Ensure the columns of the DataFrame are in the exact order
            # expected by the model, using `[features]`.
            input_df = pd.DataFrame([input_data])[features]

            try:
                # Get the predicted price from the model
                predicted_price = model.predict(input_df)[0]
                prices_for_month.append(predicted_price)
            except Exception as e:
                # Log any prediction errors but continue
                print(f"Error predicting for {commodity} in {market} on {month}/{day}/{current_year}: {e}")
                pass # Don't add to prices_for_month if prediction fails

        # Calculate the average price for the month if there are valid predictions
        if prices_for_month:
            monthly_avg[month] = sum(prices_for_month) / len(prices_for_month)
        else:
            monthly_avg[month] = 0 # If no valid predictions for the month, set average to 0

    return monthly_avg

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def home():
    # Set default commodity and market for the initial page load
    commodity = 'onion'
    market = 'Angul' # Make sure 'Angul' is in ONION_MARKET_MAP

    # Dynamically select the list of markets to display based on the default commodity
    markets_for_commodity = ONION_MARKETS if commodity == 'onion' else RICE_MARKETS

    # Define the date range for the initial price trend graph (next 12 months)
    start_date = datetime.now()
    end_date = start_date + timedelta(days=365)
    dates = generate_date_range(start_date, end_date)

    graph_dates = []
    graph_prices = []

    # Select the correct features, market map, and model for the default commodity
    features = ONION_FEATURES if commodity == 'onion' else RICE_FEATURES
    market_map = ONION_MARKET_MAP if commodity == 'onion' else RICE_MARKET_MAP
    model = onion_model if commodity == 'onion' else rice_model

    # Get the feature name for the default market
    market_col = market_map.get(market)

    # If the default market is not valid for the default commodity,
    # try to fall back to the first available market for that commodity.
    if not market_col:
        if markets_for_commodity:
            market = markets_for_commodity[0] # Set new default market
            market_col = market_map.get(market)
        else:
            # This case should ideally not be reached if maps are correctly populated
            print(f"Error: No markets available for {commodity}.")
            return "Error: No markets available for this commodity."

    # Generate initial price trend data for the graph
    for date in dates:
        input_data = dict.fromkeys(features, 0)
        input_data['day'] = date.day
        input_data['month'] = date.month
        input_data['year'] = date.year
        input_data[market_col] = 1 # Set the specific market's dummy variable to 1

        # Create DataFrame and ensure column order
        input_df = pd.DataFrame([input_data])[features]
        pred_price = model.predict(input_df)[0]

        graph_dates.append(date.strftime('%Y-%m-%d'))
        graph_prices.append(round(pred_price, 2))

    # Render the index.html template with the initial data
    return render_template(
        'index.html',
        commodities=COMMODITIES,
        markets=markets_for_commodity, # Pass the filtered list of markets
        selected_commodity=commodity,
        selected_market=market,
        graph_dates=graph_dates,
        graph_prices=graph_prices
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        day = int(data['day'])
        month = int(data['month'])
        year = int(data['year'])
        market = data['market']
        commodity = data['commodity'].lower()

        if commodity not in COMMODITIES:
            return jsonify({'error': 'Invalid commodity selected'}), 400

        # Select model and features based on commodity
        features = RICE_FEATURES if commodity == 'rice' else ONION_FEATURES
        market_map = RICE_MARKET_MAP if commodity == 'rice' else ONION_MARKET_MAP
        model = rice_model if commodity == 'rice' else onion_model

        # Get the feature name for the selected market
        market_col = market_map.get(market)
        if not market_col:
            return jsonify({'error': f'Market "{market}" not available for {commodity}'}), 400

        # Initialize input data with all features set to 0
        input_data = dict.fromkeys(features, 0)
        input_data['day'] = day
        input_data['month'] = month
        input_data['year'] = year
        input_data[market_col] = 1 # Set the selected market's dummy variable to 1

        # Create DataFrame and ensure column order
        input_df = pd.DataFrame([input_data])[features]
        pred = model.predict(input_df)[0]

        return jsonify({'prediction': round(pred, 2)})

    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/get-markets', methods=['POST'])
def get_markets():
    try:
        data = request.json
        commodity = data.get('commodity', 'onion').lower()

        if commodity == 'rice':
            return jsonify({'markets': RICE_MARKETS})
        else: # Default to onion if not rice or specified as onion
            return jsonify({'markets': ONION_MARKETS})

    except Exception as e:
        print(f"Error in /get-markets: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/market-history', methods=['POST'])
def market_history():
    try:
        data = request.json
        market = data['market']
        commodity = data['commodity'].lower()

        if commodity not in COMMODITIES:
            return jsonify({'error': 'Invalid commodity selected'}), 400

        # Select model and features based on commodity
        features = RICE_FEATURES if commodity == 'rice' else ONION_FEATURES
        market_map = RICE_MARKET_MAP if commodity == 'rice' else ONION_MARKET_MAP
        model = rice_model if commodity == 'rice' else onion_model

        # Get the feature name for the selected market
        market_col = market_map.get(market)
        if not market_col:
            return jsonify({'error': f'Market "{market}" not available for {commodity}'}), 400

        # Define date range for market history (next 12 months)
        start_date = datetime.now()
        end_date = start_date + timedelta(days=365)
        dates = generate_date_range(start_date, end_date)

        graph_dates = []
        graph_prices = []

        for date in dates:
            input_data = dict.fromkeys(features, 0)
            input_data['day'] = date.day
            input_data['month'] = date.month
            input_data['year'] = date.year
            input_data[market_col] = 1

            # Create DataFrame and ensure column order
            input_df = pd.DataFrame([input_data])[features]
            pred_price = model.predict(input_df)[0]

            graph_dates.append(date.strftime('%Y-%m-%d'))
            graph_prices.append(round(pred_price, 2))

        # Handle case where no prices could be generated (e.g., market not found for any date)
        if not graph_dates:
            return jsonify({'error': f'No history data generated for market "{market}" and commodity "{commodity}".'}), 400

        return jsonify({'dates': graph_dates, 'prices': graph_prices})

    except Exception as e:
        print(f"Error in /market-history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/recommendation', methods=['POST'])
def recommendation():
    try:
        data = request.json
        market = data['market']
        commodity = data['commodity'].lower()

        if commodity not in COMMODITIES:
            return jsonify({'error': 'Invalid commodity selected'}), 400

        # Get monthly average prices using the helper function
        monthly_avg = get_monthly_avg_prices(commodity, market)

        # Check if the get_monthly_avg_prices function returned an error
        if isinstance(monthly_avg, dict) and 'error' in monthly_avg:
            return jsonify(monthly_avg), 400

        # Filter out months with zero average (where prediction might have failed for some reason)
        valid_monthly_avg = {m: p for m, p in monthly_avg.items() if p > 0}

        # If after filtering, there's no valid data, return an error
        if not valid_monthly_avg:
            return jsonify({'error': 'Not enough valid price data to generate recommendations for this market/commodity.'}), 400

        # Determine the month with the highest predicted average price (best for harvest)
        recommended_harvest_month = max(valid_monthly_avg, key=valid_monthly_avg.get)

        # Determine the recommended sow month. A common simplification is 4 months before harvest.
        # This calculation ensures the month wraps around correctly (e.g., if harvest is Jan (1), sow is Sep (9)).
        # (month - 1) makes it 0-indexed, - 4 for 4 months prior, % 12 wraps it, + 1 makes it 1-indexed again.
        recommended_sow_month = (recommended_harvest_month - 1 - 4) % 12 + 1

        # Convert monthly averages to strings for JSON keys (as months are integers)
        monthly_avg_json_ready = {str(k): round(v, 2) for k, v in monthly_avg.items()}

        return jsonify({
            'monthly_avg': monthly_avg_json_ready,
            'recommended_harvest_month': recommended_harvest_month,
            'recommended_sow_month': recommended_sow_month
        })

    except Exception as e:
        print(f"Error in /recommendation: {e}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    # When you run this script directly, Flask will start the development server.
    # debug=True allows for automatic reloading on code changes and provides a debugger.
    app.run(debug=True)