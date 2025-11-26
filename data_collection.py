import os
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time

# --- Configuration ---
# This dictionary should match the one in your app.py
COMPANIES = {
    #"Tata Motors": "TATAMOTORS.NS",
    #"Reliance Industries": "RELIANCE.NS",
    #"Infosys": "INFY.NS",
    #"Apple": "AAPL",
    #"Google": "GOOGL",
    #"Microsoft": "MSFT",
    #"Apple Inc.": "AAPL",
    #"Amazon.com, Inc.": "AMZN",
    #"Meta Platforms, Inc.": "META",
    #"Tesla, Inc.": "TSLA",
    #"NVIDIA Corporation": "NVDA",
    #"Berkshire Hathaway Inc.": "BRK.B",
    #"JPMorgan Chase & Co.": "JPM",
    #"Johnson & Johnson": "JNJ",
    #"Visa Inc.": "V",
    #"UnitedHealth Group Incorporated": "UNH",
    #"Procter & Gamble Company": "PG",
    #"Mastercard Incorporated": "MA",
    #"Home Depot, Inc.": "HD",
    #"Exxon Mobil Corporation": "XOM",
    #"Bank of America Corporation": "BAC",
    #"Walt Disney Company": "DIS",
    #"Pfizer Inc.": "PFE",
    #"AbbVie Inc.": "ABBV",
    #"Merck & Co., Inc.": "MRK",
    #"PepsiCo, Inc.": "PEP",
    #"Chevron Corporation": "CVX",
    #"Intel Corporation": "INTC",
    #"Cisco Systems, Inc.": "CSCO",
    #"Coca-Cola Company": "KO",
    #"Adobe Inc.": "ADBE",
    #"Salesforce, Inc.": "CRM",
    #"Netflix, Inc.": "NFLX",
    #"Oracle Corporation": "ORCL",
    #"Walmart Inc.": "WMT",
    #"McDonald's Corporation": "MCD",
    #"Nike, Inc.": "NKE",
    #"Texas Instruments Incorporated": "TXN",
    #"Qualcomm Incorporated": "QCOM",
    #"Costco Wholesale Corporation": "COST",
    #"Thermo Fisher Scientific Inc.": "TMO",
    #"Medtronic plc": "MDT",
    #"Danaher Corporation": "DHR",
    #"Abbott Laboratories": "ABT",
    #"Eli Lilly and Company": "LLY",
    #"Amgen Inc.": "AMGN",
    #"Gilead Sciences, Inc.": "GILD",
    #"Intuitive Surgical, Inc.": "ISRG",#
    #"Zoetis Inc.": "ZTS",
    #"Cigna Group": "CI",
    #"Regeneron Pharmaceuticals, Inc.": "REGN",
    #"Vertex Pharmaceuticals Incorporated": "VRTX",
    #"Broadcom Inc.": "AVGO",
    #"Analog Devices, Inc.": "ADI",
    #"Micron Technology, Inc.": "MU",
    #"Fidelity National Information Services": "FIS",
    #"FedEx Corporation": "FDX",
    #"Deere & Company": "DE",
    # "General Motors Company": "GM",
    #"Ford Motor Company": "F",
    #"General Electric Company": "GE",
    #"Boeing Company": "BA",
    #"Caterpillar Inc.": "CAT",
    #"3M Company": "MMM",
    #"International Business Machines Corporation": "IBM",
    #"Altria Group, Inc.": "MO",
    #"Philip Morris International Inc.": "PM",
    #"Colgate-Palmolive Company": "CL",
    #"Kimberly-Clark Corporation": "KMB",
    #"Estée Lauder Companies Inc.": "EL",
    #"Ross Stores, Inc.": "ROST",
    #"TJX Companies, Inc.": "TJX",
    #"Dollar General Corporation": "DG",
    #"Dollar Tree, Inc.": "DLTR",
    #"eBay Inc.": "EBAY",
    #"Etsy, Inc.": "ETSY",#
    #"PayPal Holdings, Inc.": "PYPL",
    #"Block, Inc.": "SQ",
    #"Shopify Inc.": "SHOP",
    #"Twilio Inc.": "TWLO",
    #"Snowflake Inc.": "SNOW",
    #"Palantir Technologies Inc.": "PLTR",
    #"Zscaler, Inc.": "ZS",
    #"CrowdStrike Holdings, Inc.": "CRWD",
    #"Datadog, Inc.": "DDOG",
    #"Okta, Inc.": "OKTA",
    #"DocuSign, Inc.": "DOCU",
    #"Roku, Inc.": "ROKU",
    #"Uber Technologies, Inc.": "UBER",
    #"Lyft, Inc.": "LYFT",
    #"Rivian Automotive, Inc.": "RIVN",
    #"Lucid Group, Inc.": "LCID",
    #"SoFi Technologies, Inc.": "SOFI",
    #"Robinhood Markets, Inc.": "HOOD",
    #"American Express Company": "AXP",
    #"S&P Global Inc.": "SPGI",
    #"Union Pacific Corporation": "UNP",
    #"Honeywell International Inc.": "HON",
    #"Lowe's Companies, Inc.": "LOW",
    #"Starbucks Corporation": "SBUX",
    #"NextEra Energy, Inc.": "NEE",
    "United Parcel Service, Inc.": "UPS",
    "Southern Company": "SO",
    "Dominion Energy, Inc.": "D",
    "Exelon Corporation": "EXC",
    "Duke Energy Corporation": "DUK"
}

GUARDIAN_API_KEY = '' # Your Guardian API Key
DATA_DIR = '../data'
START_DATE = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')

# --- Helper Functions ---
def fetch_stock_data(ticker, start, end):
    """Fetches historical stock data from Yahoo Finance."""
    print(f"Fetching stock data for {ticker} from {start} to {end}...")
    stock_data = yf.download(ticker, start=start, end=end,timeout=120)
    stock_data.reset_index(inplace=True)
    print(f"Stock data for {ticker} fetched successfully.")
    return stock_data

def fetch_news_data(api_key, company_name, start_date, end_date):
    """Fetches news headlines from The Guardian API with pagination."""
    print(f"Fetching news for '{company_name}'...")
    all_articles = []
    url = 'https://content.guardianapis.com/search'
    params = { 'q': company_name, 'from-date': start_date, 'to-date': end_date, 'api-key': api_key, 'page-size': 50, 'page': 1 }
    while True:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json().get('response', {})
            articles = data.get('results', [])
            if not articles: break
            for article in articles:
                all_articles.append({
                    'Date': pd.to_datetime(article['webPublicationDate']).strftime('%Y-%m-%d'),
                    'News': article['webTitle']
                })
            if params['page'] >= data.get('pages', 0): break
            params['page'] += 1
            time.sleep(1)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred for {company_name}: {e}")
            break
    print(f"Fetched {len(all_articles)} news articles for {company_name}.")
    return pd.DataFrame(all_articles)

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Loop through each company to fetch and save its data
    for company_name, ticker in COMPANIES.items():
        print(f"\n--- Processing {company_name} ({ticker}) ---")

        # Fetch and save stock data
        stock_df = fetch_stock_data(ticker, START_DATE, END_DATE)
        if not stock_df.empty:
            stock_path = os.path.join(DATA_DIR, f'stock_data_{ticker}.csv')
            stock_df.to_csv(stock_path, index=False)
            print(f"Stock data saved to {stock_path}")

        # Fetch and save news data
        news_df = fetch_news_data(GUARDIAN_API_KEY, company_name, START_DATE, END_DATE)
        if not news_df.empty:
            news_path = os.path.join(DATA_DIR, f'news_headlines_{ticker}.csv')
            news_df.to_csv(news_path, index=False)
            print(f"News data saved to {news_path}")
