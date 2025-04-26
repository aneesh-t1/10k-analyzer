from sec_edgar_downloader import Downloader

def fetch_10k(ticker):
    print(f"Fetching 10-K for {ticker}...") #Debugging line

    email = "aneesh.tugg@gmail.com"

    dl = Downloader("downloads", email_address=email) # Initialize the downloader with a custom path and email


    dl.get("10-K", ticker)
    print(f"10-K for {ticker} downloaded successfully.") #Debugging line

if __name__ == "__main__":
    fetch_10k("AAPL")