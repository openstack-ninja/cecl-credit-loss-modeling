import os
import zipfile
from playwright.sync_api import sync_playwright

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(accept_downloads=True, viewport={'width': 1920, 'height': 1080})
    page = context.new_page()
    page.goto("https://datadynamics.fanniemae.com/")

    page.wait_for_selector('input', timeout=10000)

    email = os.environ.get('FANNIE_MAE_EMAIL')
    password = os.environ.get('FANNIE_MAE_PASSWORD')

    if not email or not password:
        raise ValueError("Please set FANNIE_MAE_EMAIL and FANNIE_MAE_PASSWORD environment variables.")

    inputs = page.query_selector_all('input')
    inputs[0].fill(email)
    inputs[1].fill(password)

    page.click('button:has-text("Login")')

    page.wait_for_timeout(10000) # Wait for login

    page.click('text=Historical Loan Credit Performance Data')

    page.wait_for_timeout(5000) # Wait for loading

    page.click('text=Download Quarterly Data')
    page.wait_for_timeout(5000)

    print("Finding the rows for all years...")
    rows = page.query_selector_all('tr')

    for row in rows:
        text = row.inner_text().strip()
        year = text.split()[0] if text else ""

        # Check if year is a digit and represents a valid year (e.g., starts with 20)
        if year.isdigit() and len(year) == 4 and year.startswith('20'):
            print(f"Downloading for year: {year}")

            # Find the first 4 download links (Q1 to Q4)
            # The structure is: <td><dd-button><button>Acquisition and Performance</button></dd-button></td>
            buttons = row.query_selector_all('button.button-root.link')

            for i in range(4):
                if i < len(buttons):
                    print(f"  Clicking Q{i+1}...")
                    try:
                        with page.expect_download(timeout=120000) as download_info:
                            buttons[i].click()
                        download = download_info.value

                        # Save the downloaded file into data/raw/YYYYQ{i+1}/
                        out_dir = f"data/raw/{year}Q{i+1}"
                        os.makedirs(out_dir, exist_ok=True)

                        out_path = os.path.join(out_dir, download.suggested_filename)
                        download.save_as(out_path)
                        print(f"  Saved to: {out_path}")

                        # Extract the zip file
                        if out_path.endswith('.zip'):
                            print(f"  Extracting {out_path}...")
                            with zipfile.ZipFile(out_path, 'r') as zip_ref:
                                zip_ref.extractall(out_dir)
                            print(f"  Extracted to {out_dir}")
                            # Delete the zip file after extraction
                            os.remove(out_path)
                            print(f"  Deleted zip file {out_path}")
                    except Exception as e:
                        print(f"  Failed Q{i+1}: {e}")

    browser.close()

if __name__ == "__main__":
    with sync_playwright() as playwright:
        run(playwright)
