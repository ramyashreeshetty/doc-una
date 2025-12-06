import os, time, yaml, pathlib, requests, json
from bs4 import BeautifulSoup
from urllib.parse import urljoin


RAW_DIR = pathlib.Path('data/raw'); RAW_DIR.mkdir(parents=True, exist_ok=True)


def extract_main(html, base):
    soup = BeautifulSoup(html, 'html.parser')
    # Many mkdocs themes use <article> for main content
    art = soup.find('article') or soup.find('main') or soup.body
    # Strip nav/aside/footers
    for tag in art.select('nav, aside, footer'): tag.decompose()
    
    # Keep headings and paragraphs only
    text = []
    for el in art.find_all(['h1','h2','h3','p','li','pre','code']):
        text.append(el.get_text('', strip=True))
    return ''.join([t for t in text if t])


if __name__ == '__main__':
    cfg = yaml.safe_load(open('ingest/config.yaml'))
    base = cfg['base_url']
    
    failed_urls = []
    manifest_path = RAW_DIR / '_manifest.json'
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    except Exception:
        manifest = {}
    for path in cfg['urls']:
        url = urljoin(base, path)
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            txt = extract_main(r.text, base)
            slug = path.strip('/').replace('/','__') or 'index'
            (RAW_DIR / f'{slug}.md').write_text(txt, encoding='utf-8')
            print('saved', slug)

            # capture title and final URL for nicer UI labels
            try:
                soup = BeautifulSoup(r.text, 'html.parser')
                title_el = soup.find('h1') or soup.title or soup.find('header')
                title = title_el.get_text(strip=True) if title_el else slug
                final_url = r.url
                manifest[f'{slug}.md'] = {"title": title, "url": final_url}
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception:
                pass
        except requests.exceptions.RequestException as e:
            print(f'ERROR: Failed to fetch {url} - {e}')
            failed_urls.append(url)
        except Exception as e:
            print(f'ERROR: Unexpected error for {url} - {e}')
            failed_urls.append(url)
        
        time.sleep(0.3)
    
    if failed_urls:
        print(f'\nFailed to scrape {len(failed_urls)} URLs:')
        for url in failed_urls:
            print(f'  - {url}')
    else:
        print('\nAll URLs scraped successfully!')