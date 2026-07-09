from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urljoin
from urllib.error import HTTPError, URLError
import re
import zipfile
import io
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timezone, timedelta


APP_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = APP_ROOT / "data" / "kuroshio_axis.csv"
RAW_DIR = APP_ROOT / "data" / "raw_kuroshio"

PAGE_URL = "https://www1.kaiho.mlit.go.jp/KANKYO/KAIYO/qboc/kurosio-num.html"


def fetch_bytes(url, timeout=60):
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        }
    )
    with urlopen(req, timeout=timeout) as res:
        return res.read()


def find_qboc_zip_candidates():
    html = fetch_bytes(PAGE_URL).decode("utf-8", errors="ignore")

    links = re.findall(
        r'href=["\']([^"\']*qboc(\d{4})(\d{3})\.zip)["\']',
        html
    )

    if not links:
        raise RuntimeError("海上保安庁ページから qbocYYYYNNN.zip のリンクを見つけられませんでした。")

    records = []

    for href, year, number in links:
        full_url = urljoin(PAGE_URL, href)

        records.append({
            "url": full_url,
            "year": int(year),
            "number": int(number),
            "file": f"qboc{year}{number}.zip",
        })

    records = sorted(
        records,
        key=lambda x: (x["year"], x["number"]),
        reverse=True
    )

    return records


def download_latest_available_zip(candidates):
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    errors = []

    for cand in candidates:
        url = cand["url"]
        print("Try:", cand["file"], url)

        try:
            zip_bytes = fetch_bytes(url, timeout=60)

            if len(zip_bytes) < 1000:
                errors.append((cand["file"], "too small"))
                continue

            raw_zip_path = RAW_DIR / cand["file"]
            raw_zip_path.write_bytes(zip_bytes)

            print("Downloaded:", raw_zip_path)
            return cand, zip_bytes

        except HTTPError as e:
            print("Skip:", cand["file"], "HTTP", e.code)
            errors.append((cand["file"], f"HTTP {e.code}"))

        except URLError as e:
            print("Skip:", cand["file"], e)
            errors.append((cand["file"], str(e)))

        except Exception as e:
            print("Skip:", cand["file"], e)
            errors.append((cand["file"], str(e)))

    raise RuntimeError(f"利用可能なQBOC ZIPをダウンロードできませんでした: {errors[:10]}")


def parse_kml_coordinates(kml_bytes):
    text = kml_bytes.decode("utf-8", errors="ignore")
    coords = []

    try:
        root = ET.fromstring(text)

        for elem in root.iter():
            if elem.tag.endswith("coordinates") and elem.text:
                raw = elem.text.strip()
                parts = re.split(r"\s+", raw)

                for p in parts:
                    vals = p.split(",")

                    if len(vals) >= 2:
                        try:
                            lon = float(vals[0])
                            lat = float(vals[1])

                            if 120 <= lon <= 150 and 20 <= lat <= 40:
                                coords.append((lat, lon))

                        except ValueError:
                            pass

    except ET.ParseError:
        pattern = re.findall(
            r"<coordinates>(.*?)</coordinates>",
            text,
            flags=re.S
        )

        for block in pattern:
            parts = re.split(r"\s+", block.strip())

            for p in parts:
                vals = p.split(",")

                if len(vals) >= 2:
                    try:
                        lon = float(vals[0])
                        lat = float(vals[1])

                        if 120 <= lon <= 150 and 20 <= lat <= 40:
                            coords.append((lat, lon))

                    except ValueError:
                        pass

    return coords


def main():
    candidates = find_qboc_zip_candidates()

    print("Candidate count:", len(candidates))
    print("Top candidates:")
    for c in candidates[:10]:
        print(c)

    latest, zip_bytes = download_latest_available_zip(candidates)

    print("Using QBOC zip:")
    print(latest)

    all_coords = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = z.namelist()

        print("ZIP files:")
        for n in names[:30]:
            print("-", n)

        kml_names = [
            n for n in names
            if n.lower().endswith(".kml")
            and (
                n.replace("\\", "/").lower().startswith("k/")
                or "/k/" in n.replace("\\", "/").lower()
                or "kuro" in n.lower()
                or "kuroshio" in n.lower()
            )
        ]

        if not kml_names:
            kml_names = [n for n in names if n.lower().endswith(".kml")]

        print("KML files used:")
        for n in kml_names:
            print("-", n)

        for kml_name in kml_names:
            kml_bytes = z.read(kml_name)
            coords = parse_kml_coordinates(kml_bytes)

            for lat, lon in coords:
                all_coords.append({
                    "axis_lat": lat,
                    "axis_lon": lon,
                    "source_file": latest["file"],
                    "source_url": latest["url"],
                    "kml_name": kml_name,
                    "downloaded_at_jst": datetime.now(
                        timezone(timedelta(hours=9))
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                })

    if not all_coords:
        raise RuntimeError("KMLから黒潮流線の座標を抽出できませんでした。")

    axis = pd.DataFrame(all_coords)

    axis = axis.drop_duplicates(
        subset=["axis_lat", "axis_lon"]
    ).copy()

    axis = axis.sort_values(
        ["axis_lon", "axis_lat"]
    ).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    axis.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("Saved:", OUT_PATH)
    print("n_axis_points:", len(axis))
    print(axis.head())
    print(axis.tail())


if __name__ == "__main__":
    main()
