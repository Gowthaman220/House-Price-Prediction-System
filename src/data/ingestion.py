"""Data ingestion and dataset generation module for House Price Prediction."""

import argparse
import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from src.utils.config import get_logger, get_project_root, load_config

logger = get_logger("data_ingestion")


def generate_benchmark_dataset(
    num_samples: int = 1500,
    random_state: int = 42,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """Generate a realistic benchmark House Price dataset modeled on the Ames Housing benchmark.
    
    Includes realistic distributions, non-linear relationships, realistic noise,
    and missing values to exercise data validation and preprocessing pipelines.
    
    Args:
        num_samples: Number of property records to generate.
        random_state: Random seed for reproducibility.
        output_path: Optional path to save the generated CSV.
        
    Returns:
        Pandas DataFrame containing raw house prices.
    """
    rng = np.random.RandomState(random_state)
    logger.info("Generating realistic benchmark dataset with %d records (seed=%d)...", num_samples, random_state)

    # 1. Neighborhoods with price premiums
    neighborhoods = [
        "NorthAmes", "CollegeCreek", "OldTown", "Edwards", "Somerset",
        "Gilbert", "Sawyer", "Northridge", "Crawford", "Mitchell"
    ]
    neighborhood_weights = [0.20, 0.15, 0.12, 0.10, 0.10, 0.08, 0.08, 0.07, 0.05, 0.05]
    neighborhood_premium = {
        "Northridge": 65000, "Somerset": 35000, "CollegeCreek": 20000,
        "Gilbert": 15000, "Crawford": 10000, "NorthAmes": 0,
        "Sawyer": -5000, "Mitchell": -10000, "Edwards": -18000, "OldTown": -25000
    }
    
    nhood_choices = rng.choice(neighborhoods, size=num_samples, p=neighborhood_weights)
    
    # 2. Building Type
    bldg_types = ["1Fam", "TwnhsE", "Twnhs", "Duplex", "2fmCon"]
    bldg_weights = [0.80, 0.08, 0.04, 0.05, 0.03]
    bldg_choices = rng.choice(bldg_types, size=num_samples, p=bldg_weights)
    
    # 3. House Style
    house_styles = ["1Story", "2Story", "1.5Fin", "SLvl"]
    style_weights = [0.50, 0.35, 0.10, 0.05]
    style_choices = rng.choice(house_styles, size=num_samples, p=style_weights)
    
    # 4. Central Air
    central_air_choices = rng.choice(["Y", "N"], size=num_samples, p=[0.93, 0.07])
    
    # 5. Year Built & Remod
    year_built = rng.randint(1920, 2024, size=num_samples)
    year_remod = np.clip(year_built + rng.randint(0, 30, size=num_samples), 1950, 2024)
    
    # 6. Overall Quality (1 to 10 scale, roughly normal around 6)
    overall_qual = np.clip(rng.normal(6.1, 1.4, size=num_samples).round().astype(int), 1, 10)
    
    # 7. Ground Living Area (sq ft, correlated with quality)
    gr_liv_area = (rng.normal(1500, 420, size=num_samples) + (overall_qual - 5) * 80).round()
    gr_liv_area = np.clip(gr_liv_area, 600, 4500)
    
    # 8. Total Basement Area (sq ft)
    total_bsmt_sf = (gr_liv_area * rng.uniform(0.6, 1.1, size=num_samples)).round()
    # Some houses don't have basements
    no_bsmt_mask = rng.rand(num_samples) < 0.04
    total_bsmt_sf[no_bsmt_mask] = 0.0
    
    # 9. Garage Cars (0 to 4)
    garage_cars = np.clip((overall_qual / 2.8 + rng.normal(0, 0.6, size=num_samples)).round().astype(int), 0, 4)
    
    # 10. Full Bathrooms (1 to 4)
    full_bath = np.clip((gr_liv_area / 900 + rng.normal(0, 0.4, size=num_samples)).round().astype(int), 1, 4)
    
    # 11. Lot Area (sq ft)
    lot_area = np.clip(rng.lognormal(mean=9.1, sigma=0.5, size=num_samples).round(), 1800, 45000)
    
    # 12. Fireplaces (0 to 3)
    fireplaces = np.clip(rng.choice([0, 1, 2, 3], size=num_samples, p=[0.48, 0.40, 0.10, 0.02]), 0, 3)
    
    # 13. Generate Realistic Sale Price (Regression Target)
    base_price = 55000.0
    liv_area_contrib = gr_liv_area * 68.0
    qual_contrib = (overall_qual ** 1.85) * 4200.0
    bsmt_contrib = total_bsmt_sf * 36.0
    garage_contrib = garage_cars * 12500.0
    bath_contrib = full_bath * 11000.0
    fire_contrib = fireplaces * 6500.0
    age_factor = (year_built - 1920) * 450.0 + (year_remod - 1950) * 250.0
    lot_contrib = np.sqrt(lot_area) * 85.0
    ac_contrib = np.where(central_air_choices == "Y", 14000.0, -10000.0)
    nhood_contrib = np.array([neighborhood_premium[nh] for nh in nhood_choices])
    
    # Noise and market variance
    random_noise = rng.normal(0, 14000, size=num_samples)
    
    sale_price = (
        base_price
        + liv_area_contrib
        + qual_contrib
        + bsmt_contrib
        + garage_contrib
        + bath_contrib
        + fire_contrib
        + age_factor
        + lot_contrib
        + ac_contrib
        + nhood_contrib
        + random_noise
    )
    sale_price = np.clip(sale_price, 50000.0, 750000.0).round(-2)
    
    # Assemble DataFrame
    df = pd.DataFrame({
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "TotalBsmtSF": total_bsmt_sf,
        "GarageCars": garage_cars,
        "FullBath": full_bath,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod,
        "LotArea": lot_area,
        "Fireplaces": fireplaces,
        "Neighborhood": nhood_choices,
        "BldgType": bldg_choices,
        "HouseStyle": style_choices,
        "CentralAir": central_air_choices,
        "SalePrice": sale_price,
    })
    
    # Inject realistic missing values (e.g. ~1.5% in TotalBsmtSF, ~1% in GarageCars)
    # to realistically test imputers and validation
    missing_bsmt_idx = rng.choice(num_samples, size=int(num_samples * 0.015), replace=False)
    df.loc[missing_bsmt_idx, "TotalBsmtSF"] = np.nan
    
    missing_garage_idx = rng.choice(num_samples, size=int(num_samples * 0.01), replace=False)
    df.loc[missing_garage_idx, "GarageCars"] = np.nan
    
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)
        logger.info("Saved benchmark raw dataset to: %s (%d rows)", out_file, len(df))
        
    return df


def ingest_data(
    source_path_or_url: Optional[str] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """Ingest raw house price data from a specified path, URL, or fallback to generated benchmark.
    
    Args:
        source_path_or_url: Path to CSV or HTTP(S) URL. If None, uses config or benchmark.
        output_path: Destination path for raw data.
        
    Returns:
        Loaded or generated DataFrame.
    """
    root = get_project_root()
    config = load_config()
    raw_path = output_path or config.get("data", {}).get("raw_path", "data/raw/house_prices.csv")
    target_dest = root / Path(raw_path)
    target_dest.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if a custom source is given
    source = source_path_or_url or os.getenv("DATA_SOURCE_URL")
    
    if source and (source.startswith("http://") or source.startswith("https://")):
        logger.info("Downloading dataset from remote URL: %s", source)
        df = pd.read_csv(source)
        df.to_csv(target_dest, index=False)
        logger.info("Successfully saved downloaded dataset to %s (%d records)", target_dest, len(df))
        return df
    
    # Check if already present on disk
    if target_dest.exists() and target_dest.stat().st_size > 0:
        logger.info("Loading existing raw dataset from: %s", target_dest)
        return pd.read_csv(target_dest)
        
    # Generate benchmark dataset
    logger.info("No raw data found at %s. Generating standard benchmark dataset...", target_dest)
    return generate_benchmark_dataset(num_samples=1500, output_path=str(target_dest))


def main():
    """CLI entrypoint for data ingestion stage."""
    parser = argparse.ArgumentParser(description="Ingest or generate House Price raw dataset.")
    parser.add_argument("--source", type=str, default=None, help="Optional remote URL or path to source CSV.")
    parser.add_argument("--output", type=str, default=None, help="Output destination path for raw CSV.")
    parser.add_argument("--force-regenerate", action="store_true", help="Force regenerate benchmark data.")
    args = parser.parse_args()
    
    if args.force_regenerate:
        config = load_config()
        out = args.output or config.get("data", {}).get("raw_path", "data/raw/house_prices.csv")
        generate_benchmark_dataset(num_samples=1500, output_path=out)
    else:
        ingest_data(source_path_or_url=args.source, output_path=args.output)


if __name__ == "__main__":
    main()
