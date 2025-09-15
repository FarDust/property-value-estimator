#!/usr/bin/env python3
"""
CSV to Database Migration Script - Raw Data

Migrates CSV data files into raw database tables preserving original data structure.
For house sales data, preserves all records including multiple sales of same property.
"""
import pandas as pd
import typer
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from property_value_estimator.infrastructure.db.entities.raw import (
    RawHouseSales,
    RawZipcodeDemographics,
    RawFutureHouseExample
)
from property_value_estimator.core.settings import settings

app = typer.Typer(help="Migrate CSV data to database")


def get_db_session():
    """Create database session."""
    engine = create_engine(settings.database.uri)
    # Don't create tables here - use Alembic for that
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


@app.command()
def migrate_house_sales(
    csv_path: Path = typer.Option(
        "mle-project-challenge-2/data/kc_house_data.csv",
        help="Path to house sales CSV file"
    )
):
    """Migrate house sales data from CSV to raw database table.
    
    PRESERVES ALL RECORDS INCLUDING MULTIPLE SALES OF SAME PROPERTY.
    This is important for time-series analysis and price trend modeling.
    """
    if not csv_path.exists():
        typer.echo(f"❌ CSV file not found: {csv_path}")
        raise typer.Exit(1)
    
    typer.echo(f"📁 Reading house sales data from {csv_path}")
    df = pd.read_csv(csv_path, dtype={'zipcode': str})
    
    # DO NOT remove duplicates - these are legitimate multiple sales
    typer.echo(f"📊 Found {len(df)} total sales records (including repeat sales)")
    
    session = get_db_session()
    try:
        # Clear existing raw data
        session.query(RawHouseSales).delete()
        session.commit()
        
        # Insert new data in batches, preserving all records
        batch_size = 1000
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            house_sales = []
            
            for _, row in batch.iterrows():
                house_sale = RawHouseSales(
                    id=row['id'],
                    date=pd.to_datetime(row['date']),
                    price=float(row['price']),
                    bedrooms=int(row['bedrooms']),
                    bathrooms=float(row['bathrooms']),
                    sqft_living=int(row['sqft_living']),
                    sqft_lot=int(row['sqft_lot']),
                    floors=float(row['floors']),
                    waterfront=int(row['waterfront']),
                    view=int(row['view']),
                    condition=int(row['condition']),
                    grade=int(row['grade']),
                    sqft_above=int(row['sqft_above']),
                    sqft_basement=int(row['sqft_basement']),
                    yr_built=int(row['yr_built']),
                    yr_renovated=int(row['yr_renovated']),
                    zipcode=str(row['zipcode']),
                    lat=float(row['lat']),
                    long=float(row['long']),
                    sqft_living15=int(row['sqft_living15']),
                    sqft_lot15=int(row['sqft_lot15'])
                )
                house_sales.append(house_sale)
            
            session.add_all(house_sales)
            session.commit()
            typer.echo(f"  ✅ Inserted batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")
        
        typer.echo(f"✅ Successfully migrated {len(df)} house sales records (preserving time-series data)")
        
    except Exception as e:
        session.rollback()
        typer.echo(f"❌ Error migrating house sales: {e}")
        raise typer.Exit(1)
    finally:
        session.close()


@app.command()
def migrate_demographics(
    csv_path: Path = typer.Option(
        "mle-project-challenge-2/data/zipcode_demographics.csv",
        help="Path to demographics CSV file"
    )
):
    """Migrate zipcode demographics data from CSV to raw database table."""
    if not csv_path.exists():
        typer.echo(f"❌ CSV file not found: {csv_path}")
        raise typer.Exit(1)
    
    typer.echo(f"📁 Reading demographics data from {csv_path}")
    df = pd.read_csv(csv_path, dtype={'zipcode': str})
    
    session = get_db_session()
    try:
        # Clear existing raw data
        session.query(RawZipcodeDemographics).delete()
        session.commit()
        
        # Insert new data
        demographics_list = []
        for _, row in df.iterrows():
            demographics = RawZipcodeDemographics(**row.to_dict())
            demographics_list.append(demographics)
        
        session.add_all(demographics_list)
        session.commit()
        typer.echo(f"✅ Successfully migrated {len(df)} demographics records")
        
    except Exception as e:
        session.rollback()
        typer.echo(f"❌ Error migrating demographics: {e}")
        raise typer.Exit(1)
    finally:
        session.close()


@app.command()
def migrate_future_examples(
    csv_path: Path = typer.Option(
        "mle-project-challenge-2/data/future_unseen_examples.csv",
        help="Path to future examples CSV file"
    )
):
    """Migrate future house examples from CSV to raw database table."""
    if not csv_path.exists():
        typer.echo(f"❌ CSV file not found: {csv_path}")
        raise typer.Exit(1)
    
    typer.echo(f"📁 Reading future examples data from {csv_path}")
    df = pd.read_csv(csv_path, dtype={'zipcode': str})
    
    session = get_db_session()
    try:
        # Clear existing raw data
        session.query(RawFutureHouseExample).delete()
        
        # Insert new data (ID will be auto-generated)
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            future_example = RawFutureHouseExample(**row_dict)
            session.add(future_example)
        
        session.commit()
        typer.echo(f"✅ Successfully migrated {len(df)} future example records")
        
    except Exception as e:
        session.rollback()
        typer.echo(f"❌ Error migrating future examples: {e}")
        raise typer.Exit(1)
    finally:
        session.close()


@app.command()
def migrate_all():
    """Migrate all CSV files to database."""
    typer.echo("🚀 Starting full migration...")
    
    try:
        # Call each migration with default paths
        migrate_house_sales(Path("mle-project-challenge-2/data/kc_house_data.csv"))
        migrate_demographics(Path("mle-project-challenge-2/data/zipcode_demographics.csv"))
        migrate_future_examples(Path("mle-project-challenge-2/data/future_unseen_examples.csv"))
        typer.echo("🎉 All data migrated successfully!")
    except typer.Exit:
        typer.echo("❌ Migration failed")
        raise


@app.command()
def status():
    """Show raw database status and record counts."""
    session = get_db_session()
    try:
        house_count = session.query(RawHouseSales).count()
        demo_count = session.query(RawZipcodeDemographics).count()
        future_count = session.query(RawFutureHouseExample).count()
        
        typer.echo("📊 Raw Database Status:")
        typer.echo(f"  Raw House Sales: {house_count:,} records")
        typer.echo(f"  Raw Demographics: {demo_count:,} records")
        typer.echo(f"  Raw Future Examples: {future_count:,} records")
        typer.echo(f"  Database: {settings.database.uri}")
        
    except Exception as e:
        typer.echo(f"❌ Error checking status: {e}")
        raise typer.Exit(1)
    finally:
        session.close()


if __name__ == "__main__":
    app()
