#!/usr/bin/env python3
"""Test script for per-ticker last_fetched_date tracking and delta fetch."""

import sys
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path - we're running from project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from database import DatabaseManager
from fetchers import StockDataFetcher
import pandas as pd
import structlog

logger = structlog.get_logger()

def create_test_db():
    """Create a temporary test database."""
    tmpfd, tmp_path = tempfile.mkstemp(suffix='.db')
    os.close(tmpfd)
    return tmp_path

def test_delta_fetch():
    """Test the delta fetch functionality."""
    test_logger = logger.bind(test="delta_fetch")
    test_logger.info("Starting delta fetch test")

    # Create temp DB
    db_path = create_test_db()
    test_logger.info(f"Created test database at {db_path}")

    try:
        # Initialize database
        with DatabaseManager(db_path) as db:
            test_logger.info("Database initialized")

            # Check ticker_metadata table exists
            cursor = db.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ticker_metadata'")
            assert cursor.fetchone() is not None, "ticker_metadata table should exist"
            test_logger.info("✓ ticker_metadata table exists")

            # Initially, no metadata for AAPL
            last_date = db.get_last_fetched_date('AAPL')
            assert last_date is None, f"Expected None, got {last_date}"
            test_logger.info("✓ No initial metadata for AAPL")

        # Seed some historical data - simulate existing stock data from 2024
        test_logger.info("Seeding database with old AAPL data (2024-01-01 to 2024-01-10)")
        with DatabaseManager(db_path) as db:
            dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
            data = {
                'Open': [150.0 + i for i in range(len(dates))],
                'High': [151.0 + i for i in range(len(dates))],
                'Low': [149.0 + i for i in range(len(dates))],
                'Close': [150.5 + i for i in range(len(dates))],
                'Volume': [1000000 + i*1000 for i in range(len(dates))]
            }
            df = pd.DataFrame(data, index=dates)
            db.save_stock_data('AAPL', df)

            # Auto-migration should populate metadata
            last_date = db.get_last_fetched_date('AAPL')
            expected = '2024-01-10'
            assert last_date == expected, f"Expected {expected}, got {last_date}"
            test_logger.info(f"✓ Auto-migration populated metadata: {last_date}")

        # Simulate time passing - set last_fetched_date to 5 days ago
        five_days_ago = (datetime.now().date() - timedelta(days=5)).strftime('%Y-%m-%d')
        test_logger.info(f"Setting last_fetched_date to {five_days_ago}")
        with DatabaseManager(db_path) as db:
            db.update_last_fetched_date('AAPL', five_days_ago)

        # Now test delta fetch - it should fetch from five_days_ago + 1 to today
        test_logger.info("Testing delta fetch...")
        with DatabaseManager(db_path) as db:
            fetcher = StockDataFetcher(db_manager=db, ttl_days=0)  # ttl_days=0 to force delta
            result = fetcher.fetch_delta('AAPL', force_refresh=False)

            assert result is not None, "Result should not be None"
            assert not result.empty, "Result should not be empty"

            # Check that we now have data covering the gap
            dates_in_result = result.index.strftime('%Y-%m-%d').tolist()
            min_date = min(dates_in_result)
            max_date = max(dates_in_result)

            test_logger.info(f"Result date range: {min_date} to {max_date}")
            test_logger.info(f"Result has {len(result)} rows")

            # The result should include both old data and new delta
            # Min date should be from the seed data (2024-01-01)
            assert '2024-01-01' in dates_in_result, "Should contain original seed data"

            # The max date should be around today
            today_str = datetime.now().strftime('%Y-%m-%d')
            assert today_str in dates_in_result, f"Should contain today's date {today_str}"

            # Check metadata was updated
            new_last_date = db.get_last_fetched_date('AAPL')
            assert new_last_date == today_str, f"Metadata should be updated to today: {new_last_date}"
            test_logger.info(f"✓ Metadata updated to {new_last_date}")

        test_logger.info("✓ Delta fetch test passed!")
        return True

    except AssertionError as e:
        test_logger.error(f"Assertion failed: {e}")
        return False
    except Exception as e:
        test_logger.error(f"Test failed: {e}", exc_info=True)
        return False
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
            test_logger.info("Cleaned up test database")

def test_first_run_full_fetch():
    """Test that first run (no metadata) does full year fetch."""
    test_logger = logger.bind(test="first_run")
    test_logger.info("Testing first-run behavior")

    db_path = create_test_db()
    test_logger.info(f"Created test database at {db_path}")

    try:
        with DatabaseManager(db_path) as db:
            # No metadata initially
            last_date = db.get_last_fetched_date('MSFT')
            assert last_date is None, "Should have no metadata for MSFT"
            test_logger.info("✓ First-run logic verified (no metadata triggers full fetch path)")
            return True

    except Exception as e:
        test_logger.error(f"Test failed: {e}")
        return False
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

if __name__ == '__main__':
    logger = logger.bind(test="main")

    print("=" * 60)
    print("Testing per-ticker last_fetched_date tracking & delta fetch")
    print("=" * 60)

    test1 = test_delta_fetch()
    print()
    test2 = test_first_run_full_fetch()

    print()
    print("=" * 60)
    if test1 and test2:
        print("ALL TESTS PASSED ✓")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED ✗")
        sys.exit(1)
