import os
import sqlite3
import unittest
import logging
from unittest.mock import patch, MagicMock

from logs.log import log_close, CandleTelemetry, TelemetryLogger, _telemetry_logger
from src.config.logging import LOGGING


class TestLogging(unittest.TestCase):
    def setUp(self):
        # We can use a temporary in-memory or a temporary file DB for testing
        self.test_db_path = "logs/test_telemetry.db"
        self.telemetry_logger = TelemetryLogger(self.test_db_path)

    def tearDown(self):
        # Clean up database file
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except OSError:
                pass
            
        # Clean up any WAL files if they exist
        for suffix in ["-wal", "-shm"]:
            path = self.test_db_path + suffix
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def test_database_initialization(self):
        self.assertTrue(os.path.exists(self.test_db_path))
        with sqlite3.connect(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='telemetry';")
            table = cursor.fetchone()
            self.assertIsNotNone(table)
            self.assertEqual(table[0], "telemetry")

    def test_save_telemetry(self):
        telemetry = CandleTelemetry(
            timestamp=1629000000000,
            datetime_utc="2021-08-15 00:00:00",
            datetime_wib="2021-08-15 07:00:00",
            close_price=45000.50,
            rsi=55.4,
            rsi_smoothing=53.2,
            adx=25.0,
            adx_smoothing=24.5,
            cci=110.2,
            cci_smoothing=100.1,
            wt1=12.5,
            wt2=10.0,
            neighbours=[1, -1, 1],
            raw_prediction=1,
            signal_name="LONG",
            ram_usage_mb=75.5,
            atr=1.23,
            applied_atr_period=14,
            market_regime="RANGING",
            target_rr_ratio=2.0
        )
        
        self.telemetry_logger.save_telemetry(telemetry)
        
        with sqlite3.connect(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM telemetry WHERE timestamp=?;", (1629000000000,))
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], 1629000000000)
            self.assertEqual(row[1], "2021-08-15 00:00:00")
            self.assertEqual(row[2], "2021-08-15 07:00:00")
            self.assertEqual(row[3], 45000.50)
            self.assertEqual(row[12], "[1, -1, 1]")  # neighbours JSON string
            self.assertEqual(row[13], 1)
            self.assertEqual(row[14], "LONG")
            self.assertEqual(row[15], 75.5)
            self.assertEqual(row[16], 1.23)
            self.assertEqual(row[17], 14)
            self.assertEqual(row[18], "RANGING")
            self.assertEqual(row[19], 2.0)

    def test_log_close_execution(self):
        # Override the logger db path temporarily using patch
        with patch("logs.log._telemetry_logger", self.telemetry_logger):
            data_kline = {"close": "46000.75"}
            
            # This should run without crashing and print logs
            log_close(
                data_kline=data_kline,
                time_close=1629000060000,
                rsi_value=56.7,
                rsi_smoothing=55.0,
                adx_value=None,  # test Optional
                adx_smoothing=None,
                cci_value=120.5,
                cci_smoothing=110.0,
                wt1_value=15.0,
                wt2_value=12.2,
                array_tetangga=[1, 1, -1],
                raw_prediction=2,
                signal_name="LONG",
                atr_value=1.5,
                applied_atr_period=10,
                market_regime="TRENDING",
                target_rr_ratio=3.0
            )
            
            # Verify data was written to the test database
            with sqlite3.connect(self.test_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM telemetry WHERE timestamp=?;", (1629000060000,))
                row = cursor.fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row[3], 46000.75)  # close_price
                self.assertIsNone(row[6])  # adx is None/NULL at index 6
                self.assertEqual(row[14], "LONG")  # signal_name at index 14
                self.assertEqual(row[16], 1.5)  # atr_value at index 16
                self.assertEqual(row[17], 10)
                self.assertEqual(row[18], "TRENDING")
                self.assertEqual(row[19], 3.0)


if __name__ == "__main__":
    unittest.main()
