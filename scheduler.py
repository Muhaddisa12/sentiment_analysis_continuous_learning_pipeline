"""
Automated Retraining Scheduler

This scheduler monitors the database for new data and triggers
drift-aware retraining when new samples are detected.

The scheduler runs drift detection before retraining to ensure
retraining only occurs when necessary.
"""

import time
from datetime import datetime
from database.db import get_connection
from ml.trainer import train_pipeline
from utils.logger import get_logger

logger = get_logger()

def check_and_retrain():
    """
    Check for new data and trigger retraining if needed.
    Uses drift detection to determine if retraining is necessary.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Get last row count (note: table name may vary - adjust as needed)
        try:
            cur.execute("SELECT last_row_count FROM data_flag WHERE id=1")
            result = cur.fetchone()
            if result:
                last = result[0]
            else:
                # Initialize if not exists
                last = 0
                cur.execute("INSERT INTO data_flag (id, last_row_count) VALUES (1, 0)")
                conn.commit()
        except Exception as e:
            logger.warning(f"Could not read data_flag table: {e}. Assuming first run.")
            last = 0
        
        # Get current row count
        cur.execute("SELECT COUNT(*) FROM twitter_data")
        current = cur.fetchone()[0]
        
        if current > last:
            logger.info(f"New data detected: {current - last} new samples. Checking for drift...")
            
            # Train with drift detection (will only retrain if drift is detected)
            try:
                model_name, accuracy, drift_info = train_pipeline(check_drift=True)
                
                if drift_info and drift_info.get("drift_detected"):
                    logger.info("Drift detected, model retrained successfully")
                elif model_name:
                    logger.info("Model retrained (forced or first model)")
                else:
                    logger.info("No drift detected, skipping retraining")
                
                # Update last row count
                cur.execute("UPDATE data_flag SET last_row_count = %s WHERE id = 1", (current,))
                conn.commit()
            except Exception as e:
                logger.error(f"Training failed: {e}")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Scheduler error: {e}")


if __name__ == "__main__":
    logger.info("Starting automated retraining scheduler...")
    logger.info("Scheduler will check for new data and drift every hour at midnight")
    
    while True:
        now = datetime.now()
        # Check at midnight (00:00)
        if now.hour == 0 and now.minute == 0:
            check_and_retrain()
            # Sleep for 60 seconds to avoid multiple triggers
            time.sleep(60)
        time.sleep(30)  # Check every 30 seconds