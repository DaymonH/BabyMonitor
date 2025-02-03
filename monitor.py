# Baby Sleep Monitor Script

from sqlalchemy import create_engine, Column, Integer, DateTime, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import cv2
import numpy as np
from picamera2 import Picamera2
import time
from datetime import datetime
import os

# Define the SQLAlchemy model
Base = declarative_base()

class SleepData(Base):
    __tablename__ = 'sleep_data'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    movement_level = Column(Float, nullable=False)
    frame_path = Column(String(255))

class BabySleepMonitor:
    def __init__(self, db_url, capture_interval=5):
        # Initialize camera
        self.camera = Picamera2()
        self.camera.configure(self.camera.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
        self.camera.start()
        time.sleep(2)  # Give camera time to warm up

        # Setup SQLAlchemy
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # Motion detection parameters
        self.prev_frame = None
        self.movement_threshold = 25
        self.capture_interval = capture_interval

        # Create frames directory if it doesn't exist
        if not os.path.exists('frames'):
            os.makedirs('frames')

    def detect_movement(self, frame):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return 0

        # Calculate difference between frames
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # Calculate movement level (percentage of pixels that changed)
        movement_level = (np.sum(thresh) / 255.0) / (thresh.shape[0] * thresh.shape[1]) * 100

        self.prev_frame = gray
        return movement_level

    def save_data(self, movement_level, frame):
        try:
            # Print to console if significant movement is detected
            if movement_level > self.movement_threshold:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                print(f"Movement detected at {timestamp} with level: {movement_level}")

                # Save frame if significant movement detected
                frame_path = f'frames/frame_{timestamp}.jpg'
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                print(f"Frame saved at {frame_path}")

        except Exception as e:
            print(f"Error saving data: {e}")

    def run(self):
        print("Starting sleep monitoring...")
        try:
            while True:
                # Capture frame
                frame = self.camera.capture_array()

                # Detect movement
                movement_level = self.detect_movement(frame)

                # Save data
                self.save_data(movement_level, frame)

                # Wait for next capture
                time.sleep(self.capture_interval)

        except KeyboardInterrupt:
            print("\nStopping monitoring...")
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            self.camera.stop()
            print("Monitoring stopped")

if __name__ == "__main__":
    # Database URL (not used anymore, but kept for future use)
    db_url = 'mysql+pymysql://RasberryPi:RasberryPi@/192.168.1.112/MySQL80'

    monitor = BabySleepMonitor(db_url=db_url, capture_interval=0.5)  # Capture every 0.5 seconds
    monitor.run()
