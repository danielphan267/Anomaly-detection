from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import os
import subprocess

class VideoHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith('.mp4'):
            self.process_video(event.src_path)

    def process_video(self, video_path):
        while True:
            try:
                size1 = os.path.getsize(video_path)
                time.sleep(0.5)  # Wait for 1 second
                size2 = os.path.getsize(video_path)

                if size1 == size2:  # If the file size hasn't changed, it's likely fully written
                    break
            except OSError:
                pass
        # Add your video processing code
        print(f"Processing video: {video_path}")
        subprocess.run(['python', 'predict.py', video_path])


def watch_folder(folder_path):
    event_handler = VideoHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    print('run detection')
    # Call the function to start watching the folder
    watch_folder("F:/Anomaly Detection/videos")
