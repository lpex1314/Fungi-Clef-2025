import shutil

# Define the source and destination paths
source_path = '/root/.cache/kagglehub/competitions/fungi-clef-2025'
destination_path = '/root/autodl-tmp/KaggleHub/fungi-clef-2025/'  # Change this to your desired location


# Move the directory
shutil.move(source_path, destination_path)

print(f"Directory moved to: {destination_path}")
