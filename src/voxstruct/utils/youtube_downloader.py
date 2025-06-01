import os
import subprocess
import tempfile
import shutil
import logging
import re # For sanitizing filenames
from typing import Optional, Tuple, Union

class YoutubeDownloader:
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitizes a string to be a valid filename."""
        # Remove invalid characters
        filename = re.sub(r'[\\/*?:"<>|]', "", filename)
        # Replace spaces with underscores
        filename = filename.replace(" ", "_")
        # Truncate if too long (optional, but good practice)
        return filename[:200] # Max filename length can be an issue

    def download_audio_from_youtube(self, url: str) -> Optional[Tuple[str, str, str]]:
        """
        Downloads the best audio from a YouTube URL using yt-dlp,
        saves it as an mp3 file named after the video title.

        Args:
            url: The YouTube video URL.

        Returns:
            A tuple (audio_file_path, temp_dir_path, base_filename) if successful, None otherwise.
            The base_filename is the sanitized video title without extension.
            The caller is responsible for cleaning up the temp_dir_path.
        """
        temp_dir = tempfile.mkdtemp(prefix="voxstruct_youtube_")
        
        # Step 1: Get the video title using yt-dlp
        try:
            title_command = ["yt-dlp", "--get-title", "--no-playlist", url]
            title_process = subprocess.run(title_command, capture_output=True, text=True, check=True)
            video_title = title_process.stdout.strip()
            sanitized_base_filename = self._sanitize_filename(video_title)
            # print(f"Original Title: {video_title}, Sanitized: {sanitized_base_filename}") # Debug
        except subprocess.CalledProcessError as e:
            logging.error(f"yt-dlp failed to get video title: {e.stderr}")
            shutil.rmtree(temp_dir)
            return None
        except Exception as e:
            logging.error(f"Error getting video title: {e}")
            shutil.rmtree(temp_dir)
            return None

        if not sanitized_base_filename: # Handle empty title case
            sanitized_base_filename = "youtube_audio" 
            
        output_template = os.path.join(temp_dir, f"{sanitized_base_filename}.%(ext)s")
        expected_audio_file = os.path.join(temp_dir, f"{sanitized_base_filename}.mp3")

        try:
            download_command = [
                "yt-dlp",
                "--no-playlist",
                "-f", "bestaudio/best",
                "-x",
                "--audio-format", "mp3",
                "--output", output_template,
                url
            ]
            # print(f"Executing download: {' '.join(download_command)}") # Debug
            process = subprocess.run(download_command, capture_output=True, text=True, check=False)
            
            if process.returncode != 0:
                logging.error(f"yt-dlp download failed with return code {process.returncode}")
                logging.error(f"yt-dlp stdout: {process.stdout}")
                logging.error(f"yt-dlp stderr: {process.stderr}")
                shutil.rmtree(temp_dir)
                return None

            if not os.path.exists(expected_audio_file):
                # Check if file exists with a slightly different sanitized name or if yt-dlp used a fallback
                found_files = [f for f in os.listdir(temp_dir) if f.startswith(sanitized_base_filename) and f.endswith(".mp3")]
                if found_files:
                    actual_filename = found_files[0]
                    expected_audio_file = os.path.join(temp_dir, actual_filename)
                    # Update sanitized_base_filename to reflect the actual file created (without extension)
                    sanitized_base_filename = os.path.splitext(actual_filename)[0]
                    # print(f"Adjusted expected file to: {expected_audio_file}, new base: {sanitized_base_filename}") #Debug
                else:
                    logging.error(f"yt-dlp ran, but expected audio file '{expected_audio_file}' not found in {temp_dir}.")
                    files_in_temp = os.listdir(temp_dir)
                    logging.error(f"Files found: {files_in_temp}" if files_in_temp else "No files found.")
                    shutil.rmtree(temp_dir)
                    return None

            print(f"Successfully downloaded and extracted audio to: {expected_audio_file}")
            return expected_audio_file, temp_dir, sanitized_base_filename

        except Exception as e:
            logging.error(f"An unexpected error occurred during YouTube download: {e}")
            if hasattr(e, 'stderr'): logging.error(f"Stderr: {e.stderr}")
            if hasattr(e, 'stdout'): logging.error(f"Stdout: {e.stdout}")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return None

if __name__ == '__main__':
    # Example usage:
    logging.basicConfig(level=logging.INFO)
    downloader = YoutubeDownloader()
    # test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Rick Astley
    test_url = "https://www.youtube.com/watch?v=O-T_6KOXML4" # Thu Vu
    # test_url = "https://www.youtube.com/watch?v=long_video_id_example_12345" # Test non-existent
    # test_url = "https://www.youtube.com/watch?v=video_with_lots_of%%%%special%%%%chars_in_title"

    result = downloader.download_audio_from_youtube(test_url)
    if result:
        audio_path, temp_directory, video_title_base = result
        print(f"Audio downloaded to: {audio_path}")
        print(f"Temporary directory: {temp_directory}")
        print(f"Video title base for filename: {video_title_base}")
        
        # Example of how main.py would use it:
        # output_basename = video_title_base 
        # raw_output_file = os.path.join("output", f"raw_transcript_{output_basename}.txt")
        # print(f"Example raw transcript path: {raw_output_file}")

        # shutil.rmtree(temp_directory) # Clean up
        # print(f"Cleaned up temp directory: {temp_directory}")
    else:
        print("Failed to download audio.") 