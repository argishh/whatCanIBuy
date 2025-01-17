import subprocess

URL_LIST = ["https://www.youtube.com/watch?v=7TKZRc2Cn00", ]

for link in URL_LIST:
    subprocess.run(["yt-dlp", "--downloader", "ffmpeg", "--downloader-args", "ffmpeg:-t 60", link])



######### How to use? #########

# 1) download yt-dlp.exe from https://github.com/yt-dlp/yt-dlp?tab=readme-ov-file#recommended
# 2) Place yt-dlp in C:\
# 3) Ensure C:\ is added to path

# (Optional) Shell command to download first 60 seconds of a video from youtube -
# yt-dlp --downloader ffmpeg --downloader-args "ffmpeg:-t 60"  "https://www.youtube.com/watch?v=7TKZRc2Cn00"