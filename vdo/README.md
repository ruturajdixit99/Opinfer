# Test Videos for Opinfer Performance Testing

This folder contains test videos used for performance benchmarking and validation of Opinfer's motion-gated inference.

## Video Files

- **slowtraffic.mp4** (~13.7 MB) - Static/slow traffic camera scenario
- **fastbikedrive.mp4** (~66.0 MB) - Fast motion scenario (bike ride)
- **fastcarnightdrivedashcam.mp4** (~43.4 MB) - Night drive with low contrast

**Total size**: ~123 MB

## Usage

These videos are used by the test script `test_video_performance.py`:

```bash
python test_video_performance.py
```

The script will automatically process all videos in this folder and generate comprehensive performance graphs in the `graphs/` directory.

## Note for Repository

Due to file size constraints (~123 MB total), these videos may not be included directly in the repository. Users can:

1. **Download separately**: Videos can be provided via alternative hosting (Google Drive, Dropbox, etc.)
2. **Use their own videos**: Place any MP4/AVI/MOV files in this folder to test
3. **Clone from releases**: Videos may be included in GitHub releases as downloadable assets

## Video Characteristics

- **Format**: MP4 (H.264)
- **Frame Rate**: 30 FPS (typical)
- **Resolution**: Varies by video
- **Duration**: 10-30 seconds per video (500 frames processed in tests)

## Custom Videos

You can add your own test videos to this folder. The test script will automatically detect and process them. For best results, use videos that represent your target use case:
- Static scenes (traffic cameras, security feeds)
- Moderate motion (walking, normal traffic)
- Fast motion (sports, drones, vehicles)


