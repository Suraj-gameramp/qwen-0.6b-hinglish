Volume structure (confirmed):

  hiacc-data:/
  ├── Corpus.zip                          (original 531 MB zip)
  └── hiacc/Corpus/
      ├── readme.txt
      ├── adult/
      │   ├── audio/         ← WAV files
      │   ├── annotations/   ← code_switched_labels.json
      │   ├── metadata/      ← sentence_stats.csv, speaker_info.csv
      │   └── transcription/ ← train/val/test_output.txt
      └── children/
          ├── audio/
          ├── annotations/
          ├── metadata/
          └── transcript/    ← train/val/test_output.txt
  
  Three ways to view it:
  
  1. Modal dashboard (browser, easiest). Open https://modal.com/storage/suraj-77958/main/volumes/hiacc-data — you can browse the tree, click into folders, preview text files, and download any single file with one click.
  
  2. CLI to your laptop. Run any of these from this repo:
  
  # read the readme
  modal volume get hiacc-data /hiacc/Corpus/readme.txt ./
  
  # pull the JSON labels (small)
  modal volume get hiacc-data /hiacc/Corpus/children/annotations/code_switched_labels.json ./
  modal volume get hiacc-data /hiacc/Corpus/adult/annotations/code_switched_labels.json ./adult_code_switched_labels.json
  
  # pull the split files
  modal volume get hiacc-data /hiacc/Corpus/children/transcript/train_output.txt ./
  modal volume get hiacc-data /hiacc/Corpus/adult/transcription/train_output.txt ./adult_train_output.txt
  
  # pull a few audio samples to play locally
  modal volume get hiacc-data /hiacc/Corpus/children/audio/CH03001.wav ./samples/
  modal volume get hiacc-data /hiacc/Corpus/adult/audio/    ./samples_adult/   # whole dir; ~150 MB
  
  3. I can pull a small sample bundle for you (readme + both label JSONs + both split TXTs + 3 children WAVs + 3 adult WAVs, ~5 MB) into a dataset_samples/ folder so you can open them locally. Want me to do that?
  
