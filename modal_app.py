import modal

app = modal.App("hinglish-finetune")

vol_data = modal.Volume.from_name("hiacc-data", create_if_missing=True)
vol_ckpt = modal.Volume.from_name("hinglish-ckpts", create_if_missing=True)

base_image = (
    modal.Image.debian_slim()
    .apt_install("unzip")
    .pip_install("soundfile", "librosa", "numpy")
)

ZENODO_URL = "https://zenodo.org/records/15551669/files/Corpus.zip"


@app.function(image=base_image, volumes={"/data": vol_data}, timeout=1800)
def download_hiacc():
    import urllib.request, zipfile, pathlib, os

    dest_dir = pathlib.Path("/data/hiacc")
    zip_path = pathlib.Path("/data/Corpus.zip")

    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"[skip] {dest_dir} already populated")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {ZENODO_URL} -> {zip_path}")
    urllib.request.urlretrieve(ZENODO_URL, zip_path)
    size_mb = zip_path.stat().st_size / 1e6
    print(f"[download] done, {size_mb:.1f} MB")

    print(f"[unzip] -> {dest_dir}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    print(f"[unzip] done, top-level entries: {os.listdir(dest_dir)}")

    vol_data.commit()
    print("[commit] volume committed")


@app.function(image=base_image, volumes={"/data": vol_data}, timeout=600)
def inspect_hiacc():
    import pathlib, collections, random, json

    root = pathlib.Path("/data/hiacc")
    if not root.exists():
        print("[error] /data/hiacc missing — run download_hiacc first")
        return

    print(f"[tree] top-level of {root}:")
    for p in sorted(root.iterdir()):
        kind = "dir " if p.is_dir() else "file"
        print(f"  {kind}  {p.name}")

    all_files = list(root.rglob("*"))
    ext_counts = collections.Counter(p.suffix.lower() for p in all_files if p.is_file())
    print(f"\n[file-types] {dict(ext_counts)}")

    audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    audios = [p for p in all_files if p.suffix.lower() in audio_exts]
    print(f"\n[audio] {len(audios)} files")

    if audios:
        import soundfile as sf
        durations, sr_counts, ch_counts = [], collections.Counter(), collections.Counter()
        sample = random.sample(audios, min(50, len(audios)))
        for p in sample:
            try:
                info = sf.info(str(p))
                durations.append(info.duration)
                sr_counts[info.samplerate] += 1
                ch_counts[info.channels] += 1
            except Exception as e:
                print(f"  [warn] {p.name}: {e}")
        if durations:
            print(f"  duration (n={len(durations)}): min={min(durations):.2f}s "
                  f"median={sorted(durations)[len(durations)//2]:.2f}s "
                  f"max={max(durations):.2f}s")
            print(f"  sample rates: {dict(sr_counts)}")
            print(f"  channels:     {dict(ch_counts)}")

    text_exts = {".txt", ".csv", ".tsv", ".json", ".jsonl", ".trn", ".lab"}
    texts = [p for p in all_files if p.suffix.lower() in text_exts]
    print(f"\n[transcripts] {len(texts)} candidate files: "
          f"{[p.name for p in texts[:10]]}{'...' if len(texts) > 10 else ''}")

    samples_to_show = 5
    shown = 0
    devanagari = 0
    latin = 0
    other = 0
    total_chars = 0
    for p in texts:
        if shown >= samples_to_show:
            break
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  [warn] {p.name}: {e}")
            continue
        print(f"\n[transcript-sample] {p.relative_to(root)}")
        head = content[:600].replace("\n", "\\n")
        print(f"  {head}{'...' if len(content) > 600 else ''}")
        shown += 1
        for ch in content:
            cp = ord(ch)
            if 0x0900 <= cp <= 0x097F:
                devanagari += 1
            elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
                latin += 1
            else:
                other += 1
            total_chars += 1

    print(f"\n[script-detect] across sampled transcripts (total chars={total_chars}):")
    if total_chars:
        print(f"  devanagari: {devanagari} ({100*devanagari/total_chars:.1f}%)")
        print(f"  latin:      {latin} ({100*latin/total_chars:.1f}%)")
        print(f"  other:      {other} ({100*other/total_chars:.1f}%)")


deep_image = (
    modal.Image.debian_slim()
    .pip_install("transformers>=4.45.0", "soundfile", "numpy")
)


OPENSLR104_URLS = {
    "train": "https://openslr.trmal.net/resources/104/Hindi-English_train.tar.gz",
    "test":  "https://openslr.trmal.net/resources/104/Hindi-English_test.tar.gz",
}

vol_openslr = modal.Volume.from_name("openslr104-data", create_if_missing=True)


@app.function(image=base_image, volumes={"/data": vol_openslr}, timeout=2 * 3600)
def download_openslr104():
    """Download OpenSLR-104 Hindi-English code-switched corpus (train+test) into Modal Volume."""
    import urllib.request, tarfile, pathlib, time, os

    root = pathlib.Path("/data/openslr104")
    root.mkdir(parents=True, exist_ok=True)

    for split, url in OPENSLR104_URLS.items():
        dest_dir = root / split
        tar_path = root / f"{split}.tar.gz"
        if dest_dir.exists() and any(dest_dir.iterdir()):
            print(f"[skip] {dest_dir} already populated")
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"[download] {url}")
        t0 = time.time()
        urllib.request.urlretrieve(url, tar_path)
        size_mb = tar_path.stat().st_size / 1e6
        print(f"[download] done {split}: {size_mb:.1f} MB in {time.time()-t0:.0f}s")

        print(f"[extract] -> {dest_dir}")
        t0 = time.time()
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(dest_dir)
        print(f"[extract] done in {time.time()-t0:.0f}s, top-level: "
              f"{sorted(os.listdir(dest_dir))[:10]}")

    vol_openslr.commit()
    print("[commit] openslr104-data volume committed")


@app.function(image=deep_image, volumes={"/data": vol_openslr}, timeout=900)
def inspect_openslr104():
    """Inspect OpenSLR-104: audio specs, transcript format, script, segments file."""
    import pathlib, collections, json, random

    root = pathlib.Path("/data/openslr104")
    if not root.exists():
        print("[error] /data/openslr104 missing — run download_openslr104 first")
        return

    def is_devanagari(ch): return 0x0900 <= ord(ch) <= 0x097F
    def is_latin(ch):      return (0x0041 <= ord(ch) <= 0x005A) or (0x0061 <= ord(ch) <= 0x007A)

    for split in ["train", "test"]:
        split_root = root / split
        if not split_root.exists():
            print(f"[{split}] missing"); continue

        print(f"\n========== {split.upper()} ==========")
        # full tree walk for the structure
        all_files = list(split_root.rglob("*"))
        files_only = [p for p in all_files if p.is_file()]
        ext_counts = collections.Counter(p.suffix.lower() for p in files_only)
        print(f"[file-types] {dict(ext_counts.most_common(10))}")

        # top-level entries
        print(f"[tree] top-level of {split_root}:")
        for p in sorted(split_root.iterdir())[:20]:
            kind = "dir " if p.is_dir() else "file"
            print(f"  {kind}  {p.name}")

        # find key files Kaldi-style: text, segments, wav.scp, utt2spk, spk2utt
        special_names = ["text", "segments", "wav.scp", "utt2spk", "spk2utt"]
        found = {}
        for p in files_only:
            if p.name in special_names:
                found.setdefault(p.name, []).append(p)
        for name in special_names:
            if name in found:
                for p in found[name][:3]:
                    print(f"[{name}] {p.relative_to(split_root)}  size={p.stat().st_size} bytes")

        # audio specs (sample 20 wavs)
        wavs = [p for p in files_only if p.suffix.lower() in {".wav", ".flac", ".mp3"}]
        print(f"\n[audio] {len(wavs)} audio files")
        if wavs:
            import soundfile as sf
            sample = random.sample(wavs, min(20, len(wavs)))
            durations, sr_counts, ch_counts = [], collections.Counter(), collections.Counter()
            for p in sample:
                try:
                    info = sf.info(str(p))
                    durations.append(info.duration)
                    sr_counts[info.samplerate] += 1
                    ch_counts[info.channels] += 1
                except Exception as e:
                    print(f"  [warn] {p.name}: {e}")
            if durations:
                durations.sort()
                print(f"  duration (n={len(durations)}): min={durations[0]:.2f}s "
                      f"median={durations[len(durations)//2]:.2f}s max={durations[-1]:.2f}s")
                print(f"  total of sampled: {sum(durations)/60:.1f} min")
                print(f"  sample rates: {dict(sr_counts)}")
                print(f"  channels:     {dict(ch_counts)}")

        # text content + script analysis
        text_files = found.get("text", [])
        if text_files:
            tp = text_files[0]
            content = tp.read_text(encoding="utf-8", errors="replace")
            lines = [ln for ln in content.split("\n") if ln.strip()]
            print(f"\n[text] {tp.relative_to(split_root)}: {len(lines)} lines")
            print(f"[text-sample] first 5 lines:")
            for ln in lines[:5]:
                print(f"  {ln[:200]}{'...' if len(ln) > 200 else ''}")

            # script analysis on text content (skip the leading utt-id column)
            dev = lat = oth = 0
            for ln in lines:
                # Kaldi text format: "utt-id <space> transcript"
                parts = ln.split(maxsplit=1)
                if len(parts) < 2: continue
                trans = parts[1]
                for ch in trans:
                    if is_devanagari(ch): dev += 1
                    elif is_latin(ch):    lat += 1
                    else:                 oth += 1
            tot = dev + lat + oth
            if tot:
                print(f"[script] transcript chars: devanagari={100*dev/tot:.1f}%  "
                      f"latin={100*lat/tot:.1f}%  other={100*oth/tot:.1f}%")

        # segments file
        seg_files = found.get("segments", [])
        if seg_files:
            sp = seg_files[0]
            seg_lines = sp.read_text(encoding="utf-8", errors="replace").split("\n")
            seg_lines = [ln for ln in seg_lines if ln.strip()]
            print(f"\n[segments] {sp.relative_to(split_root)}: {len(seg_lines)} lines")
            print(f"[segments-sample] first 5 lines:")
            for ln in seg_lines[:5]:
                print(f"  {ln}")
            # parse durations
            durs = []
            for ln in seg_lines:
                parts = ln.split()
                if len(parts) >= 4:
                    try:
                        durs.append(float(parts[3]) - float(parts[2]))
                    except ValueError:
                        pass
            if durs:
                durs.sort()
                print(f"[segments] segment durations (n={len(durs)}): "
                      f"min={durs[0]:.2f}s p50={durs[len(durs)//2]:.2f}s "
                      f"p95={durs[int(0.95*len(durs))]:.2f}s max={durs[-1]:.2f}s")
                print(f"[segments] total covered: {sum(durs)/3600:.2f} hours")

        # wav.scp
        scp_files = found.get("wav.scp", [])
        if scp_files:
            sp = scp_files[0]
            scp_lines = [ln for ln in sp.read_text(encoding="utf-8", errors="replace").split("\n") if ln.strip()]
            print(f"\n[wav.scp] {sp.relative_to(split_root)}: {len(scp_lines)} lines")
            for ln in scp_lines[:3]:
                print(f"  {ln[:200]}")


@app.function(image=deep_image, volumes={"/data": vol_data}, timeout=900)
def inspect_hiacc_deep():
    """Answer preprocessing questions before we write train.py."""
    import pathlib, json, collections

    root = pathlib.Path("/data/hiacc/Corpus")
    cohorts = {
        "adult":    {"transcript_dir": "transcription"},
        "children": {"transcript_dir": "transcript"},
    }

    def is_devanagari(ch): return 0x0900 <= ord(ch) <= 0x097F
    def is_latin(ch):      return (0x0041 <= ord(ch) <= 0x005A) or (0x0061 <= ord(ch) <= 0x007A)

    summary = {}

    for cohort, paths in cohorts.items():
        print(f"\n========== {cohort.upper()} ==========")
        audio_root = root / cohort / "audio"
        ann_path   = root / cohort / "annotations" / "code_switched_labels.json"
        tr_dir     = root / cohort / paths["transcript_dir"]

        # Audio: indexed by basename, with the split-folder it lives in
        on_disk = {}                                            # basename -> (path, split)
        for split_dir in audio_root.iterdir():
            if split_dir.is_dir():
                for p in split_dir.iterdir():
                    if p.suffix.lower() == ".wav":
                        on_disk[p.name] = (p, split_dir.name)
        print(f"[audio] {len(on_disk)} wav files across "
              f"{set(s for _, s in on_disk.values())}")
        split_size = collections.Counter(s for _, s in on_disk.values())
        print(f"  per-split: {dict(split_size)}")

        # JSON labels
        with open(ann_path) as f:
            labels = json.load(f)
        print(f"[labels-json] {len(labels)} entries; keys: {list(labels[0].keys())}")

        json_basenames = [pathlib.Path(e.get("audio") or e.get("audio_filepath")).name for e in labels]
        resolved = sum(1 for n in json_basenames if n in on_disk)
        missing  = [n for n in json_basenames if n not in on_disk]
        extra    = [n for n in on_disk if n not in set(json_basenames)]
        print(f"  json→disk: {resolved} resolved, {len(missing)} missing, {len(extra)} extra-on-disk")
        if missing[:3]: print(f"  first missing: {missing[:3]}")
        if extra[:3]:   print(f"  first extra:   {extra[:3]}")

        # Label distribution
        label_counts = collections.Counter(e["label"] for e in labels)
        print(f"  label types (code-switching annotations): {dict(label_counts)}")

        # Char-script + length
        char_dev = char_lat = char_other = 0
        utt_lengths = []
        for e in labels:
            t = e["transcription"]
            utt_lengths.append(len(t))
            for ch in t:
                if is_devanagari(ch): char_dev += 1
                elif is_latin(ch):    char_lat += 1
                else:                 char_other += 1
        total = char_dev + char_lat + char_other
        print(f"  transcript char-script: devanagari={100*char_dev/total:.1f}%  "
              f"latin={100*char_lat/total:.1f}%  other(ws/punct/digit)={100*char_other/total:.1f}%")
        utt_lengths.sort()
        print(f"  utterance length (chars): min={utt_lengths[0]} "
              f"p50={utt_lengths[len(utt_lengths)//2]} "
              f"p95={utt_lengths[int(0.95*len(utt_lengths))]} "
              f"max={utt_lengths[-1]}")

        # Punctuation + case
        punct = collections.Counter()
        upper = lower = digit = 0
        for e in labels:
            for ch in e["transcription"]:
                if is_devanagari(ch) or ch.isspace():
                    continue
                if 0x0041 <= ord(ch) <= 0x005A: upper += 1
                elif 0x0061 <= ord(ch) <= 0x007A: lower += 1
                elif ch.isdigit(): digit += 1
                else: punct[ch] += 1
        print(f"  non-alnum chars (top 10): {punct.most_common(10)}")
        print(f"  latin case: upper={upper} lower={lower}  "
              f"({100*upper/(upper+lower):.1f}% uppercase)")
        print(f"  digits: {digit}")

        # Splits from transcript txt
        splits = {}
        for name in ["train", "val", "test"]:
            fp = tr_dir / f"{name}_output.txt"
            if not fp.exists():
                print(f"  [split-{name}] MISSING {fp}")
                continue
            entries = []
            with open(fp, encoding="utf-8") as f:
                for ln in f:
                    ln = ln.rstrip("\n")
                    if not ln.strip(): continue
                    fname, _, text = ln.partition(",")
                    entries.append((fname.strip(), text.strip()))
            splits[name] = entries
            print(f"  [split-{name}] {len(entries)} entries from {fp.name}")

        # Speaker leakage (speaker id = first 4 chars of filename)
        def speaker_of(fname): return fname[:4]
        sp_per_split = {s: set(speaker_of(f) for f, _ in entries) for s, entries in splits.items()}
        for a, b in [("train","val"), ("train","test"), ("val","test")]:
            if a in sp_per_split and b in sp_per_split:
                ov = sp_per_split[a] & sp_per_split[b]
                tag = "OK (no leak)" if not ov else f"LEAK ({len(ov)} speakers)"
                print(f"  [speaker-leak] {a}∩{b}: {tag}")

        # Cross-check: do split files match audio split folders?
        for s, entries in splits.items():
            on_disk_basenames = {n for n, (_, ds) in on_disk.items() if ds == f"{s}_split"}
            split_basenames = {f for f, _ in entries}
            both = on_disk_basenames & split_basenames
            print(f"  [split-vs-folder] {s}: split-file has {len(split_basenames)} files, "
                  f"{s}_split/ folder has {len(on_disk_basenames)} files, "
                  f"intersection {len(both)}")

        # 3 random sample lines
        import random
        random.seed(7)
        if labels:
            for e in random.sample(labels, min(3, len(labels))):
                print(f"  [sample] label={e['label']!r:20s}  text={e['transcription']}")

        summary[cohort] = {
            "n_audio_on_disk": len(on_disk),
            "n_labels_json": len(labels),
            "splits": {s: len(e) for s, e in splits.items()},
            "code_switch_label_counts": dict(label_counts),
            "audio_split_counts": dict(split_size),
            "missing_audio": len(missing),
        }

    # Tokenizer sanity (does Qwen3 tokenizer encode Devanagari without losing characters?)
    print("\n========== TOKENIZER SANITY ==========")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-ASR-0.6B", trust_remote_code=True)
        test_strings = [
            "मेरा favourite festival diwali है",
            "इसका जो reason  है कि that it is a festival of light",
            "क्या आप मुझे अपनी favourite dish के बारे में बता सकते हैं",
        ]
        for s in test_strings:
            ids = tok.encode(s, add_special_tokens=False)
            decoded = tok.decode(ids)
            ok = decoded.strip() == s.strip()
            print(f"  text:     {s}")
            print(f"  n_tokens: {len(ids)}   roundtrip-exact: {ok}")
            if not ok:
                print(f"  decoded:  {decoded}")
    except Exception as e:
        print(f"  [skip] could not load Qwen3 tokenizer: {e}")

    print("\n========== SUMMARY ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


@app.function(image=base_image, volumes={"/data": vol_data}, timeout=600)
def prepare_hiacc_jsonl():
    """Convert HiACC into train/val/test JSONL files for qwen3_asr_sft.py.

    Output format per line:
      {"audio": "/data/hiacc/Corpus/<cohort>/audio/<split>_split/<file>.wav",
       "text":  "language None<asr_text>HiACC label as-is"}

    Decisions (see docs/technical_v1_hiacc.md):
      - language = None (Polyglot-Lion-style, language-agnostic decoding)
      - mixed-script transcripts preserved as-is (no romanization)
      - punctuation and case PRESERVED (HiACC fidelity; Qwen3-ASR base outputs both)
      - only normalization: collapse runs of whitespace
      - splits from audio folder layout (HiACC's own, both cohorts)
      - basename.strip() to fix 9 children entries with trailing-whitespace IDs
    """
    import pathlib, json, re, collections

    root    = pathlib.Path("/data/hiacc/Corpus")
    out_dir = pathlib.Path("/data/hiacc/jsonl")
    out_dir.mkdir(parents=True, exist_ok=True)

    WS_RE = re.compile(r"\s+")

    def normalize(text: str) -> str:
        # Preserve HiACC label fidelity: no lowercase, no punctuation stripping.
        # Only collapse internal whitespace and trim ends.
        return WS_RE.sub(" ", text).strip()

    cohorts = ["adult", "children"]
    split_lines = {"train": [], "val": [], "test": []}
    stats = collections.defaultdict(lambda: collections.defaultdict(int))

    for cohort in cohorts:
        ann_path = root / cohort / "annotations" / "code_switched_labels.json"
        with open(ann_path, encoding="utf-8") as f:
            labels = json.load(f)

        bn_to_text = {}
        dups = 0
        for e in labels:
            raw_path = e.get("audio") or e.get("audio_filepath")
            bn = pathlib.Path(raw_path).name.strip()
            if bn in bn_to_text:
                dups += 1
            bn_to_text[bn] = e["transcription"]
        if dups:
            print(f"[{cohort}] {dups} duplicate basenames in labels JSON (later wins)")

        for split in ["train", "val", "test"]:
            audio_dir = root / cohort / "audio" / f"{split}_split"
            if not audio_dir.exists():
                print(f"[warn] missing {audio_dir}")
                continue
            wavs = sorted(audio_dir.glob("*.wav"))
            for wav in wavs:
                bn = wav.name
                text = bn_to_text.get(bn)
                if text is None:
                    print(f"[warn] {cohort}/{split}: no label for {bn}")
                    stats[cohort]["missing_label"] += 1
                    continue
                norm = normalize(text)
                if not norm:
                    print(f"[warn] {cohort}/{split}: empty after normalize for {bn} "
                          f"(orig={text!r})")
                    stats[cohort][f"{split}_empty"] += 1
                    continue
                split_lines[split].append({
                    "audio": str(wav),
                    "text":  f"language None<asr_text>{norm}",
                    # extra metadata (not consumed by the trainer but useful for debugging)
                    "_cohort": cohort,
                    "_basename": bn,
                })
                stats[cohort][split] += 1

    # Deterministic order: sort by audio path
    for s in split_lines:
        split_lines[s].sort(key=lambda d: d["audio"])

    # Write JSONL (drop the underscore keys so the file is what the trainer reads)
    train_keys = {"audio", "text"}
    for split, lines in split_lines.items():
        out_path = out_dir / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for d in lines:
                row = {k: v for k, v in d.items() if k in train_keys}
                f.write(json.dumps(row, ensure_ascii=False))
                f.write("\n")
        print(f"[write] {out_path}: {len(lines)} lines")

    # Also write a sidecar with cohort/basename metadata (so eval can slice by cohort)
    for split, lines in split_lines.items():
        meta_path = out_dir / f"{split}.meta.jsonl"
        with open(meta_path, "w", encoding="utf-8") as f:
            for d in lines:
                f.write(json.dumps({
                    "audio":    d["audio"],
                    "cohort":   d["_cohort"],
                    "basename": d["_basename"],
                    "ref_text": d["text"].split("<asr_text>", 1)[1],
                }, ensure_ascii=False))
                f.write("\n")
        print(f"[write] {meta_path}: {len(lines)} lines (sidecar for eval)")

    # First 3 lines of each train/val/test file for eyeballing
    for split in ["train", "val", "test"]:
        out_path = out_dir / f"{split}.jsonl"
        print(f"\n========== {split}.jsonl (first 3 lines) ==========")
        with open(out_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3: break
                print(line.rstrip())

    print(f"\n========== stats per cohort ==========")
    for cohort in cohorts:
        print(f"  {cohort}: {dict(stats[cohort])}")
    print(f"\n========== totals ==========")
    for s in ["train", "val", "test"]:
        print(f"  {s}: {len(split_lines[s])} utterances")

    vol_data.commit()


QWEN3_ASR_COMMIT = "c17a131fe028b2e428b6e80a33d30bb4fa57b8df"

train_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "ffmpeg", "build-essential")
    .pip_install("torch==2.5.1", "wheel", "packaging", "ninja")
    .pip_install("qwen-asr==0.0.6", "datasets", "jiwer", "huggingface_hub")
    .pip_install(
        "flash-attn==2.7.4.post1",
        extra_options="--no-build-isolation",
    )
    .run_commands(
        "git clone https://github.com/QwenLM/Qwen3-ASR.git /opt/Qwen3-ASR",
        f"cd /opt/Qwen3-ASR && git checkout {QWEN3_ASR_COMMIT}",
    )
    .env({"HF_HOME": "/hf_cache", "TRANSFORMERS_CACHE": "/hf_cache"})
    .add_local_file("docs/hf_model_card.md",    "/opt/hf_model_card.md")
    .add_local_file("docs/hf_model_card_v2.md", "/opt/hf_model_card_v2.md")
    .add_local_file("docs/hf_model_card_v3.md", "/opt/hf_model_card_v3.md")
    .add_local_file("docs/figures/v1_training_curves.png", "/opt/v1_training_curves.png")
    .add_local_file("docs/figures/v2_training_curves.png", "/opt/v2_training_curves.png")
    .add_local_file("docs/figures/v3_training_curves.png", "/opt/v3_training_curves.png")
)

vol_hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


def _train_cmd(
    train_file: str,
    eval_file: str,
    output_dir: str,
    epochs: float,
    save_steps: int,
    log_steps: int = 10,
    save_total_limit: int = 5,
):
    """Build the torchrun command for qwen3_asr_sft.py."""
    return [
        "torchrun", "--nproc_per_node=2",
        "/opt/Qwen3-ASR/finetuning/qwen3_asr_sft.py",
        "--model_path",         "Qwen/Qwen3-ASR-0.6B",
        "--train_file",         train_file,
        "--eval_file",          eval_file,
        "--output_dir",         output_dir,
        "--batch_size",         "8",
        "--grad_acc",           "2",
        "--lr",                 "2e-5",
        "--epochs",             str(epochs),
        "--log_steps",          str(log_steps),
        "--save_strategy",      "steps",
        "--save_steps",         str(save_steps),
        "--save_total_limit",   str(save_total_limit),
        "--lr_scheduler_type",  "linear",
        "--warmup_ratio",       "0.02",
        "--num_workers",        "4",
        "--pin_memory",         "1",
        "--persistent_workers", "1",
        "--prefetch_factor",    "2",
    ]


def _run_training(cmd: list, output_dir, vol_ckpt, vol_data):
    import os, subprocess, time, json, pathlib

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "train.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    # Avoid HF Trainer trying to push to Hub
    env["HF_HUB_OFFLINE"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"

    print(f"[cmd] {' '.join(cmd)}")
    print(f"[log] tee -> {log_path}")
    t0 = time.time()
    with open(log_path, "w") as log_f:
        # Tee: send stdout/stderr to log file AND parent stdout
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True,
        )
        for line in proc.stdout:
            log_f.write(line)
            log_f.flush()
            print(line, end="")
        rc = proc.wait()

    wall = time.time() - t0
    (output_dir / "run.json").write_text(json.dumps({
        "command":             cmd,
        "exit_code":           rc,
        "wall_clock_seconds":  wall,
        "qwen3_asr_commit":    QWEN3_ASR_COMMIT,
    }, indent=2))

    vol_ckpt.commit()
    print(f"\n[done] exit_code={rc}  wall={wall:.0f}s  output_dir={output_dir}")
    if rc != 0:
        raise RuntimeError(f"training failed with exit code {rc}")


@app.function(
    image=train_image,
    gpu="H100:2",
    volumes={"/data": vol_data, "/ckpt": vol_ckpt, "/hf_cache": vol_hf_cache},
    timeout=6 * 3600,
)
def smoke_v1_hiacc():
    """Quick pipeline smoke test: ~6 steps, ~5 min, ~$0.50. Verifies data flow + DDP."""
    cmd = _train_cmd(
        train_file="/data/hiacc/jsonl/train.jsonl",
        eval_file="/data/hiacc/jsonl/val.jsonl",
        output_dir="/ckpt/smoke-v1-hiacc-h100x2",
        epochs=0.05,
        save_steps=5,
        log_steps=1,
    )
    _run_training(cmd, "/ckpt/smoke-v1-hiacc-h100x2", vol_ckpt, vol_data)


@app.function(
    image=train_image,
    gpu="H100:2",
    volumes={"/data": vol_openslr, "/ckpt": vol_ckpt, "/hf_cache": vol_hf_cache},
    timeout=8 * 3600,
)
def smoke_v2_openslr():
    """Quick pipeline smoke test for v2 on OpenSLR-104 chunks. ~5-10 min, ~$1."""
    cmd = _train_cmd(
        train_file="/data/openslr104/jsonl/train.jsonl",
        eval_file="/data/openslr104/jsonl/val.jsonl",
        output_dir="/ckpt/smoke-v2-openslr-h100x2",
        epochs=0.005,                                  # ~8 steps at 1562 steps/epoch
        save_steps=4,
        log_steps=1,
    )
    _run_training(cmd, "/ckpt/smoke-v2-openslr-h100x2", vol_ckpt, vol_openslr)


@app.function(
    image=train_image,
    gpu="H100:2",
    volumes={
        "/data_hiacc":   vol_data,
        "/data_openslr": vol_openslr,
        "/ckpt":         vol_ckpt,
        "/hf_cache":     vol_hf_cache,
    },
    timeout=12 * 3600,
)
def smoke_v3_union():
    """v3 pipeline smoke: ~10 steps on union, both volumes mounted, ~5 min."""
    cmd = _train_cmd(
        train_file="/data_openslr/openslr104/jsonl/train_union.jsonl",
        eval_file="/data_openslr/openslr104/jsonl/val_union.jsonl",
        output_dir="/ckpt/smoke-v3-union-h100x2",
        epochs=0.006,                                  # ~10 steps at union scale
        save_steps=5,
        log_steps=1,
    )
    _run_training(cmd, "/ckpt/smoke-v3-union-h100x2", vol_ckpt, vol_openslr)


@app.function(
    image=train_image,
    gpu="H100:2",
    volumes={
        "/data_hiacc":   vol_data,
        "/data_openslr": vol_openslr,
        "/ckpt":         vol_ckpt,
        "/hf_cache":     vol_hf_cache,
    },
    timeout=12 * 3600,
)
def train_v3_union():
    """Full v3 union fine-tune: HiACC + OpenSLR-104. 2 epochs, ~70 min, ~$10.

    Train size ~53,627 utts → 1676 steps/epoch (effective batch 32).
    2 epochs → ~3,350 steps. Eval/save every 200 → ~17 checkpoints. save_total_limit=30.
    """
    cmd = _train_cmd(
        train_file="/data_openslr/openslr104/jsonl/train_union.jsonl",
        eval_file="/data_openslr/openslr104/jsonl/val_union.jsonl",
        output_dir="/ckpt/v3-union-h100x2",
        epochs=2,
        save_steps=200,
        log_steps=20,
        save_total_limit=30,
    )
    _run_training(cmd, "/ckpt/v3-union-h100x2", vol_ckpt, vol_openslr)


@app.function(
    image=train_image,
    gpu="H100:2",
    volumes={"/data": vol_openslr, "/ckpt": vol_ckpt, "/hf_cache": vol_hf_cache},
    timeout=8 * 3600,
)
def train_v2_openslr():
    """Full v2 fine-tune: OpenSLR-104 train/val, 3 epochs, ~1.5-2 h wall-clock, ~$12-16."""
    cmd = _train_cmd(
        train_file="/data/openslr104/jsonl/train.jsonl",
        eval_file="/data/openslr104/jsonl/val.jsonl",
        output_dir="/ckpt/v2-openslr-h100x2",
        epochs=3,
        save_steps=200,
        log_steps=20,
        save_total_limit=30,
    )
    _run_training(cmd, "/ckpt/v2-openslr-h100x2", vol_ckpt, vol_openslr)


@app.function(
    image=train_image,
    gpu="H100:2",
    volumes={"/data": vol_data, "/ckpt": vol_ckpt, "/hf_cache": vol_hf_cache},
    timeout=6 * 3600,
)
def train_v1_hiacc():
    """Full v1 fine-tune: HiACC train/val, 5 epochs, ~1.5-3 h wall-clock, ~$12-24.
    Keeps all ~11 checkpoints (save_total_limit=15) so we can pick best by eval_loss."""
    cmd = _train_cmd(
        train_file="/data/hiacc/jsonl/train.jsonl",
        eval_file="/data/hiacc/jsonl/val.jsonl",
        output_dir="/ckpt/v1-hiacc-h100x2",
        epochs=5,
        save_steps=50,
        log_steps=10,
        save_total_limit=15,
    )
    _run_training(cmd, "/ckpt/v1-hiacc-h100x2", vol_ckpt, vol_data)


@app.function(
    image=train_image,
    gpu="H100:1",                                       # single GPU is fine for inference
    volumes={"/data": vol_data, "/ckpt": vol_ckpt, "/hf_cache": vol_hf_cache},
    timeout=2 * 3600,
)
def eval_v1_hiacc(
    output_dir: str = "/ckpt/v1-hiacc-h100x2",
    checkpoint: str = "",                               # "" = auto-pick best by eval_loss
    batch_size: int = 16,
):
    """Evaluate fine-tuned vs zero-shot Qwen3-ASR-0.6B on HiACC test split.

    - Picks best checkpoint by eval_loss (from trainer_state.json), unless overridden.
    - Batched transcription on all 1,036 test utterances.
    - WER overall + sliced by adult/child (using *.meta.jsonl sidecar).
    - Writes eval.json + predictions.jsonl into output_dir.
    """
    import json, re, pathlib, time
    import torch
    from qwen_asr import Qwen3ASRModel
    import jiwer

    out_dir   = pathlib.Path(output_dir)
    test_path = pathlib.Path("/data/hiacc/jsonl/test.jsonl")
    meta_path = pathlib.Path("/data/hiacc/jsonl/test.meta.jsonl")

    # ---- 1. Pick best checkpoint by eval_loss ----
    def find_best_checkpoint(out_dir: pathlib.Path):
        ckpts = sorted(
            out_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )
        if not ckpts:
            raise RuntimeError(f"no checkpoints in {out_dir}")
        # Latest checkpoint's trainer_state.json has the full eval history
        state_path = ckpts[-1] / "trainer_state.json"
        with open(state_path) as f:
            state = json.load(f)
        best_step, best_loss = None, float("inf")
        eval_history = []
        for entry in state.get("log_history", []):
            if "eval_loss" in entry:
                eval_history.append((entry["step"], entry["eval_loss"]))
                if entry["eval_loss"] < best_loss:
                    best_loss = entry["eval_loss"]
                    best_step = entry["step"]
        if best_step is None:
            raise RuntimeError("no eval_loss entries in trainer_state.json")
        best_ckpt = out_dir / f"checkpoint-{best_step}"
        if not best_ckpt.exists():
            raise RuntimeError(f"best checkpoint {best_ckpt} was not saved "
                               f"(save_total_limit too tight?)")
        return best_ckpt, best_loss, eval_history, ckpts

    if checkpoint:
        best_ckpt = out_dir / checkpoint
        best_loss = None
        eval_history = []
        ckpts = sorted(out_dir.glob("checkpoint-*"))
    else:
        best_ckpt, best_loss, eval_history, ckpts = find_best_checkpoint(out_dir)
    print(f"[best] {best_ckpt}  eval_loss={best_loss}")
    print(f"[eval-history] {eval_history}")

    # ---- 1b. Repair: copy preprocessor/processor/chat-template files from base
    # Qwen3-ASR's `qwen3_asr_sft.py` MakeEveryCheckpointInferableCallback only copies
    # these files if model_path is a local dir. Since we trained with the HF Hub ID,
    # it copied nothing. Restore them here from the cached base-model snapshot.
    def ensure_checkpoint_inferable(ckpt_dir: pathlib.Path, base_id: str):
        import shutil
        from huggingface_hub import snapshot_download
        required = [
            "preprocessor_config.json",
            "processor_config.json",
            "chat_template.json",
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "merges.txt",
            "vocab.json",
        ]
        missing = [f for f in required if not (ckpt_dir / f).exists()]
        if not missing:
            print(f"[ckpt-fix] {ckpt_dir.name}: nothing to copy")
            return
        base_dir = pathlib.Path(snapshot_download(base_id))
        for f in missing:
            src = base_dir / f
            if src.exists():
                shutil.copy2(src, ckpt_dir / f)
                print(f"[ckpt-fix] {ckpt_dir.name}: copied {f}")
            else:
                print(f"[ckpt-fix] {ckpt_dir.name}: WARN '{f}' not in base snapshot")

    ensure_checkpoint_inferable(best_ckpt, "Qwen/Qwen3-ASR-0.6B")

    # ---- 2. Load test entries + sidecar ----
    test_entries = [json.loads(l) for l in open(test_path)]
    meta_entries = [json.loads(l) for l in open(meta_path)]
    assert len(test_entries) == len(meta_entries), \
        f"test={len(test_entries)} meta={len(meta_entries)}"
    print(f"[test] {len(test_entries)} utterances")
    print(f"[cohorts] adult={sum(1 for m in meta_entries if m['cohort']=='adult')} "
          f"children={sum(1 for m in meta_entries if m['cohort']=='children')}")

    # ---- 3. Eval-time normalizer (Polyglot-Lion style) — applied symmetrically ----
    EVAL_PUNCT_RE = re.compile(r"[\.,\?!\"'|\-\/“”…]")
    WS_RE = re.compile(r"\s+")
    def eval_normalize(text: str) -> str:
        t = text.strip().lower()
        t = EVAL_PUNCT_RE.sub(" ", t)
        t = WS_RE.sub(" ", t).strip()
        return t

    def strip_prefix(text: str) -> str:
        # Model output may include "language X<asr_text>..."; ref JSONLs do too.
        return text.split("<asr_text>", 1)[1] if "<asr_text>" in text else text

    # ---- 4. Inference helper ----
    def run_inference(model_path: str, label: str):
        print(f"\n========== {label} ==========")
        print(f"[model] {model_path}")
        t0 = time.time()
        model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2",
        )
        print(f"[load] {time.time()-t0:.1f}s")

        hyps_raw = []
        langs = []
        t0 = time.time()
        n = len(test_entries)
        for i in range(0, n, batch_size):
            batch = test_entries[i:i + batch_size]
            audios = [ex["audio"] for ex in batch]
            try:
                results = model.transcribe(audio=audios, language=None)
            except Exception as e:
                # Fall back to one-at-a-time if batch fails
                print(f"[warn] batch starting {i} failed ({e}); retrying singly")
                results = []
                for a in audios:
                    results.extend(model.transcribe(audio=a, language=None))
            for r in results:
                hyps_raw.append(r.text)
                langs.append(getattr(r, "language", "") or "")
            done = i + len(batch)
            if (i // batch_size) % 4 == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (n - done) / max(rate, 1e-6)
                print(f"[infer] {done}/{n}  rate={rate:.1f}/s  eta={eta:.0f}s")
        del model
        torch.cuda.empty_cache()
        return hyps_raw, langs

    # ---- 5. WER (overall + per-cohort), raw & normalized ----
    def compute_wer(refs, hyps):
        if not refs:
            return float("nan")
        return jiwer.wer(refs, hyps)

    def metrics_for(hyps_raw, refs_raw, meta):
        refs_norm = [eval_normalize(strip_prefix(r)) for r in refs_raw]
        hyps_norm = [eval_normalize(strip_prefix(h)) for h in hyps_raw]
        adult_idx = [i for i, m in enumerate(meta) if m["cohort"] == "adult"]
        child_idx = [i for i, m in enumerate(meta) if m["cohort"] == "children"]
        return {
            "wer_overall":  compute_wer(refs_norm, hyps_norm),
            "wer_adult":    compute_wer([refs_norm[i] for i in adult_idx],
                                        [hyps_norm[i] for i in adult_idx]),
            "wer_children": compute_wer([refs_norm[i] for i in child_idx],
                                        [hyps_norm[i] for i in child_idx]),
            "n_overall":    len(refs_norm),
            "n_adult":      len(adult_idx),
            "n_children":   len(child_idx),
        }

    refs_raw = [ex["text"] for ex in test_entries]

    # ---- 6. Run both models ----
    ft_hyps,   ft_langs   = run_inference(str(best_ckpt),         "FINE-TUNED")
    base_hyps, base_langs = run_inference("Qwen/Qwen3-ASR-0.6B",  "ZERO-SHOT BASELINE")

    ft_metrics   = metrics_for(ft_hyps,   refs_raw, meta_entries)
    base_metrics = metrics_for(base_hyps, refs_raw, meta_entries)

    # ---- 7. Print sample predictions side-by-side ----
    print("\n========== SAMPLE PREDICTIONS ==========")
    sample_idxs = [0, 1, 100, 250, 500, 750, 1000]
    for i in sample_idxs:
        if i >= len(test_entries): continue
        m = meta_entries[i]
        print(f"\n[{i}] cohort={m['cohort']}  basename={m['basename']}")
        print(f"  REF:        {strip_prefix(refs_raw[i])[:160]}")
        print(f"  BASELINE:   {strip_prefix(base_hyps[i])[:160]}")
        print(f"  FINE-TUNED: {strip_prefix(ft_hyps[i])[:160]}")

    # ---- 8. Write eval.json + predictions.jsonl ----
    eval_out = {
        "checkpoint":           best_ckpt.name,
        "checkpoint_eval_loss": best_loss,
        "all_checkpoints":      [c.name for c in ckpts],
        "eval_history":         eval_history,
        "finetuned":            ft_metrics,
        "baseline_zero_shot":   base_metrics,
        "wer_reduction": {
            "overall":  base_metrics["wer_overall"]  - ft_metrics["wer_overall"],
            "adult":    base_metrics["wer_adult"]    - ft_metrics["wer_adult"],
            "children": base_metrics["wer_children"] - ft_metrics["wer_children"],
        },
    }
    (out_dir / "eval.json").write_text(json.dumps(eval_out, indent=2))

    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for i, m in enumerate(meta_entries):
            f.write(json.dumps({
                "audio":          test_entries[i]["audio"],
                "cohort":         m["cohort"],
                "basename":       m["basename"],
                "ref":            strip_prefix(refs_raw[i]),
                "baseline":       strip_prefix(base_hyps[i]),
                "baseline_lang":  base_langs[i],
                "finetuned":      strip_prefix(ft_hyps[i]),
                "finetuned_lang": ft_langs[i],
            }, ensure_ascii=False) + "\n")

    vol_ckpt.commit()

    print("\n========== FINAL ==========")
    print(json.dumps(eval_out, indent=2))


@app.local_entrypoint()
def prepare_jsonl():
    prepare_hiacc_jsonl.remote()


@app.local_entrypoint()
def smoke():
    smoke_v1_hiacc.remote()


@app.local_entrypoint()
def train():
    train_v1_hiacc.remote()


@app.local_entrypoint()
def smoke_v2():
    smoke_v2_openslr.remote()


@app.local_entrypoint()
def train_v2():
    train_v2_openslr.remote()


@app.local_entrypoint()
def smoke_v3():
    smoke_v3_union.remote()


@app.local_entrypoint()
def train_v3():
    train_v3_union.remote()


@app.cls(
    image=train_image,
    gpu="A10G",
    volumes={"/ckpt": vol_ckpt, "/hf_cache": vol_hf_cache},
    timeout=600,
    scaledown_window=180,
)
class RealtimeASR:
    """Warm-container Qwen3-ASR inference for realtime use from a local mic.

    Parameterised by `model_id` — different values get their own container set, so
    you can A/B between {v1 HF repo, v2 local checkpoint, base} without thrashing.
    Lifetime: container stays warm 3 minutes after the last call (scaledown_window).
    """
    model_id: str = modal.parameter()

    @modal.enter()
    def load(self):
        import pathlib, shutil, time
        import torch
        from qwen_asr import Qwen3ASRModel
        from huggingface_hub import snapshot_download

        path = self.model_id
        ckpt = pathlib.Path(path)
        if ckpt.exists() and ckpt.is_dir():
            required = [
                "preprocessor_config.json", "processor_config.json", "chat_template.json",
                "config.json", "generation_config.json",
                "tokenizer_config.json", "tokenizer.json",
                "special_tokens_map.json", "added_tokens.json",
                "merges.txt", "vocab.json",
            ]
            missing = [f for f in required if not (ckpt / f).exists()]
            if missing:
                base = pathlib.Path(snapshot_download("Qwen/Qwen3-ASR-0.6B"))
                for f in missing:
                    src = base / f
                    if src.exists():
                        shutil.copy2(src, ckpt / f)
                        print(f"[ckpt-fix] copied {f}")

        print(f"[load] model_id={path}")
        t0 = time.time()
        self.model = Qwen3ASRModel.from_pretrained(
            path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2",
        )
        print(f"[load] ready in {time.time()-t0:.1f}s")

    @modal.method()
    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> dict:
        """Transcribe a WAV byte-string. Returns text + detected/forced language."""
        import io, time
        import numpy as np
        import soundfile as sf

        wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)

        t0 = time.time()
        results = self.model.transcribe(audio=(wav, int(sr)), language=language)
        return {
            "text":     results[0].text,
            "language": str(results[0].language or ""),
            "ms":       int((time.time() - t0) * 1000),
            "dur_s":    float(len(wav) / sr),
        }


@app.function(
    image=train_image,
    volumes={"/ckpt": vol_ckpt, "/hf_cache": vol_hf_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=2 * 3600,
)
def push_v1_to_hub(
    output_dir: str = "/ckpt/v1-hiacc-h100x2",
    checkpoint: str = "",                                         # "" = read from eval.json
    repo_id: str = "Surajgameramp/qwen3-asr-0.6b-hinglish-hiacc-v1",
    private: bool = False,
):
    """Upload the best fine-tuned checkpoint + model card to Hugging Face Hub."""
    import os, json, pathlib, shutil
    from huggingface_hub import HfApi, snapshot_download

    out_dir = pathlib.Path(output_dir)

    # Pick checkpoint (default: best per eval.json)
    if not checkpoint:
        eval_path = out_dir / "eval.json"
        if not eval_path.exists():
            raise RuntimeError(f"no eval.json at {eval_path}; pass --checkpoint")
        with open(eval_path) as f:
            checkpoint = json.load(f)["checkpoint"]
    ckpt_dir = out_dir / checkpoint
    if not ckpt_dir.exists():
        raise RuntimeError(f"checkpoint dir not found: {ckpt_dir}")
    print(f"[push] checkpoint={ckpt_dir}")

    # Repair: ensure preprocessor_config / chat_template / processor_config are present
    # (the official sft.py callback only copies these if model_path is a local dir; ours wasn't)
    required = [
        "preprocessor_config.json",
        "processor_config.json",
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
    ]
    missing = [f for f in required if not (ckpt_dir / f).exists()]
    if missing:
        base_dir = pathlib.Path(snapshot_download("Qwen/Qwen3-ASR-0.6B"))
        for f in missing:
            src = base_dir / f
            if src.exists():
                shutil.copy2(src, ckpt_dir / f)
                print(f"[ckpt-fix] copied {f}")
            else:
                print(f"[ckpt-fix] WARN '{f}' not in base snapshot, skipping")

    # Drop README.md (model card) into the checkpoint dir
    shutil.copy2("/opt/hf_model_card.md", ckpt_dir / "README.md")

    # Drop training curves alongside for transparency
    figs_dir = ckpt_dir / "figures"
    figs_dir.mkdir(exist_ok=True)
    shutil.copy2("/opt/v1_training_curves.png", figs_dir / "training_curves.png")

    # Also copy eval.json + predictions.jsonl from output_dir (one level up) so
    # users see the metrics alongside the weights.
    for f in ("eval.json", "predictions.jsonl"):
        src = out_dir / f
        if src.exists():
            shutil.copy2(src, ckpt_dir / f)
            print(f"[copy] {f} alongside weights")

    # Upload
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not in environment (Modal secret 'huggingface' missing or empty)")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(ckpt_dir),
        repo_id=repo_id,
        ignore_patterns=[
            "optimizer.pt",
            "scheduler.pt",
            "rng_state_*.pth",
            "training_args.bin",
        ],
        commit_message=f"Upload v1 fine-tune from {checkpoint}",
    )
    print(f"[push] done → https://huggingface.co/{repo_id}")


@app.function(
    image=train_image,
    gpu="H100:1",
    volumes={
        "/data_hiacc":   vol_data,
        "/data_openslr": vol_openslr,
        "/ckpt":         vol_ckpt,
        "/hf_cache":     vol_hf_cache,
    },
    timeout=2 * 3600,
)
def eval_v2_openslr(
    output_dir: str = "/ckpt/v2-openslr-h100x2",
    checkpoint: str = "",                                       # "" = best by eval_loss
    batch_size: int = 16,
):
    """v2 eval: WER for {fine-tuned, zero-shot baseline} × {OpenSLR test, HiACC test}.

    OpenSLR test = in-domain. HiACC test = cross-domain.
    Writes /ckpt/v2-openslr-h100x2/eval.json + predictions.jsonl files.
    """
    import json, re, pathlib, time
    import torch
    from qwen_asr import Qwen3ASRModel
    import jiwer

    out_dir = pathlib.Path(output_dir)

    # ---- 1. Pick best checkpoint by eval_loss ----
    def find_best_checkpoint(out_dir: pathlib.Path):
        ckpts = sorted(
            out_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )
        if not ckpts:
            raise RuntimeError(f"no checkpoints in {out_dir}")
        with open(ckpts[-1] / "trainer_state.json") as f:
            state = json.load(f)
        best_step, best_loss = None, float("inf")
        eval_history = []
        for entry in state.get("log_history", []):
            if "eval_loss" in entry:
                eval_history.append((entry["step"], entry["eval_loss"]))
                if entry["eval_loss"] < best_loss:
                    best_loss = entry["eval_loss"]
                    best_step = entry["step"]
        return out_dir / f"checkpoint-{best_step}", best_loss, eval_history, ckpts

    if checkpoint:
        best_ckpt = out_dir / checkpoint
        best_loss = None
        eval_history = []
        ckpts = sorted(out_dir.glob("checkpoint-*"))
    else:
        best_ckpt, best_loss, eval_history, ckpts = find_best_checkpoint(out_dir)
    print(f"[best] {best_ckpt}  eval_loss={best_loss}")

    # Repair: copy missing preprocessor/chat-template files from base snapshot
    def ensure_inferable(ckpt_dir, base_id="Qwen/Qwen3-ASR-0.6B"):
        import shutil
        from huggingface_hub import snapshot_download
        required = [
            "preprocessor_config.json", "processor_config.json", "chat_template.json",
            "config.json", "generation_config.json",
            "tokenizer_config.json", "tokenizer.json",
            "special_tokens_map.json", "added_tokens.json",
            "merges.txt", "vocab.json",
        ]
        missing = [f for f in required if not (ckpt_dir / f).exists()]
        if not missing: return
        base = pathlib.Path(snapshot_download(base_id))
        for f in missing:
            src = base / f
            if src.exists():
                shutil.copy2(src, ckpt_dir / f)
                print(f"[ckpt-fix] copied {f}")
    ensure_inferable(best_ckpt)

    # ---- 2. Eval-time normalizer (Polyglot-Lion style) ----
    EVAL_PUNCT_RE = re.compile(r"[\.,\?!\"'|\-\/“”…]")
    WS_RE = re.compile(r"\s+")
    def eval_normalize(text):
        t = text.strip().lower()
        t = EVAL_PUNCT_RE.sub(" ", t)
        t = WS_RE.sub(" ", t).strip()
        return t
    def strip_prefix(text):
        return text.split("<asr_text>", 1)[1] if "<asr_text>" in text else text

    # ---- 3. Load test sets ----
    # JSONLs were written with the volume mounted at /data; here we mount both
    # at different paths, so audio paths need remapping at load time.
    PATH_REMAP = [
        ("/data/openslr104/", "/data_openslr/openslr104/"),
        ("/data/hiacc/",      "/data_hiacc/hiacc/"),
    ]
    def remap(p):
        for a, b in PATH_REMAP:
            if p.startswith(a):
                return p.replace(a, b, 1)
        return p

    test_sets = {}
    for name, path in [
        ("openslr", "/data_openslr/openslr104/jsonl/test.jsonl"),
        ("hiacc",   "/data_hiacc/hiacc/jsonl/test.jsonl"),
    ]:
        entries = []
        for line in open(path):
            d = json.loads(line)
            d["audio"] = remap(d["audio"])
            entries.append(d)
        # meta sidecar for cohort/utt slicing
        meta_path = path.replace(".jsonl", ".meta.jsonl")
        meta = None
        if pathlib.Path(meta_path).exists():
            meta = []
            for line in open(meta_path):
                m = json.loads(line)
                if "audio" in m:
                    m["audio"] = remap(m["audio"])
                meta.append(m)
        test_sets[name] = {"path": path, "entries": entries, "meta": meta}
        print(f"[test:{name}] {len(entries)} utterances "
              f"(first audio: {entries[0]['audio']})")

    # ---- 4. Inference helper ----
    def run_inference(model_path, label, entries):
        print(f"\n[{label}] loading {model_path}")
        t0 = time.time()
        model = Qwen3ASRModel.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="flash_attention_2",
        )
        print(f"[{label}] load {time.time()-t0:.1f}s, n={len(entries)}")
        hyps = []
        t0 = time.time()
        for i in range(0, len(entries), batch_size):
            batch  = entries[i:i + batch_size]
            audios = [ex["audio"] for ex in batch]
            try:
                results = model.transcribe(audio=audios, language=None)
            except Exception as e:
                print(f"[warn] batch starting {i} failed ({e}); retrying singly")
                results = []
                for a in audios:
                    results.extend(model.transcribe(audio=a, language=None))
            for r in results:
                hyps.append(r.text)
            done = i + len(batch)
            if (i // batch_size) % 16 == 0 or done == len(entries):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (len(entries) - done) / max(rate, 1e-6)
                print(f"[{label}] {done}/{len(entries)}  rate={rate:.1f}/s  eta={eta:.0f}s")
        del model
        torch.cuda.empty_cache()
        return hyps

    def wer(refs, hyps):
        if not refs: return float("nan")
        return jiwer.wer(refs, hyps)

    def metrics_for(hyps_raw, refs_raw, meta=None):
        refs_norm = [eval_normalize(strip_prefix(r)) for r in refs_raw]
        hyps_norm = [eval_normalize(strip_prefix(h)) for h in hyps_raw]
        out = {"wer_overall": wer(refs_norm, hyps_norm), "n": len(refs_norm)}
        if meta and "cohort" in meta[0]:
            adult_idx = [i for i, m in enumerate(meta) if m["cohort"] == "adult"]
            child_idx = [i for i, m in enumerate(meta) if m["cohort"] == "children"]
            out["wer_adult"]    = wer([refs_norm[i] for i in adult_idx],
                                      [hyps_norm[i] for i in adult_idx])
            out["wer_children"] = wer([refs_norm[i] for i in child_idx],
                                      [hyps_norm[i] for i in child_idx])
            out["n_adult"]      = len(adult_idx)
            out["n_children"]   = len(child_idx)
        return out

    # ---- 5. Run all 4 combinations (fine-tuned & base, on OpenSLR & HiACC) ----
    all_results = {}
    for model_label, model_path in [("finetuned", str(best_ckpt)),
                                    ("baseline",  "Qwen/Qwen3-ASR-0.6B")]:
        for test_label, t in test_sets.items():
            label = f"{model_label}_{test_label}"
            hyps = run_inference(model_path, label, t["entries"])
            refs = [ex["text"] for ex in t["entries"]]
            m = metrics_for(hyps, refs, t["meta"])
            print(f"\n[{label}] WER = {m}")
            all_results[label] = {"metrics": m, "hyps": hyps, "refs_raw": refs}

    # ---- 6. Sample predictions ----
    def print_samples(test_label, idxs):
        ft   = all_results[f"finetuned_{test_label}"]
        base = all_results[f"baseline_{test_label}"]
        meta = test_sets[test_label]["meta"]
        ents = test_sets[test_label]["entries"]
        print(f"\n========== {test_label.upper()} SAMPLES ==========")
        for i in idxs:
            if i >= len(ents): continue
            m = (meta[i] if meta else {}) or {}
            tag = m.get("cohort", "") or m.get("utt_id", "")
            print(f"\n[{i}] {tag}")
            print(f"  REF:        {strip_prefix(ft['refs_raw'][i])[:160]}")
            print(f"  BASELINE:   {strip_prefix(base['hyps'][i])[:160]}")
            print(f"  FINE-TUNED: {strip_prefix(ft['hyps'][i])[:160]}")
    print_samples("openslr", [0, 100, 500, 1500, 3000])
    print_samples("hiacc",   [0, 100, 500, 800, 1000])

    # ---- 7. Write eval.json + predictions ----
    summary = {
        "checkpoint":            best_ckpt.name,
        "checkpoint_eval_loss":  best_loss,
        "all_checkpoints":       [c.name for c in ckpts],
        "eval_history":          eval_history,
        "in_domain_openslr": {
            "finetuned": all_results["finetuned_openslr"]["metrics"],
            "baseline":  all_results["baseline_openslr"]["metrics"],
            "wer_reduction_pp":
                all_results["baseline_openslr"]["metrics"]["wer_overall"]
                - all_results["finetuned_openslr"]["metrics"]["wer_overall"],
        },
        "cross_domain_hiacc": {
            "finetuned": all_results["finetuned_hiacc"]["metrics"],
            "baseline":  all_results["baseline_hiacc"]["metrics"],
            "wer_reduction_pp":
                all_results["baseline_hiacc"]["metrics"]["wer_overall"]
                - all_results["finetuned_hiacc"]["metrics"]["wer_overall"],
        },
    }
    (out_dir / "eval.json").write_text(json.dumps(summary, indent=2))

    # Per-test prediction files
    for test_label in ("openslr", "hiacc"):
        ents = test_sets[test_label]["entries"]
        meta = test_sets[test_label]["meta"] or [{}] * len(ents)
        ft   = all_results[f"finetuned_{test_label}"]
        base = all_results[f"baseline_{test_label}"]
        with open(out_dir / f"predictions_{test_label}.jsonl", "w", encoding="utf-8") as f:
            for i, ex in enumerate(ents):
                f.write(json.dumps({
                    "audio":     ex["audio"],
                    "meta":      meta[i],
                    "ref":       strip_prefix(ft["refs_raw"][i]),
                    "baseline":  strip_prefix(base["hyps"][i]),
                    "finetuned": strip_prefix(ft["hyps"][i]),
                }, ensure_ascii=False) + "\n")

    vol_ckpt.commit()
    print("\n========== FINAL ==========")
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def evaluate(checkpoint: str = ""):
    eval_v1_hiacc.remote(checkpoint=checkpoint)


@app.local_entrypoint()
def evaluate_v2(checkpoint: str = "", batch_size: int = 32):
    eval_v2_openslr.remote(checkpoint=checkpoint, batch_size=batch_size)


@app.local_entrypoint()
def evaluate_v3(checkpoint: str = "", batch_size: int = 32):
    # Same eval function — just point at v3's output dir.
    eval_v2_openslr.remote(
        output_dir="/ckpt/v3-union-h100x2",
        checkpoint=checkpoint,
        batch_size=batch_size,
    )


@app.local_entrypoint()
def push(checkpoint: str = "", private: bool = False):
    push_v1_to_hub.remote(checkpoint=checkpoint, private=private)


@app.function(
    image=train_image,
    volumes={"/ckpt": vol_ckpt, "/hf_cache": vol_hf_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=2 * 3600,
)
def push_v2_to_hub(
    output_dir: str = "/ckpt/v2-openslr-h100x2",
    checkpoint: str = "",                                              # "" = best by eval_loss
    repo_id: str = "Surajgameramp/qwen3-asr-0.6b-hinglish-openslr104-v2",
    private: bool = False,
):
    """Upload the best v2 (OpenSLR-104) checkpoint + model card to Hugging Face Hub."""
    import os, json, pathlib, shutil
    from huggingface_hub import HfApi, snapshot_download

    out_dir = pathlib.Path(output_dir)
    if not checkpoint:
        eval_path = out_dir / "eval.json"
        if not eval_path.exists():
            raise RuntimeError(f"no eval.json at {eval_path}; pass --checkpoint")
        with open(eval_path) as f:
            checkpoint = json.load(f)["checkpoint"]
    ckpt_dir = out_dir / checkpoint
    if not ckpt_dir.exists():
        raise RuntimeError(f"checkpoint dir not found: {ckpt_dir}")
    print(f"[push] checkpoint={ckpt_dir}")

    # Same upstream-bug repair as v1: copy preprocessor/processor/chat-template
    # from the base snapshot if missing in checkpoint.
    required = [
        "preprocessor_config.json", "processor_config.json", "chat_template.json",
        "config.json", "generation_config.json",
        "tokenizer_config.json", "tokenizer.json",
        "special_tokens_map.json", "added_tokens.json",
        "merges.txt", "vocab.json",
    ]
    missing = [f for f in required if not (ckpt_dir / f).exists()]
    if missing:
        base_dir = pathlib.Path(snapshot_download("Qwen/Qwen3-ASR-0.6B"))
        for f in missing:
            src = base_dir / f
            if src.exists():
                shutil.copy2(src, ckpt_dir / f)
                print(f"[ckpt-fix] copied {f}")
            else:
                print(f"[ckpt-fix] WARN '{f}' not in base snapshot, skipping")

    # Model card as README.md; training curves into figures/
    shutil.copy2("/opt/hf_model_card_v2.md", ckpt_dir / "README.md")
    figs_dir = ckpt_dir / "figures"
    figs_dir.mkdir(exist_ok=True)
    shutil.copy2("/opt/v2_training_curves.png", figs_dir / "training_curves.png")

    # Drop eval.json + predictions alongside for transparency
    for f in ("eval.json", "predictions_openslr.jsonl", "predictions_hiacc.jsonl"):
        src = out_dir / f
        if src.exists():
            shutil.copy2(src, ckpt_dir / f)
            print(f"[copy] {f} alongside weights")

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN missing in environment (Modal secret 'huggingface' empty?)")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(ckpt_dir),
        repo_id=repo_id,
        ignore_patterns=[
            "optimizer.pt", "scheduler.pt", "rng_state_*.pth", "training_args.bin",
        ],
        commit_message=f"Upload v2 (OpenSLR-104) fine-tune from {checkpoint}",
    )
    print(f"[push] done → https://huggingface.co/{repo_id}")


@app.local_entrypoint()
def push_v2(checkpoint: str = "", private: bool = False):
    push_v2_to_hub.remote(checkpoint=checkpoint, private=private)


@app.function(
    image=train_image,
    volumes={"/ckpt": vol_ckpt, "/hf_cache": vol_hf_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=2 * 3600,
)
def push_v3_to_hub(
    output_dir: str = "/ckpt/v3-union-h100x2",
    checkpoint: str = "",                                              # "" = best by eval_loss
    repo_id: str = "Surajgameramp/qwen3-asr-0.6b-hinglish-union-v3",
    private: bool = False,
):
    """Upload the best v3 (union) checkpoint + model card to Hugging Face Hub."""
    import os, json, pathlib, shutil
    from huggingface_hub import HfApi, snapshot_download

    out_dir = pathlib.Path(output_dir)
    if not checkpoint:
        eval_path = out_dir / "eval.json"
        if not eval_path.exists():
            raise RuntimeError(f"no eval.json at {eval_path}; pass --checkpoint")
        with open(eval_path) as f:
            checkpoint = json.load(f)["checkpoint"]
    ckpt_dir = out_dir / checkpoint
    if not ckpt_dir.exists():
        raise RuntimeError(f"checkpoint dir not found: {ckpt_dir}")
    print(f"[push] checkpoint={ckpt_dir}")

    required = [
        "preprocessor_config.json", "processor_config.json", "chat_template.json",
        "config.json", "generation_config.json",
        "tokenizer_config.json", "tokenizer.json",
        "special_tokens_map.json", "added_tokens.json",
        "merges.txt", "vocab.json",
    ]
    missing = [f for f in required if not (ckpt_dir / f).exists()]
    if missing:
        base_dir = pathlib.Path(snapshot_download("Qwen/Qwen3-ASR-0.6B"))
        for f in missing:
            src = base_dir / f
            if src.exists():
                shutil.copy2(src, ckpt_dir / f)
                print(f"[ckpt-fix] copied {f}")
            else:
                print(f"[ckpt-fix] WARN '{f}' not in base snapshot, skipping")

    shutil.copy2("/opt/hf_model_card_v3.md", ckpt_dir / "README.md")
    figs_dir = ckpt_dir / "figures"
    figs_dir.mkdir(exist_ok=True)
    shutil.copy2("/opt/v3_training_curves.png", figs_dir / "training_curves.png")

    for f in ("eval.json", "predictions_openslr.jsonl", "predictions_hiacc.jsonl"):
        src = out_dir / f
        if src.exists():
            shutil.copy2(src, ckpt_dir / f)
            print(f"[copy] {f} alongside weights")

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN missing in environment (Modal secret 'huggingface' empty?)")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(ckpt_dir),
        repo_id=repo_id,
        ignore_patterns=[
            "optimizer.pt", "scheduler.pt", "rng_state_*.pth", "training_args.bin",
        ],
        commit_message=f"Upload v3 (union) fine-tune from {checkpoint}",
    )
    print(f"[push] done → https://huggingface.co/{repo_id}")


@app.local_entrypoint()
def push_v3(checkpoint: str = "", private: bool = False):
    push_v3_to_hub.remote(checkpoint=checkpoint, private=private)


@app.local_entrypoint()
def download():
    download_hiacc.remote()


@app.local_entrypoint()
def inspect():
    inspect_hiacc.remote()


@app.local_entrypoint()
def inspect_deep():
    inspect_hiacc_deep.remote()


@app.local_entrypoint()
def download_openslr():
    download_openslr104.remote()


@app.local_entrypoint()
def inspect_openslr():
    inspect_openslr104.remote()


@app.function(
    image=base_image,
    volumes={"/data": vol_openslr},
    timeout=3 * 3600,
    cpu=8,
)
def prepare_openslr104_jsonl():
    """Slice OpenSLR-104 long-form audio into per-utterance chunks and write JSONL.

    - Splits train into 95% train / 5% val by recording_id (seeded).
    - Uses official test set as v2 test.
    - Filters: 0.5s <= segment <= 30s.
    - Normalization: whitespace-only collapse (same as HiACC v1).
    """
    import json, pathlib, random, re
    import soundfile as sf

    SR = 16000
    MIN_S, MAX_S = 0.5, 30.0
    SEED = 42

    root      = pathlib.Path("/data/openslr104")
    out_dir   = pathlib.Path("/data/openslr104/jsonl")
    chunks_dir = pathlib.Path("/data/openslr104/chunks")
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    WS_RE = re.compile(r"\s+")
    def normalize(text: str) -> str:
        return WS_RE.sub(" ", text).strip()

    def load_kaldi_split(split_root: pathlib.Path):
        tdir = split_root / "transcripts"
        utts = {}
        # text: "utt_id <text>"
        for line in (tdir / "text").read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2: continue
            utts[parts[0]] = {"text": parts[1]}
        # segments: "utt_id rec_id start end"
        for line in (tdir / "segments").read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            parts = line.split()
            if len(parts) != 4: continue
            uid, rid, s, e = parts
            if uid in utts:
                utts[uid].update(rec_id=rid, start=float(s), end=float(e))
        # wav.scp: "rec_id <wav filename>"  — files live flat in split_root
        rec_to_wav = {}
        for line in (tdir / "wav.scp").read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2: continue
            rid, wav_rel = parts
            rec_to_wav[rid] = split_root / pathlib.Path(wav_rel).name
        for uid in list(utts):
            rid = utts[uid].get("rec_id")
            if rid and rid in rec_to_wav:
                utts[uid]["wav_path"] = rec_to_wav[rid]
        return utts

    def chunk_split(split_name: str, utts: dict):
        # Group by recording so we load each source WAV exactly once
        by_rec = {}
        for uid, info in utts.items():
            if "wav_path" not in info or "start" not in info:
                continue
            by_rec.setdefault(info["rec_id"], []).append((uid, info))

        split_chunks = chunks_dir / split_name
        split_chunks.mkdir(parents=True, exist_ok=True)

        lines = []
        n_ok = n_short = n_long = n_loadfail = 0
        recs_sorted = sorted(by_rec.keys())
        for ri, rec_id in enumerate(recs_sorted):
            items = by_rec[rec_id]
            wav_path = items[0][1]["wav_path"]
            try:
                audio, sr = sf.read(str(wav_path), dtype="int16", always_2d=False)
            except Exception as e:
                print(f"[skip-recording] {wav_path.name}: {e}")
                n_loadfail += len(items)
                continue
            if sr != SR:
                print(f"[skip-recording] {wav_path.name}: sr={sr} != {SR}")
                n_loadfail += len(items)
                continue
            for uid, info in items:
                dur = info["end"] - info["start"]
                if dur < MIN_S:
                    n_short += 1; continue
                if dur > MAX_S:
                    n_long += 1; continue
                s = int(info["start"] * SR)
                e = int(info["end"] * SR)
                chunk = audio[s:e]
                if len(chunk) < int(MIN_S * SR):
                    n_short += 1; continue
                chunk_path = split_chunks / f"{uid}.wav"
                sf.write(str(chunk_path), chunk, SR, subtype="PCM_16")
                lines.append({
                    "utt_id":     uid,
                    "rec_id":     info["rec_id"],
                    "audio_path": str(chunk_path),
                    "text":       normalize(info["text"]),
                })
                n_ok += 1
            if (ri + 1) % 50 == 0 or (ri + 1) == len(recs_sorted):
                print(f"[{split_name}] recording {ri+1}/{len(recs_sorted)}  "
                      f"chunks_written={n_ok}")
        print(f"[{split_name}] DONE total={n_ok}  short<{MIN_S}s={n_short}  "
              f"long>{MAX_S}s={n_long}  load_fail={n_loadfail}")
        return lines

    def write_jsonl(name: str, lines: list):
        path      = out_dir / f"{name}.jsonl"
        meta_path = out_dir / f"{name}.meta.jsonl"
        with open(path, "w", encoding="utf-8") as f, \
             open(meta_path, "w", encoding="utf-8") as fm:
            for d in lines:
                f.write(json.dumps({
                    "audio": d["audio_path"],
                    "text":  f"language None<asr_text>{d['text']}",
                }, ensure_ascii=False) + "\n")
                fm.write(json.dumps({
                    "audio":    d["audio_path"],
                    "utt_id":   d["utt_id"],
                    "rec_id":   d["rec_id"],
                    "ref_text": d["text"],
                }, ensure_ascii=False) + "\n")
        print(f"[write] {path}: {len(lines)} lines")

    # ---- 1. Train: chunk, then split into train/val by recording_id ----
    print("\n========== TRAIN (chunk) ==========")
    train_utts  = load_kaldi_split(root / "train" / "train")
    print(f"[train] loaded {len(train_utts)} utterances from transcripts")
    train_lines = chunk_split("train", train_utts)

    train_recs = sorted({d["rec_id"] for d in train_lines})
    rng = random.Random(SEED)
    rng.shuffle(train_recs)
    val_n   = max(1, int(0.05 * len(train_recs)))
    val_set = set(train_recs[:val_n])
    train_out = [d for d in train_lines if d["rec_id"] not in val_set]
    val_out   = [d for d in train_lines if d["rec_id"]     in val_set]
    print(f"[split] {len(train_recs)} recordings -> {len(train_recs)-val_n} train / {val_n} val")
    print(f"[split] {len(train_out)} train utts / {len(val_out)} val utts")

    # ---- 2. Test: official MUCS-2021 test set ----
    print("\n========== TEST (chunk) ==========")
    test_utts  = load_kaldi_split(root / "test" / "test")
    print(f"[test] loaded {len(test_utts)} utterances from transcripts")
    test_lines = chunk_split("test", test_utts)

    # ---- 3. Write JSONLs ----
    write_jsonl("train", train_out)
    write_jsonl("val",   val_out)
    write_jsonl("test",  test_lines)

    # ---- 4. Eyeball samples ----
    for name, lines in [("train", train_out), ("val", val_out), ("test", test_lines)]:
        print(f"\n========== {name}.jsonl (first 3 lines) ==========")
        for d in lines[:3]:
            t = d["text"]
            t = t if len(t) <= 160 else t[:160] + "..."
            print(json.dumps({
                "audio": d["audio_path"],
                "text":  f"language None<asr_text>{t}",
            }, ensure_ascii=False))

    print(f"\n========== TOTALS ==========")
    print(f"  train: {len(train_out)} utterances")
    print(f"  val:   {len(val_out)} utterances")
    print(f"  test:  {len(test_lines)} utterances")

    vol_openslr.commit()


@app.local_entrypoint()
def prepare_openslr_jsonl():
    prepare_openslr104_jsonl.remote()


@app.function(
    image=base_image,
    volumes={"/data": vol_openslr},
    timeout=600,
)
def respit_openslr_by_speaker(val_ratio: float = 0.05, seed: int = 42):
    """Re-split train/val on the OpenSLR-104 JSONLs so that NO speaker appears in both.

    Reads existing train + val meta-sidecars (combined = the original train pool),
    looks up each utt's speaker in train/train/transcripts/utt2spk, then partitions
    by speaker (not by recording). Rewrites train.jsonl + val.jsonl + sidecars.

    Also audits transcript characters (case, punctuation) and prints stats.
    """
    import json, pathlib, random, collections

    root      = pathlib.Path("/data/openslr104")
    jsonl_dir = pathlib.Path("/data/openslr104/jsonl")
    utt2spk_path = root / "train" / "train" / "transcripts" / "utt2spk"

    # ---- Load existing train + val meta (combined = full train pool)
    pool = []
    for name in ["train", "val"]:
        with open(jsonl_dir / f"{name}.meta.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pool.append(json.loads(line))
    with open(jsonl_dir / "train.jsonl", encoding="utf-8") as f:
        train_lines_now = [json.loads(l) for l in f if l.strip()]
    with open(jsonl_dir / "val.jsonl", encoding="utf-8") as f:
        val_lines_now   = [json.loads(l) for l in f if l.strip()]
    pool_full = []
    # Re-pair meta with the actual jsonl line (same audio key)
    audio_to_text = {d["audio"]: d["text"] for d in train_lines_now + val_lines_now}
    for m in pool:
        text = audio_to_text.get(m["audio"])
        if text is None:
            continue
        pool_full.append({
            "audio":   m["audio"],
            "text":    text,
            "utt_id":  m["utt_id"],
            "rec_id":  m["rec_id"],
            "ref":     m["ref_text"],
        })
    print(f"[pool] {len(pool_full)} utterances (combined train+val)")

    # ---- Load utt2spk
    utt2spk = {}
    for line in utt2spk_path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        parts = line.split()
        if len(parts) >= 2:
            utt2spk[parts[0]] = parts[1]
    print(f"[utt2spk] {len(utt2spk)} mappings")

    # ---- Tag each utt with speaker
    n_missing = 0
    for d in pool_full:
        sp = utt2spk.get(d["utt_id"])
        if sp is None:
            n_missing += 1
            sp = "__unknown__"
        d["speaker"] = sp
    if n_missing:
        print(f"[warn] {n_missing} utts had no speaker in utt2spk")

    # ---- Audit: how many speakers were leaked by the old rec_id-based split?
    rec_to_speakers = collections.defaultdict(set)
    for d in pool_full:
        rec_to_speakers[d["rec_id"]].add(d["speaker"])
    multi_speaker_recs = sum(1 for s in rec_to_speakers.values() if len(s) > 1)
    print(f"[audit] {multi_speaker_recs} recordings had >1 speaker "
          f"(out of {len(rec_to_speakers)}); avg speakers/recording = "
          f"{sum(len(s) for s in rec_to_speakers.values())/len(rec_to_speakers):.2f}")
    # The "leak" in the old split: count speakers that appear in both old-train and old-val
    train_utt_ids = {pathlib.Path(d["audio"]).stem.split("_", 1)[1]  # unreliable; use audio path
                     for d in train_lines_now}
    train_speakers_old = {utt2spk.get(m["utt_id"], "__unk__")
                          for m in pool if m["audio"] in {x["audio"] for x in train_lines_now}}
    val_speakers_old   = {utt2spk.get(m["utt_id"], "__unk__")
                          for m in pool if m["audio"] in {x["audio"] for x in val_lines_now}}
    leaked = train_speakers_old & val_speakers_old
    print(f"[audit] old split leaked {len(leaked)} speakers across train↔val "
          f"(train_speakers={len(train_speakers_old)}, val_speakers={len(val_speakers_old)})")

    # ---- Speaker-disjoint resplit
    all_speakers = sorted({d["speaker"] for d in pool_full})
    rng = random.Random(seed)
    rng.shuffle(all_speakers)
    n_val = max(1, int(val_ratio * len(all_speakers)))
    val_speakers = set(all_speakers[:n_val])
    print(f"[split] {len(all_speakers)} unique speakers → "
          f"{len(all_speakers)-n_val} train / {n_val} val")

    train_out, val_out = [], []
    for d in pool_full:
        (val_out if d["speaker"] in val_speakers else train_out).append(d)
    print(f"[split] {len(train_out)} train utts / {len(val_out)} val utts")

    # ---- Text audit (case, punctuation)
    import re
    PUNCT = collections.Counter()
    upper_lower = [0, 0]
    for d in pool_full:
        for ch in d["ref"]:
            if 0x0041 <= ord(ch) <= 0x005A: upper_lower[0] += 1
            elif 0x0061 <= ord(ch) <= 0x007A: upper_lower[1] += 1
            elif (0x0900 <= ord(ch) <= 0x097F) or ch.isspace() or ch.isdigit():
                continue
            else:
                PUNCT[ch] += 1
    print(f"[text-audit] latin upper={upper_lower[0]}  lower={upper_lower[1]}  "
          f"upper-ratio={upper_lower[0]/(sum(upper_lower) or 1):.4f}")
    print(f"[text-audit] non-alnum chars (top 10): {PUNCT.most_common(10)}")

    # ---- Write JSONLs (overwrite previous train/val)
    def write_jsonl(name, items):
        with open(jsonl_dir / f"{name}.jsonl", "w", encoding="utf-8") as f, \
             open(jsonl_dir / f"{name}.meta.jsonl", "w", encoding="utf-8") as fm:
            for d in items:
                f.write(json.dumps({"audio": d["audio"], "text": d["text"]},
                                   ensure_ascii=False) + "\n")
                fm.write(json.dumps({
                    "audio": d["audio"], "utt_id": d["utt_id"],
                    "rec_id": d["rec_id"], "speaker": d["speaker"],
                    "ref_text": d["ref"],
                }, ensure_ascii=False) + "\n")
        print(f"[write] {jsonl_dir / (name + '.jsonl')}: {len(items)} lines")

    write_jsonl("train", train_out)
    write_jsonl("val",   val_out)

    vol_openslr.commit()


@app.local_entrypoint()
def respit_openslr():
    respit_openslr_by_speaker.remote()


@app.function(
    image=base_image,
    volumes={
        "/data_hiacc":   vol_data,
        "/data_openslr": vol_openslr,
    },
    timeout=600,
)
def prepare_union_jsonl():
    """Concatenate HiACC + OpenSLR-104 train/val JSONLs into a union for v3.

    - Reads each source's JSONL (paths like /data/hiacc/... and /data/openslr104/...)
    - Remaps to dual-mount paths (/data_hiacc/..., /data_openslr/...)
    - Concatenates + deterministically shuffles (seed=42)
    - Writes train_union.jsonl + val_union.jsonl into /data_openslr/openslr104/jsonl/
    - NO balanced upsampling for v3 (HiACC is ~7% of union); revisit if HiACC WER regresses
    """
    import json, pathlib, random

    out_dir = pathlib.Path("/data_openslr/openslr104/jsonl")

    PATH_REMAP = [
        ("/data/hiacc/",      "/data_hiacc/hiacc/"),
        ("/data/openslr104/", "/data_openslr/openslr104/"),
    ]
    def remap(p):
        for a, b in PATH_REMAP:
            if p.startswith(a):
                return p.replace(a, b, 1)
        return p

    def load_remapped(path):
        entries = []
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            d["audio"] = remap(d["audio"])
            entries.append(d)
        return entries

    train_hi = load_remapped("/data_hiacc/hiacc/jsonl/train.jsonl")
    val_hi   = load_remapped("/data_hiacc/hiacc/jsonl/val.jsonl")
    train_os = load_remapped("/data_openslr/openslr104/jsonl/train.jsonl")
    val_os   = load_remapped("/data_openslr/openslr104/jsonl/val.jsonl")

    train_union = train_hi + train_os
    val_union   = val_hi + val_os

    rng = random.Random(42)
    rng.shuffle(train_union)
    rng.shuffle(val_union)

    print(f"[union-train] {len(train_union)} = {len(train_hi)} HiACC ({100*len(train_hi)/len(train_union):.1f}%) "
          f"+ {len(train_os)} OpenSLR ({100*len(train_os)/len(train_union):.1f}%)")
    print(f"[union-val]   {len(val_union)} = {len(val_hi)} HiACC + {len(val_os)} OpenSLR")

    for name, entries in [("train_union", train_union), ("val_union", val_union)]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for d in entries:
                f.write(json.dumps({"audio": d["audio"], "text": d["text"]},
                                   ensure_ascii=False) + "\n")
        print(f"[write] {path}: {len(entries)} lines")

    print(f"\n========== train_union.jsonl (first 3) ==========")
    for d in train_union[:3]:
        t = d["text"][:140] + ("..." if len(d["text"]) > 140 else "")
        print(json.dumps({"audio": d["audio"], "text": t}, ensure_ascii=False))

    vol_openslr.commit()


@app.local_entrypoint()
def prepare_union():
    prepare_union_jsonl.remote()
