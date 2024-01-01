# Install Coqui TTS and g2p for Irish phonemizer support
!pip install -U pip
!pip install "git+https://github.com/seocahill/g2p.git@a5d318c158ab376a403f9af94f50aeee4adda68f#egg=g2p"
!pip install "git+https://github.com/seocahill/TTS.git@7bb173d53824b89e0b40ae4e9716368c463d6174#egg=TTS"

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os

def main():
    # BaseDatasetConfig: defines name, formatter and path of the dataset.
    from TTS.tts.configs.shared_configs import BaseDatasetConfig

    # Update output_path to a directory in Google Drive
    output_path = "/content/drive/My Drive/tts_train_dir"
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Paths for dataset and metadata
    dataset_path = f"{output_path}/ga.ie.cll.48000.tar.gz"
    metadata_path = f"{output_path}/ga_ie_cll_48000/metadata.xml"

    # Download and extract Irish dataset if not already done
    if not os.path.exists(dataset_path):
        !wget -O "$dataset_path" https://archive.org/download/ga.ie.cll.48000.tar/ga.ie.cll.48000.tar.gz

    extracted_path = f"{output_path}/ga_ie_cll_48000"
    if not os.path.exists(extracted_path):
        !mkdir -p "$extracted_path/wavs"
        !tar -xzvf "$dataset_path" -C "$extracted_path"
        # Dir format per LJSpeech
        !mv "$extracted_path/48000_orig/*" "$extracted_path/wavs"

    # Download and format metadata if not already done
    if not os.path.exists(metadata_path):
        !wget -O "$metadata_path" https://raw.githubusercontent.com/Idlak/Living-Audio-Dataset/master/ga/text.xml

    import xml.etree.ElementTree as ET
    import csv

    # Update file paths for the Google Drive directory
    xml_file_path = f"{output_path}/ga_ie_cll_48000/metadata.xml"
    csv_file_path = f"{output_path}/ga_ie_cll_48000/metadata.csv"

    # Parse the XML file and create CSV
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for fileid in root.findall('fileid'):
            id_attr = fileid.get('id')
            text = fileid.text.strip()
            writer.writerow([id_attr, text, text])

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", meta_file_train="metadata.csv", path=os.path.join(output_path, "ga_ie_cll_48000/")
    )

    # GlowTTSConfig: all model related values for training, validating and testing.
    from TTS.tts.configs.glow_tts_config import GlowTTSConfig
    config = GlowTTSConfig(
        batch_size=32,
        eval_batch_size=16,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=50,
        use_phonemes=True,
        phoneme_language="ga",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=25,
        print_eval=False,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        save_step=1000,
    )

    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    tokenizer, config = TTSTokenizer.init_from_config(config)

    from TTS.utils.audio import AudioProcessor
    ap = AudioProcessor.init_from_config(config)
    # Modify sample rate if for a custom audio dataset:
    # ap.sample_rate = 22050


    from TTS.tts.datasets import load_tts_samples
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    from TTS.tts.models.glow_tts import GlowTTS
    model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

    # Path to the TensorBoard log directory
    log_dir = f"{output_path}/logs"

    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Start TensorBoard to monitor the training logs
    # %tensorboard --logdir $log_dir

    from trainer import Trainer, TrainerArgs
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )

    trainer.fit()

if __name__ == '__main__':
    main()
    # !tensorboard --logdir=tts_train_dir
