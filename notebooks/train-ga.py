## Install Coqui TTS and g2p for irish phonemizer support
!pip install -U pip
!pip install git+https://github.com/seocahill/g2p.git
!pip install git+https://github.com/seocahill/TTS.git

import os

def main():
    # BaseDatasetConfig: defines name, formatter and path of the dataset.
    from TTS.tts.configs.shared_configs import BaseDatasetConfig

    output_path = "tts_train_dir"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Download and extract irish dataset.
    !wget -O $output_path/ga.ie.cll.48000.tar.gz https://archive.org/download/ga.ie.cll.48000.tar/ga.ie.cll.48000.tar.gz
    !mkdir -p $output_path/ga_ie_cll_48000/wavs
    !tar -xzvf $output_path/ga.ie.cll.48000.tar.gz -C $output_path/ga_ie_cll_48000
    # Dir format per LJSpeech
    !mv $output_path/ga_ie_cll_48000/48000_orig/* $output_path/ga_ie_cll_48000/wavs

    # Download and format metadata
    import xml.etree.ElementTree as ET
    import csv

    # Path to your XML file
    !wget -O $output_path/ga.ie.cll.48000/metadata.xml https://github.com/Idlak/Living-Audio-Dataset/blob/master/ga/text.xml

    xml_file_path = "tts_train_dir/ga.ie.cll.48000/metadata.xml"

    # Path for the output CSV file
    csv_file_path = "tts_train_dir/ga.ie.cll.48000/metadata.csv"

    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Open the CSV file for writing
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Iterate through each fileid element in the XML
        for fileid in root.findall('fileid'):
            # Extract the id attribute and the text
            id_attr = fileid.get('id')
            text = fileid.text.strip()

            # Write to the CSV file
            writer.writerow([id_attr, text, text])

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", meta_file_train="metadata.csv", path=os.path.join(output_path, "ga_ie_cli_4800/")
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

    from trainer import Trainer, TrainerArgs
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )

    trainer.fit()

    !pip install tensorboard
    !tensorboard --logdir=tts_train_dir

if __name__ == '__main__':
    main()

