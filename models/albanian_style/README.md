# Albanian Style Model

This project is dedicated to training the **Gemma 4:e4b** model on the stylistic elements of various Albanian text registers. Unlike the `albanian_analysis` model, which focuses on structural and grammatical analysis, this model is trained specifically for stylistic imitation and generation.

## Project Structure

- `data/seed/`: Contains the original Albanian articles used for style training, categorized by register (editorial, informational, marketing, classical, mixed).
- `notebooks/`: Contains the end-to-end training pipeline.
  - `style_training.ipynb`: A self-contained Google Colab notebook for fine-tuning the model using Unsloth.
- `training/`: (Created during training) Contains model checkpoints and final LoRA adapters.

## Training Pipeline

The training process is automated via the `notebooks/style_training.ipynb` notebook. It performs the following steps:

1. **Environment Setup**: Installs Unsloth and necessary dependencies for efficient 4-bit fine-tuning.
2. **Data Loading**: Reads raw `.txt` files from `data/seed/` and categorizes them.
3. **Prompt Construction**: Formats the data into the standard Lahuta turn-based format:
   ```
   <start_of_turn>system
   Je një redaktor ekspert i gjuhës shqipe që specializohet në stilin {register}.<end_of_turn>
   <start_of_turn>user
   Shkruaj një artikull shembull në stilin {register}.<end_of_turn>
   <start_of_turn>model
   {content}<end_of_turn>
   ```
4. **LoRA Fine-Tuning**: Applies Low-Rank Adaptation to the Gemma 4 base model, targeting attention projections for stylistic adaptation.
5. **Model Export**: Saves the fine-tuned adapters for use in the Lahuta ecosystem.

## Usage

1. Upload the `albanian_style` folder to your Google Drive or Colab environment.
2. Ensure the `data/seed/` directory is populated with the articles.
3. Open and run `notebooks/style_training.ipynb`.
