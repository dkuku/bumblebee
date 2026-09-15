defmodule Bumblebee.Audio.Qwen3ASRTest do
  use ExUnit.Case, async: true

  test "builds logits with audio placeholders in the text sequence" do
    text_config =
      Bumblebee.configure(Bumblebee.Text.Qwen3,
        architecture: :for_causal_language_modeling,
        vocab_size: 32,
        max_positions: 128,
        hidden_size: 8,
        intermediate_size: 16,
        attention_head_size: 4,
        num_blocks: 1,
        num_attention_heads: 2,
        num_key_value_heads: 2
      )

    spec =
      Bumblebee.configure(Bumblebee.Audio.Qwen3ASR,
        text_config: text_config,
        audio_num_chunks: 1,
        audio_num_blocks: 1,
        audio_num_attention_heads: 2,
        audio_intermediate_size: 16,
        hidden_size: 8,
        audio_output_dim: 8,
        audio_downsample_hidden_size: 4,
        audio_chunk_frames: 100,
        audio_num_mel_bins: 128,
        audio_attention_window: {13, 13},
        vocab_size: 32,
        audio_token_id: 31
      )

    model = Bumblebee.Audio.Qwen3ASR.model(spec)

    inputs = %{
      "input_features" => Nx.broadcast(0.0, {1, 128, 100}),
      "input_features_mask" => Nx.broadcast(1, {1, 100}),
      "input_ids" => Nx.tensor([[1, 31, 31, 31, 2]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1]])
    }

    params = Axon.init(model, inputs)
    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 5, 32}
  end
end
