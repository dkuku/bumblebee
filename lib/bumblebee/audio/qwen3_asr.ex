defmodule Bumblebee.Audio.Qwen3ASR do
  alias Bumblebee.Shared
  alias Bumblebee.Layers

  options = [
    audio_config: [default: nil, doc: "the Qwen3-ASR audio encoder configuration"],
    text_config: [default: nil, doc: "the Qwen3 text decoder configuration"],
    audio_token_id: [default: 151_676, doc: "the token used as an audio placeholder"],
    vocab_size: [default: 151_936, doc: "the text vocabulary size"],
    hidden_size: [default: 1024, doc: "the audio encoder hidden size"],
    audio_output_dim: [default: 1024, doc: "the projected audio embedding size"],
    audio_num_mel_bins: [default: 128, doc: "the number of input Mel bins"],
    audio_num_chunks: [default: 30, doc: "the number of one-second audio chunks"],
    audio_chunk_frames: [default: 100, doc: "the number of Mel frames in one chunk"],
    audio_downsample_hidden_size: [default: 480, doc: "the convolutional audio hidden size"],
    audio_num_blocks: [default: 18, doc: "the number of audio Transformer blocks"],
    audio_num_attention_heads: [default: 14, doc: "the number of audio attention heads"],
    audio_intermediate_size: [default: 3584, doc: "the audio feed-forward size"],
    audio_attention_window: [default: {50, 50}, doc: "the local audio attention window"],
    initializer_scale: [default: 0.02, doc: "the parameter initializer scale"]
  ]

  @moduledoc """
  Qwen3-ASR: an audio encoder and projector followed by a Qwen3 decoder.

  The model expects log-Mel features in `{batch, 128, 3000}` and text token
  ids. Audio placeholder tokens in the text prompt are replaced by the
  projected audio embeddings, matching the Qwen3-ASR generation path.
  """

  defstruct [architecture: :for_conditional_generation] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable
  @behaviour Bumblebee.Text.Generation

  @impl true
  def architectures, do: [:for_conditional_generation]

  @impl Bumblebee.Text.Generation
  def init_cache(spec, batch_size, max_length, _inputs) do
    Bumblebee.Text.Qwen3.init_cache(spec.text_config, batch_size, max_length, %{})
  end

  @impl Bumblebee.Text.Generation
  def traverse_cache(spec, cache, fun) do
    Bumblebee.Text.Qwen3.traverse_cache(spec.text_config, cache, fun)
  end

  @impl true
  def config(spec, opts), do: Shared.put_config_attrs(spec, opts)

  @impl true
  def input_template(spec) do
    %{
      "input_features" =>
        Nx.template(
          {1, spec.audio_num_mel_bins, spec.audio_num_chunks * spec.audio_chunk_frames},
          :f32
        ),
      "input_features_mask" =>
        Nx.template({1, spec.audio_num_chunks * spec.audio_chunk_frames}, :u32),
      "input_ids" => Nx.template({1, 1}, :s64),
      "attention_mask" => Nx.template({1, 1}, :u32)
    }
  end

  @impl true
  def model(%__MODULE__{} = spec) do
    inputs = inputs(spec)
    audio_embeddings = audio_embeddings(inputs["input_features"], spec)

    text_spec = spec.text_config

    text_embeddings =
      Axon.embedding(inputs["input_ids"], text_spec.vocab_size, text_spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: "language_model.embedder.token_embedding"
      )

    embeddings =
      replace_audio_embeddings(text_embeddings, audio_embeddings, inputs["input_ids"], spec)

    attention_mask = inputs["attention_mask"]

    language_inputs = %{
      "input_ids" => inputs["input_ids"],
      "input_embeddings" => embeddings,
      "attention_mask" => attention_mask,
      "position_ids" => inputs["position_ids"],
      "attention_head_mask" => inputs["attention_head_mask"],
      "cache" => inputs["cache"]
    }

    Bumblebee.Text.Qwen3.model_from_inputs(text_spec, language_inputs,
      name_prefix: "language_model"
    )
  end

  defp replace_audio_embeddings(text_embeddings, audio_embeddings, input_ids, spec) do
    Axon.layer(
      fn text, audio, ids, _opts ->
        audio_mask = Nx.equal(ids, spec.audio_token_id)
        audio_index = Nx.subtract(Nx.cumulative_sum(audio_mask, axis: 1), 1)
        audio_positions = Nx.iota({Nx.axis_size(ids, 1), spec.audio_num_chunks * 13}, axis: 0)

        selector =
          Nx.equal(
            Nx.new_axis(audio_index, 2),
            Nx.reshape(audio_positions, {1, Nx.axis_size(ids, 1), spec.audio_num_chunks * 13})
          )

        audio_at_positions =
          Nx.sum(
            Nx.multiply(Nx.as_type(Nx.new_axis(selector, 3), :f32), Nx.new_axis(audio, 1)),
            axes: [2]
          )

        mask = Nx.broadcast(Nx.new_axis(audio_mask, 2), Nx.shape(text))
        Nx.select(mask, audio_at_positions, text)
      end,
      [text_embeddings, audio_embeddings, input_ids],
      name: "language_model.audio_embedding_injector"
    )
  end

  defp inputs(spec) do
    audio_shape = {nil, spec.audio_num_mel_bins, spec.audio_num_chunks * spec.audio_chunk_frames}
    text_shape = {nil, nil}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_features", shape: audio_shape),
      Axon.input("input_features_mask",
        shape: {nil, spec.audio_num_chunks * spec.audio_chunk_frames}
      ),
      Axon.input("input_ids", shape: text_shape),
      Axon.input("attention_mask", shape: text_shape, optional: true),
      Axon.input("position_ids", shape: text_shape, optional: true),
      Axon.input("attention_head_mask",
        shape: {spec.text_config.num_blocks, spec.text_config.num_attention_heads},
        optional: true
      ),
      Axon.input("cache", optional: true)
    ])
  end

  defp audio_embeddings(input_features, spec) do
    chunks =
      Axon.layer(
        fn input, _opts ->
          batch = Nx.axis_size(input, 0)

          Nx.reshape(
            input,
            {batch * spec.audio_num_chunks, spec.audio_num_mel_bins, spec.audio_chunk_frames, 1}
          )
        end,
        [input_features],
        name: "audio_tower.chunk"
      )

    hidden =
      chunks
      |> Axon.conv(spec.audio_downsample_hidden_size,
        kernel_size: {3, 3},
        strides: [2, 2],
        padding: :same,
        kernel_initializer: kernel_initializer(spec),
        name: "audio_tower.conv2d1"
      )
      |> Axon.gelu()
      |> Axon.conv(spec.audio_downsample_hidden_size,
        kernel_size: {3, 3},
        strides: [2, 2],
        padding: :same,
        kernel_initializer: kernel_initializer(spec),
        name: "audio_tower.conv2d2"
      )
      |> Axon.gelu()
      |> Axon.conv(spec.audio_downsample_hidden_size,
        kernel_size: {3, 3},
        strides: [2, 2],
        padding: :same,
        kernel_initializer: kernel_initializer(spec),
        name: "audio_tower.conv2d3"
      )
      |> Axon.gelu()
      |> then(fn input ->
        Axon.layer(
          fn input, _opts -> Nx.transpose(input, axes: [0, 2, 1, 3]) end,
          [input],
          name: "audio_tower.transpose"
        )
      end)
      |> Axon.reshape({:batch, :auto, spec.audio_downsample_hidden_size * 16},
        name: "audio_tower.flatten"
      )
      |> Axon.dense(spec.hidden_size,
        use_bias: false,
        kernel_initializer: kernel_initializer(spec),
        name: "audio_tower.conv_out"
      )
      |> then(fn input ->
        Axon.layer(
          fn input, _opts ->
            chunks = Nx.axis_size(input, 0)
            batch = div(chunks, spec.audio_num_chunks)
            Nx.reshape(input, {batch, spec.audio_num_chunks * 13, spec.hidden_size})
          end,
          [input],
          name: "audio_tower.sequence"
        )
      end)

    positions = sinusoidal_positions(13, spec.hidden_size)

    hidden =
      Axon.layer(
        fn hidden, positions, _opts ->
          batch = Nx.axis_size(hidden, 0)
          positions = Nx.broadcast(positions, {batch, 13, spec.hidden_size})
          positions = Nx.tile(positions, [1, spec.audio_num_chunks, 1])
          Nx.add(hidden, positions)
        end,
        [hidden, positions],
        name: "audio_tower.position_embedding"
      )

    outputs =
      Layers.Transformer.blocks(hidden,
        num_blocks: spec.audio_num_blocks,
        num_attention_heads: spec.audio_num_attention_heads,
        hidden_size: spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        dropout_rate: 0.0,
        attention_dropout_rate: 0.0,
        attention_window_size: spec.audio_attention_window,
        block_type: :norm_first,
        layer_norm: [epsilon: 1.0e-5],
        ffn: [intermediate_size: spec.audio_intermediate_size, activation: :gelu],
        name: "audio_tower.layers"
      )

    hidden = Axon.layer_norm(outputs.hidden_state, name: "audio_tower.ln_post")

    hidden
    |> Axon.dense(spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: "multi_modal_projector.linear_1"
    )
    |> Axon.gelu()
    |> Axon.dense(spec.audio_output_dim,
      kernel_initializer: kernel_initializer(spec),
      name: "multi_modal_projector.linear_2"
    )
  end

  defp sinusoidal_positions(length, channels) do
    log_increment = :math.log(10_000) / (channels / 2 - 1)
    timescales = for i <- 0..(div(channels, 2) - 1), do: :math.exp(-log_increment * i)

    values =
      for position <- 0..(length - 1), scale <- timescales do
        :math.sin(position * scale)
      end ++
        for position <- 0..(length - 1), scale <- timescales do
          :math.cos(position * scale)
        end

    Axon.layer(
      fn _opts -> Nx.tensor(values, type: :f32) |> Nx.reshape({length, channels}) end,
      [],
      name: "audio_tower.positional_embedding"
    )
  end

  defp kernel_initializer(spec), do: Axon.Initializers.normal(scale: spec.initializer_scale)

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      audio = data["audio_config"] || %{}

      text =
        (data["text_config"] || %{})
        |> Enum.reject(fn {_key, value} -> is_nil(value) end)
        |> Map.new()

      text_spec =
        Bumblebee.HuggingFace.Transformers.Config.load(
          Bumblebee.configure(Bumblebee.Text.Qwen3, architecture: :for_causal_language_modeling),
          text
        )

      opts = [
        audio_config: audio,
        text_config: text_spec,
        audio_token_id: data["audio_token_id"],
        vocab_size: text_spec.vocab_size,
        hidden_size: audio["d_model"],
        audio_output_dim: audio["output_dim"],
        audio_num_mel_bins: audio["num_mel_bins"],
        audio_num_chunks: 30,
        audio_chunk_frames: 100,
        audio_downsample_hidden_size: audio["downsample_hidden_size"],
        audio_num_blocks: audio["encoder_layers"],
        audio_num_attention_heads: audio["encoder_attention_heads"],
        audio_intermediate_size: audio["encoder_ffn_dim"],
        audio_attention_window: {audio["n_window"] || 50, audio["n_window"] || 50}
      ]

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(spec) do
      %{
        "audio_tower.conv2d1" => "model.audio_tower.conv2d1",
        "audio_tower.conv2d2" => "model.audio_tower.conv2d2",
        "audio_tower.conv2d3" => "model.audio_tower.conv2d3",
        "audio_tower.conv_out" => "model.audio_tower.conv_out",
        "audio_tower.layers.{n}.self_attention.query" =>
          "model.audio_tower.layers.{n}.self_attn.q_proj",
        "audio_tower.layers.{n}.self_attention.key" =>
          "model.audio_tower.layers.{n}.self_attn.k_proj",
        "audio_tower.layers.{n}.self_attention.value" =>
          "model.audio_tower.layers.{n}.self_attn.v_proj",
        "audio_tower.layers.{n}.self_attention.output" =>
          "model.audio_tower.layers.{n}.self_attn.out_proj",
        "audio_tower.layers.{n}.self_attention_norm" =>
          "model.audio_tower.layers.{n}.self_attn_layer_norm",
        "audio_tower.layers.{n}.ffn.intermediate" => "model.audio_tower.layers.{n}.fc1",
        "audio_tower.layers.{n}.ffn.output" => "model.audio_tower.layers.{n}.fc2",
        "audio_tower.layers.{n}.output_norm" => "model.audio_tower.layers.{n}.final_layer_norm",
        "audio_tower.ln_post" => "model.audio_tower.ln_post",
        "multi_modal_projector.linear_1" => "model.multi_modal_projector.linear_1",
        "multi_modal_projector.linear_2" => "model.multi_modal_projector.linear_2",
        "language_model.embedder.token_embedding" => "model.language_model.embed_tokens",
        "language_model.decoder.blocks.{n}.self_attention.query" =>
          "model.language_model.layers.{n}.self_attn.q_proj",
        "language_model.decoder.blocks.{n}.self_attention.key" =>
          "model.language_model.layers.{n}.self_attn.k_proj",
        "language_model.decoder.blocks.{n}.self_attention.value" =>
          "model.language_model.layers.{n}.self_attn.v_proj",
        "language_model.decoder.blocks.{n}.self_attention.output" =>
          "model.language_model.layers.{n}.self_attn.o_proj",
        "language_model.decoder.blocks.{n}.self_attention.query_norm" =>
          "model.language_model.layers.{n}.self_attn.q_norm",
        "language_model.decoder.blocks.{n}.self_attention.key_norm" =>
          "model.language_model.layers.{n}.self_attn.k_norm",
        "language_model.decoder.blocks.{n}.self_attention_norm" =>
          "model.language_model.layers.{n}.input_layernorm",
        "language_model.decoder.blocks.{n}.ffn.gate" =>
          "model.language_model.layers.{n}.mlp.gate_proj",
        "language_model.decoder.blocks.{n}.ffn.intermediate" =>
          "model.language_model.layers.{n}.mlp.up_proj",
        "language_model.decoder.blocks.{n}.ffn.output" =>
          "model.language_model.layers.{n}.mlp.down_proj",
        "language_model.decoder.blocks.{n}.output_norm" =>
          "model.language_model.layers.{n}.post_attention_layernorm",
        "language_model.output_norm" => "model.language_model.norm",
        "language_model.language_modeling_head.output" =>
          if(spec.text_config.tie_word_embeddings,
            do: "model.language_model.embed_tokens",
            else: "lm_head"
          )
      }
    end
  end
end
