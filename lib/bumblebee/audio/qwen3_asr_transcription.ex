defmodule Bumblebee.Audio.Qwen3ASRTranscription do
  @moduledoc false

  alias Bumblebee.Shared

  @doc false
  def transcribe(model_info, featurizer, tokenizer, generation_config, input, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :chunk_num_seconds,
        :context_num_seconds,
        :max_new_tokens,
        :seed,
        progress: nil,
        defn_options: [compiler: EXLA]
      ])

    progress = Keyword.get(opts, :progress)

    if progress && not is_function(progress, 2) do
      raise ArgumentError, ":progress must be a function accepting chunk index and text"
    end

    %{model: model, params: params, spec: spec} = model_info
    max_new_tokens = Keyword.get(opts, :max_new_tokens, generation_config.max_new_tokens)

    generation_config = %{generation_config | max_new_tokens: max_new_tokens}
    defn_options = Keyword.get(opts, :defn_options, compiler: EXLA)

    samples = decode_audio(input, featurizer.sampling_rate)

    predict = elem(Axon.build(model), 1)

    predict =
      Shared.compile_or_jit(
        fn params, inputs -> predict.(params, inputs) end,
        :qwen3_asr_step,
        defn_options,
        false,
        fn -> [] end
      )

    chunk_num_seconds = Keyword.get(opts, :chunk_num_seconds)

    samples =
      if chunk_num_seconds do
        context_num_seconds =
          Keyword.get(opts, :context_num_seconds, chunk_num_seconds / 6)

        Bumblebee.Audio.chunk_audio(
          samples,
          featurizer.sampling_rate,
          chunk_num_seconds,
          context_num_seconds
        )
      else
        samples
      end

    samples
    |> Enum.map(fn samples ->
      transcribe_chunk(
        samples,
        featurizer,
        tokenizer,
        spec,
        predict,
        params,
        generation_config,
        max_new_tokens,
        Keyword.get(opts, :seed, 0)
      )
    end)
    |> Enum.with_index(1)
    |> Enum.map(fn {text, index} ->
      if progress, do: progress.(index, text)
      text
    end)
    |> Enum.map_join(" ", &String.trim/1)
    |> String.trim()
  end

  defp transcribe_chunk(
         samples,
         featurizer,
         tokenizer,
         spec,
         predict,
         params,
         generation_config,
         max_new_tokens,
         seed
       ) do
    features =
      samples
      |> then(&Bumblebee.Featurizer.process_input(featurizer, [&1]))
      |> then(&Bumblebee.Featurizer.process_batch(featurizer, &1))

    audio_tokens = spec.audio_num_chunks * 13
    ids = prompt_ids(tokenizer, audio_tokens)
    input_ids = Nx.tensor([ids], type: :s64)

    inputs =
      features
      |> Map.put("input_features_mask", Nx.broadcast(1, {1, spec.audio_num_chunks * 100}))
      |> Map.put("input_ids", input_ids)
      |> Map.put("attention_mask", Nx.broadcast(1, {1, length(ids)}))
      |> Map.put("seed", Nx.tensor([seed], type: :s64))

    generated_ids =
      greedy_decode(predict, params, spec, inputs, generation_config, max_new_tokens)

    Bumblebee.Tokenizer.decode(tokenizer, generated_ids)
  end

  defp greedy_decode(predict, params, spec, inputs, generation_config, max_new_tokens) do
    prompt_length = Nx.axis_size(inputs["input_ids"], 1)
    cache = Bumblebee.Text.Generation.init_cache(spec, 1, prompt_length + max_new_tokens, inputs)

    inputs =
      inputs
      |> Map.put("position_ids", position_ids(inputs["attention_mask"]))
      |> Map.put("cache", cache)

    {outputs, generated} =
      Enum.reduce_while(1..max_new_tokens, {predict.(params, inputs), []}, fn _step,
                                                                              {outputs, generated} ->
        token = next_token(outputs.logits)
        generated = generated ++ [token]

        if token in List.wrap(generation_config.eos_token_id) do
          {:halt, {outputs, generated}}
        else
          next_inputs = %{
            "input_features" => inputs["input_features"],
            "input_features_mask" => inputs["input_features_mask"],
            "input_ids" => Nx.tensor([[token]], type: :s64),
            "attention_mask" => Nx.tensor([[1]], type: :u32),
            "position_ids" => next_position_ids(inputs["position_ids"]),
            "cache" => outputs.cache
          }

          {:cont, {predict.(params, next_inputs), generated}}
        end
      end)

    _ = outputs
    generated
  end

  defp position_ids(attention_mask) do
    attention_mask
    |> Nx.cumulative_sum(axis: 1)
    |> Nx.subtract(Nx.select(attention_mask, 1, 0))
  end

  defp next_position_ids(position_ids) do
    position_ids
    |> Nx.slice_along_axis(Nx.axis_size(position_ids, 1) - 1, 1, axis: 1)
    |> Nx.add(1)
  end

  defp next_token(logits) do
    logits
    |> Nx.slice_along_axis(Nx.axis_size(logits, 1) - 1, 1, axis: 1)
    |> Nx.squeeze(axes: [1])
    |> Nx.argmax(axis: -1)
    |> Nx.backend_transfer(Nx.BinaryBackend)
    |> Nx.to_number()
  end

  defp prompt_ids(tokenizer, audio_tokens) do
    ids = fn token -> Bumblebee.Tokenizer.token_to_id(tokenizer, token) end

    text = fn value ->
      tokenizer = Bumblebee.configure(tokenizer, add_special_tokens: false)

      Bumblebee.apply_tokenizer(tokenizer, value)
      |> Map.fetch!("input_ids")
      |> Nx.to_list()
      |> List.flatten()
    end

    [
      ids.("<|im_start|>")
      | text.("system\n") ++
          [ids.("<|im_end|>"), ids.("<|im_start|>")] ++
          text.("user\n") ++
          [ids.("<|audio_start|>")] ++
          List.duplicate(ids.("<|audio_pad|>"), audio_tokens) ++
          [ids.("<|audio_end|>"), ids.("<|im_end|>"), ids.("<|im_start|>")] ++
          text.("assistant\n")
    ]
  end

  defp decode_audio({:file, path}, sampling_rate) do
    format = if System.endianness() == :little, do: "f32le", else: "f32be"

    case System.cmd(
           "ffmpeg",
           ~w[-i #{path} -ac 1 -ar #{sampling_rate} -f #{format} -hide_banner -loglevel quiet pipe:1]
         ) do
      {data, 0} -> [Nx.from_binary(data, :f32, backend: Nx.BinaryBackend)]
      {_, _} -> raise ArgumentError, "ffmpeg failed to decode #{path}"
    end
  end

  defp decode_audio(%Nx.Tensor{shape: {_}} = samples, _sampling_rate), do: [samples]
  defp decode_audio(samples, _sampling_rate) when is_list(samples), do: samples
end
