defmodule Bumblebee.Audio.Chunking do
  @moduledoc false

  @doc false
  def chunk(stream, sampling_rate, chunk_num_seconds, context_num_seconds) do
    chunk_length = floor(chunk_num_seconds * sampling_rate)
    context_left = floor(context_num_seconds * sampling_rate)
    context_right = context_left

    step = chunk_length - context_left - context_right

    if step <= 0 do
      raise ArgumentError,
            ":chunk_num_seconds must be more than double the length of :context_num_seconds"
    end

    Stream.transform(
      stream,
      fn -> {[], 0} end,
      fn chunk, {buffer, buffer_size} ->
        buffer_size = buffer_size + Nx.size(chunk)
        buffer = buffer ++ [chunk]
        full_chunks([], {buffer, buffer_size}, chunk_length, step)
      end,
      fn
        {[], buffer_size} ->
          {[], {[], buffer_size}}

        {buffer, buffer_size} ->
          {[Nx.concatenate(buffer)], {buffer, buffer_size}}
      end,
      fn _ -> :ok end
    )
  end

  defp full_chunks(acc, {buffer, buffer_size}, chunk_length, _step)
       when chunk_length > buffer_size do
    {Enum.reverse(acc), {buffer, buffer_size}}
  end

  defp full_chunks(acc, {buffer, buffer_size}, chunk_length, step) do
    {subchunks1, buffer} = slice_buffer(buffer, step, [])
    {subchunks2, _buffer} = slice_buffer(buffer, chunk_length - step, [])
    chunk = Nx.concatenate(subchunks1 ++ subchunks2)
    buffer_size = buffer_size - step
    full_chunks([chunk | acc], {buffer, buffer_size}, chunk_length, step)
  end

  defp slice_buffer(buffer, 0, acc), do: {Enum.reverse(acc), buffer}

  defp slice_buffer([chunk | buffer], size, acc) do
    chunk_size = Nx.size(chunk)

    if chunk_size <= size do
      slice_buffer(buffer, size - chunk_size, [chunk | acc])
    else
      {chunk, rest} = Nx.split(chunk, size)
      slice_buffer([rest | buffer], 0, [chunk | acc])
    end
  end
end
